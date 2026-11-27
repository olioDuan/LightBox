"""API key resolution, endpoint binding, keyring readback, and atomic chmod-600 `.env` storage (SPEC §6.1, §10.2(6))."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

import keyring

from lightbox.config_io import atomic_update_text
from lightbox.logging import get_logger
from lightbox.models.types import EndpointBindingError, MissingAPIKeyError
from lightbox.paths import AppPaths, get_app_paths

logger = get_logger("lightbox.models.secrets")

KEYRING_SERVICE = "lightbox"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

SecretSource = Literal["keyring", "env", "dotenv", "none"]

PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "local": ("LIGHTBOX_LOCAL_API_KEY",),
    "custom": ("LIGHTBOX_CUSTOM_API_KEY",),
}

OFFICIAL_DEFAULT_ENDPOINTS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


@dataclass(frozen=True, slots=True)
class ResolvedSecret:
    """Resolved API key together with its credential source and bound endpoint."""

    api_key: str
    source: SecretSource
    bound_endpoint: str | None
    key_suffix: str


@dataclass(frozen=True, slots=True)
class SecretInspection:
    """Non-secret inspection snapshot of the effective credential for a provider + endpoint."""

    has_key: bool
    source: SecretSource
    bound_endpoint: str | None
    key_suffix: str | None
    endpoint_mismatch: bool = False
    mismatch_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SavedSecretResult:
    """Metadata returned after persisting an endpoint-bound API key."""

    storage: Literal["keyring", "dotenv"]
    bound_endpoint: str
    key_suffix: str


def mask_api_key_suffix(api_key: str) -> str:
    """Return a non-sensitive masked suffix (last 4 characters only) for UI/fingerprinting."""
    cleaned = api_key.strip()
    if not cleaned or cleaned == "local-no-key":
        return ""
    if len(cleaned) <= 4:
        return "*" * len(cleaned)
    return cleaned[-4:]


def _canonicalize_http_url(raw_url: str) -> str:
    cleaned = raw_url.strip()
    if not cleaned:
        raise ValueError("Endpoint URL must not be empty.")
    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported endpoint URL scheme {parsed.scheme!r}; only http/https allowed.")
    if not parsed.netloc:
        raise ValueError(f"Invalid endpoint URL (missing host): {raw_url!r}")
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunparse((scheme, host, path, "", "", ""))


def normalize_ollama_root_url(base_url: str | None = None) -> str:
    """Normalize an Ollama base URL to its service root (without trailing `/v1` or `/`)."""
    raw = (base_url or DEFAULT_OLLAMA_BASE_URL).strip() or DEFAULT_OLLAMA_BASE_URL
    canon = _canonicalize_http_url(raw)
    if canon.endswith("/v1"):
        canon = canon[:-3].rstrip("/")
    return canon


def normalize_ollama_v1_url(base_url: str | None = None) -> str:
    """Normalize an Ollama base URL to its OpenAI-compatible `/v1` endpoint."""
    return f"{normalize_ollama_root_url(base_url)}/v1"


def normalize_endpoint(provider: str, base_url: str | None = None) -> str:
    """Normalize the target API endpoint for a provider."""
    raw = (base_url or "").strip()
    if provider == "ollama":
        return normalize_ollama_v1_url(raw or None)
    if not raw:
        if provider in OFFICIAL_DEFAULT_ENDPOINTS:
            return OFFICIAL_DEFAULT_ENDPOINTS[provider]
        if provider == "local":
            return "http://127.0.0.1:8080/v1"
        raise ValueError(f"Provider {provider!r} requires an explicit base_url.")
    return _canonicalize_http_url(raw)


def is_official_endpoint(provider: str, endpoint: str) -> bool:
    """Return True iff `endpoint` matches the canonical official endpoint for `provider`."""
    official = OFFICIAL_DEFAULT_ENDPOINTS.get(provider)
    if official is None:
        return False
    try:
        return normalize_endpoint(provider, endpoint) == official
    except ValueError:
        return False


def bound_keyring_account(var_name: str, normalized_endpoint: str) -> str:
    """Encode the normalized endpoint directly into the keyring account identifier."""
    return f"{var_name}::{normalized_endpoint}"


def _meta_keyring_account(var_name: str) -> str:
    return f"{var_name}::__bound_endpoint"


def _endpoint_dotenv_var(var_name: str) -> str:
    return f"{var_name}_ENDPOINT"


def compute_connection_fingerprint(
    *,
    provider: str,
    base_url: str | None,
    agent_model: str,
    triage_model: str,
    key_suffix: str | None,
) -> str:
    """Compute a stable SHA-256 connection fingerprint (never embeds any full API key)."""
    norm_ep = normalize_endpoint(provider, base_url)
    payload = {
        "provider": provider,
        "base_url": norm_ep,
        "agent_model": agent_model.strip(),
        "triage_model": triage_model.strip(),
        "key_suffix": (key_suffix or "").strip(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"fp_{hashlib.sha256(raw).hexdigest()[:24]}"


def _read_from_keyring(account_name: str) -> str | None:
    try:
        val = keyring.get_password(KEYRING_SERVICE, account_name)
    except Exception:
        return None
    if val is not None and val.strip():
        return val.strip()
    return None


def _write_to_keyring(account_name: str, value: str) -> None:
    keyring.set_password(KEYRING_SERVICE, account_name, value)


def _delete_from_keyring(account_name: str) -> None:
    try:
        keyring.delete_password(KEYRING_SERVICE, account_name)
    except Exception:
        pass


def _parse_dotenv(dotenv_path: Path) -> dict[str, str]:
    if not dotenv_path.is_file():
        return {}
    if os.name == "posix":
        mode = stat.S_IMODE(dotenv_path.stat().st_mode)
        if mode & 0o077:
            raise PermissionError(
                f"Insecure permissions ({oct(mode)}) on {dotenv_path}; run `chmod 600 {dotenv_path}`."
            )

    parsed: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and value:
            parsed[key] = value
    return parsed


def _mutate_dotenv_lines(
    existing_text: str | None,
    *,
    set_vars: dict[str, str] | None = None,
    remove_vars: set[str] | None = None,
) -> str:
    to_set = dict(set_vars or {})
    to_remove = set(remove_vars or set())
    existing_lines = existing_text.splitlines() if existing_text is not None else []
    updated_lines: list[str] = []
    seen_set: set[str] = set()

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            raw_k, _ = stripped.split("=", 1)
            k = raw_k[7:].strip() if raw_k.strip().startswith("export ") else raw_k.strip()
            if k in to_remove and k not in to_set:
                continue
            if k in to_set:
                if k not in seen_set:
                    updated_lines.append(f'{k}="{to_set[k]}"')
                    seen_set.add(k)
                continue
        updated_lines.append(line)

    for k, val in to_set.items():
        if k not in seen_set:
            updated_lines.append(f'{k}="{val}"')

    return ("\n".join(updated_lines) + "\n") if updated_lines else ""


def resolve_api_key_with_source(
    provider: str,
    app_paths: AppPaths | None = None,
    *,
    base_url: str | None = None,
) -> ResolvedSecret:
    """Resolve provider API key with strict endpoint-binding checks: keyring -> env -> `.env`.

    Unbound environment variables and legacy unbound keys are strictly limited to the
    provider's canonical official endpoint. Never logs or embeds secret values in errors.
    """
    mapped_provider = "local" if provider == "ollama" else provider
    var_names = PROVIDER_ENV_VARS.get(mapped_provider)
    if var_names is None:
        raise MissingAPIKeyError(f"Unknown model provider: {provider!r}")

    primary_var = var_names[0]
    target_ep = normalize_endpoint(provider, base_url)
    official_ok = is_official_endpoint(mapped_provider, target_ep)
    mismatch_bound_ep: str | None = None

    # 1. System keyring: exact endpoint-bound account first
    bound_val = _read_from_keyring(bound_keyring_account(primary_var, target_ep))
    if bound_val:
        return ResolvedSecret(
            api_key=bound_val,
            source="keyring",
            bound_endpoint=target_ep,
            key_suffix=mask_api_key_suffix(bound_val),
        )

    # Check whether keyring holds a key bound to a DIFFERENT endpoint
    recorded_kr_ep = _read_from_keyring(_meta_keyring_account(primary_var))
    if recorded_kr_ep and recorded_kr_ep != target_ep:
        if _read_from_keyring(bound_keyring_account(primary_var, recorded_kr_ep)):
            mismatch_bound_ep = recorded_kr_ep

    # Legacy unbound keyring entries (only allowed for official default endpoints)
    for var in var_names:
        legacy_kr = _read_from_keyring(var)
        if legacy_kr:
            if official_ok or mapped_provider == "local":
                return ResolvedSecret(
                    api_key=legacy_kr,
                    source="keyring",
                    bound_endpoint=OFFICIAL_DEFAULT_ENDPOINTS.get(mapped_provider),
                    key_suffix=mask_api_key_suffix(legacy_kr),
                )
            mismatch_bound_ep = OFFICIAL_DEFAULT_ENDPOINTS.get(mapped_provider) or "official"
            break

    # 2. Environment variables (no endpoint binding -> strictly official endpoints only)
    for var in var_names:
        from_env = os.environ.get(var, "").strip()
        if from_env:
            if official_ok or mapped_provider == "local":
                return ResolvedSecret(
                    api_key=from_env,
                    source="env",
                    bound_endpoint=OFFICIAL_DEFAULT_ENDPOINTS.get(mapped_provider),
                    key_suffix=mask_api_key_suffix(from_env),
                )
            if mismatch_bound_ep is None:
                mismatch_bound_ep = OFFICIAL_DEFAULT_ENDPOINTS.get(mapped_provider) or "official"
            break

    # 3. chmod-600 `.env` under <APP_DATA_DIR>
    resolved_paths = app_paths or get_app_paths()
    dotenv_values = _parse_dotenv(resolved_paths.root / ".env")
    dotenv_ep_raw = dotenv_values.get(_endpoint_dotenv_var(primary_var), "").strip()
    dotenv_ep: str | None = None
    if dotenv_ep_raw:
        try:
            dotenv_ep = normalize_endpoint(provider, dotenv_ep_raw)
        except ValueError:
            dotenv_ep = dotenv_ep_raw

    for var in var_names:
        from_file = dotenv_values.get(var, "").strip()
        if not from_file:
            continue
        if dotenv_ep is not None:
            if dotenv_ep == target_ep:
                return ResolvedSecret(
                    api_key=from_file,
                    source="dotenv",
                    bound_endpoint=dotenv_ep,
                    key_suffix=mask_api_key_suffix(from_file),
                )
            mismatch_bound_ep = dotenv_ep
            break
        if official_ok or mapped_provider == "local":
            return ResolvedSecret(
                api_key=from_file,
                source="dotenv",
                bound_endpoint=OFFICIAL_DEFAULT_ENDPOINTS.get(mapped_provider),
                key_suffix=mask_api_key_suffix(from_file),
            )
        mismatch_bound_ep = OFFICIAL_DEFAULT_ENDPOINTS.get(mapped_provider) or "official"
        break

    if mapped_provider == "local":
        return ResolvedSecret(
            api_key="local-no-key",
            source="none",
            bound_endpoint=target_ep,
            key_suffix="",
        )

    if mismatch_bound_ep is not None:
        raise EndpointBindingError(
            f"Stored API key for {provider!r} is bound to {mismatch_bound_ep!r}, "
            f"not {target_ep!r}. Re-enter the API key to bind it to the new endpoint.",
            expected_endpoint=target_ep,
            bound_endpoint=mismatch_bound_ep,
        )

    expected = " or ".join(var_names)
    raise MissingAPIKeyError(
        f"Missing API key for provider {provider!r}. Set {expected} in system keyring, "
        f"environment, or chmod 600 {resolved_paths.root / '.env'}."
    )


def resolve_api_key(
    provider: str,
    app_paths: AppPaths | None = None,
    *,
    base_url: str | None = None,
) -> str:
    """Resolve provider API key strictly in order: keyring -> env -> `<APP_DATA_DIR>/.env`."""
    return resolve_api_key_with_source(provider, app_paths=app_paths, base_url=base_url).api_key


def inspect_provider_secret(
    provider: str,
    app_paths: AppPaths | None = None,
    *,
    base_url: str | None = None,
) -> SecretInspection:
    """Inspect credential presence, source, suffix, and endpoint binding without exposing secrets."""
    try:
        resolved = resolve_api_key_with_source(provider, app_paths=app_paths, base_url=base_url)
        has_key = resolved.source != "none" or provider in {"local", "ollama"}
        return SecretInspection(
            has_key=has_key,
            source=resolved.source,
            bound_endpoint=resolved.bound_endpoint,
            key_suffix=resolved.key_suffix or None,
            endpoint_mismatch=False,
            mismatch_reason=None,
        )
    except EndpointBindingError as exc:
        return SecretInspection(
            has_key=False,
            source="none",
            bound_endpoint=exc.bound_endpoint,
            key_suffix=None,
            endpoint_mismatch=True,
            mismatch_reason=str(exc),
        )
    except (MissingAPIKeyError, PermissionError, ValueError):
        return SecretInspection(
            has_key=False,
            source="none",
            bound_endpoint=None,
            key_suffix=None,
            endpoint_mismatch=False,
            mismatch_reason=None,
        )


def has_api_key(
    provider: str,
    app_paths: AppPaths | None = None,
    *,
    base_url: str | None = None,
) -> bool:
    """Return True if valid endpoint-bound credentials are available (local providers always return True)."""
    try:
        resolve_api_key(provider, app_paths=app_paths, base_url=base_url)
        return True
    except (MissingAPIKeyError, EndpointBindingError, PermissionError, ValueError):
        return False


def save_secret_to_dotenv(
    key_name: str,
    value: str,
    app_paths: AppPaths,
    *,
    endpoint: str | None = None,
) -> Path:
    """Atomically persist a secret (and optional endpoint binding line) into `0o600 <APP_DATA_DIR>/.env`."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Secret value must not be empty.")
    dotenv_path = app_paths.root / ".env"
    set_vars = {key_name: cleaned}
    remove_vars: set[str] = set()
    ep_var = _endpoint_dotenv_var(key_name)
    if endpoint is not None:
        set_vars[ep_var] = endpoint.strip()
    else:
        remove_vars.add(ep_var)

    return atomic_update_text(
        app_paths,
        dotenv_path,
        lambda existing: _mutate_dotenv_lines(
            existing,
            set_vars=set_vars,
            remove_vars=remove_vars,
        ),
        mode=0o600,
    )


def _endpoint_bindings_file(app_paths: AppPaths) -> Path:
    return app_paths.root / ".endpoint_bindings.json"


def _read_local_bound_endpoint(app_paths: AppPaths, primary_var: str) -> str | None:
    path = _endpoint_bindings_file(app_paths)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            val = raw.get(primary_var)
            if isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        return None
    return None


def _write_local_bound_endpoint(app_paths: AppPaths, primary_var: str, endpoint: str | None) -> None:
    path = _endpoint_bindings_file(app_paths)

    def _transform(existing: str | None) -> str:
        data: dict[str, str] = {}
        if existing:
            try:
                parsed = json.loads(existing)
                if isinstance(parsed, dict):
                    data = {str(k): str(v) for k, v in parsed.items() if isinstance(v, str)}
            except Exception:
                data = {}
        if endpoint is None:
            data.pop(primary_var, None)
        else:
            data[primary_var] = endpoint
        return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    atomic_update_text(app_paths, path, _transform, mode=0o600)


def save_api_key_bound(
    provider: str,
    api_key: str,
    app_paths: AppPaths,
    *,
    base_url: str | None = None,
    previous_base_url: str | None = None,
) -> SavedSecretResult:
    """Persist an endpoint-bound API key to keyring (with readback) or fallback `0o600 .env`.

    When rebinding to a different endpoint, cleans up old endpoint entries in both keyring
    and `.env` so stale credentials never linger.
    """
    cleaned = api_key.strip()
    if not cleaned:
        raise ValueError("API key must not be empty.")

    mapped_provider = "local" if provider == "ollama" else provider
    var_names = PROVIDER_ENV_VARS.get(mapped_provider)
    if var_names is None:
        raise ValueError(f"Unknown model provider: {provider!r}")

    primary_var = var_names[0]
    norm_ep = normalize_endpoint(provider, base_url)
    bound_acct = bound_keyring_account(primary_var, norm_ep)
    meta_acct = _meta_keyring_account(primary_var)

    # Collect old bound endpoints to purge upon rebinding (from keyring meta, local backup index, prev arg, and .env)
    old_endpoints: set[str] = set()
    for candidate_ep in (
        _read_from_keyring(meta_acct),
        _read_local_bound_endpoint(app_paths, primary_var),
    ):
        if candidate_ep and candidate_ep != norm_ep:
            old_endpoints.add(candidate_ep)
    if previous_base_url:
        try:
            prev_ep = normalize_endpoint(provider, previous_base_url)
            if prev_ep != norm_ep:
                old_endpoints.add(prev_ep)
        except ValueError:
            pass

    dotenv_path = app_paths.root / ".env"
    if dotenv_path.is_file():
        try:
            parsed_env = _parse_dotenv(dotenv_path)
            prev_dotenv_ep = parsed_env.get(_endpoint_dotenv_var(primary_var), "").strip()
            if prev_dotenv_ep and prev_dotenv_ep != norm_ep:
                old_endpoints.add(prev_dotenv_ep)
        except PermissionError:
            pass

    # 1. Try system keyring with mandatory readback verification
    keyring_ok = False
    try:
        _write_to_keyring(bound_acct, cleaned)
        readback = _read_from_keyring(bound_acct)
        if readback == cleaned:
            keyring_ok = True
    except Exception:
        keyring_ok = False

    if keyring_ok:
        try:
            _write_to_keyring(meta_acct, norm_ep)
        except Exception as exc:
            logger.warning(
                "keyring_bound_endpoint_meta_write_failed",
                provider=provider,
                error=type(exc).__name__,
            )
        # Always persist non-secret bound endpoint index locally so future rebinding cleans old keyring entries
        _write_local_bound_endpoint(app_paths, primary_var, norm_ep)
        # Clean up old bound endpoints and any legacy unbound keyring entries
        for old_ep in old_endpoints:
            _delete_from_keyring(bound_keyring_account(primary_var, old_ep))
        for var in var_names:
            _delete_from_keyring(var)

        # Remove same-provider stale entries and binding lines from `.env` atomically
        if dotenv_path.is_file():
            remove_keys = set(var_names) | {_endpoint_dotenv_var(primary_var)}
            atomic_update_text(
                app_paths,
                dotenv_path,
                lambda existing: _mutate_dotenv_lines(existing, remove_vars=remove_keys),
                mode=0o600,
            )
        return SavedSecretResult(
            storage="keyring",
            bound_endpoint=norm_ep,
            key_suffix=mask_api_key_suffix(cleaned),
        )

    # 2. Keyring failed or readback mismatch -> smooth fallback to atomic `0o600 .env`
    for old_ep in old_endpoints:
        _delete_from_keyring(bound_keyring_account(primary_var, old_ep))
    _delete_from_keyring(bound_acct)
    _delete_from_keyring(meta_acct)
    for var in var_names:
        _delete_from_keyring(var)
    _write_local_bound_endpoint(app_paths, primary_var, norm_ep)

    remove_aliases = set(var_names[1:])
    atomic_update_text(
        app_paths,
        dotenv_path,
        lambda existing: _mutate_dotenv_lines(
            existing,
            set_vars={
                primary_var: cleaned,
                _endpoint_dotenv_var(primary_var): norm_ep,
            },
            remove_vars=remove_aliases,
        ),
        mode=0o600,
    )
    return SavedSecretResult(
        storage="dotenv",
        bound_endpoint=norm_ep,
        key_suffix=mask_api_key_suffix(cleaned),
    )


def delete_provider_secret(
    provider: str,
    app_paths: AppPaths,
    *,
    base_url: str | None = None,
) -> bool:
    """Delete persisted keyring and `.env` credentials for `provider` (never mutates OS env).

    Returns True if any stored keyring or `.env` entry was cleaned up.
    """
    mapped_provider = "local" if provider == "ollama" else provider
    var_names = PROVIDER_ENV_VARS.get(mapped_provider)
    if var_names is None:
        return False

    primary_var = var_names[0]
    meta_acct = _meta_keyring_account(primary_var)
    deleted_any = False

    endpoints_to_clear: set[str] = set()
    recorded_kr_ep = _read_from_keyring(meta_acct)
    local_recorded_ep = _read_local_bound_endpoint(app_paths, primary_var)
    if recorded_kr_ep:
        endpoints_to_clear.add(recorded_kr_ep)
    if local_recorded_ep:
        endpoints_to_clear.add(local_recorded_ep)
    if base_url:
        try:
            endpoints_to_clear.add(normalize_endpoint(provider, base_url))
        except ValueError:
            pass
    elif mapped_provider in OFFICIAL_DEFAULT_ENDPOINTS:
        endpoints_to_clear.add(OFFICIAL_DEFAULT_ENDPOINTS[mapped_provider])

    for ep in endpoints_to_clear:
        acct = bound_keyring_account(primary_var, ep)
        if _read_from_keyring(acct) is not None:
            deleted_any = True
        _delete_from_keyring(acct)

    if recorded_kr_ep is not None:
        _delete_from_keyring(meta_acct)
    if local_recorded_ep is not None:
        _write_local_bound_endpoint(app_paths, primary_var, None)

    for var in var_names:
        if _read_from_keyring(var) is not None:
            deleted_any = True
        _delete_from_keyring(var)

    dotenv_path = app_paths.root / ".env"
    if dotenv_path.is_file():
        remove_keys = set(var_names) | {_endpoint_dotenv_var(primary_var)}
        try:
            existing_parsed = _parse_dotenv(dotenv_path)
            if any(k in existing_parsed for k in remove_keys):
                deleted_any = True
        except PermissionError:
            pass
        atomic_update_text(
            app_paths,
            dotenv_path,
            lambda existing: _mutate_dotenv_lines(existing, remove_vars=remove_keys),
            mode=0o600,
        )

    return deleted_any


__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "KEYRING_SERVICE",
    "OFFICIAL_DEFAULT_ENDPOINTS",
    "PROVIDER_ENV_VARS",
    "ResolvedSecret",
    "SavedSecretResult",
    "SecretInspection",
    "SecretSource",
    "bound_keyring_account",
    "compute_connection_fingerprint",
    "delete_provider_secret",
    "has_api_key",
    "inspect_provider_secret",
    "is_official_endpoint",
    "mask_api_key_suffix",
    "normalize_endpoint",
    "normalize_ollama_root_url",
    "normalize_ollama_v1_url",
    "resolve_api_key",
    "resolve_api_key_with_source",
    "save_api_key_bound",
    "save_secret_to_dotenv",
]
