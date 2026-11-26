"""API key resolution: keyring -> environment -> chmod-600 `.env` (SPEC §6.1, §10.2(6))."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import keyring

from lightbox.models.types import MissingAPIKeyError
from lightbox.paths import AppPaths, get_app_paths

KEYRING_SERVICE = "lightbox"

PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "local": ("LIGHTBOX_LOCAL_API_KEY",),
}


def _read_from_keyring(var_name: str) -> str | None:
    try:
        val = keyring.get_password(KEYRING_SERVICE, var_name)
    except Exception:
        return None
    if val is not None and val.strip():
        return val.strip()
    return None


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


def resolve_api_key(provider: str, app_paths: AppPaths | None = None) -> str:
    """Resolve provider API key strictly in order: keyring -> env -> `<APP_DATA_DIR>/.env`.

    Never logs or embeds any secret value in error messages.
    """
    var_names = PROVIDER_ENV_VARS.get(provider)
    if var_names is None:
        raise MissingAPIKeyError(f"Unknown model provider: {provider!r}")

    # 1. System keyring
    for var in var_names:
        from_kr = _read_from_keyring(var)
        if from_kr:
            return from_kr

    # 2. Environment variables
    for var in var_names:
        from_env = os.environ.get(var, "").strip()
        if from_env:
            return from_env

    # 3. chmod-600 `.env` under <APP_DATA_DIR>
    resolved_paths = app_paths or get_app_paths()
    dotenv_values = _parse_dotenv(resolved_paths.root / ".env")
    for var in var_names:
        from_file = dotenv_values.get(var, "").strip()
        if from_file:
            return from_file

    if provider == "local":
        return "local-no-key"

    expected = " or ".join(var_names)
    raise MissingAPIKeyError(
        f"Missing API key for provider {provider!r}. Set {expected} in system keyring, "
        f"environment, or chmod 600 {resolved_paths.root / '.env'}."
    )


def has_api_key(provider: str, app_paths: AppPaths | None = None) -> bool:
    """Return True if credentials are available (local providers always return True)."""
    try:
        resolve_api_key(provider, app_paths=app_paths)
        return True
    except (MissingAPIKeyError, PermissionError):
        return False


def save_secret_to_dotenv(key_name: str, value: str, app_paths: AppPaths) -> Path:
    """Persist a secret into `<APP_DATA_DIR>/.env` with `0o600` permissions."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Secret value must not be empty.")
    app_paths.ensure_dirs()
    dotenv_path = app_paths.root / ".env"

    existing_lines: list[str] = []
    if dotenv_path.is_file():
        existing_lines = dotenv_path.read_text(encoding="utf-8").splitlines()

    updated_lines: list[str] = []
    replaced = False
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _ = stripped.split("=", 1)
            if k.strip() == key_name:
                updated_lines.append(f'{key_name}="{cleaned}"')
                replaced = True
                continue
        updated_lines.append(line)
    if not replaced:
        updated_lines.append(f'{key_name}="{cleaned}"')

    content = "\n".join(updated_lines) + "\n"
    if os.name == "posix":
        fd = os.open(dotenv_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    else:
        dotenv_path.write_text(content, encoding="utf-8")
    return dotenv_path


__all__ = [
    "KEYRING_SERVICE",
    "PROVIDER_ENV_VARS",
    "has_api_key",
    "resolve_api_key",
    "save_secret_to_dotenv",
]
