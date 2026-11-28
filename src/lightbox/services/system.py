"""SystemService: environment health checks for `lightbox doctor` (SPEC §2.4, §12)."""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
import sys
import tomllib
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightbox.config import DEFAULT_CONFIG_TOML, AIProviderName, AppConfig, load_config
from lightbox.config_io import atomic_update_text
from lightbox.core import find_exiftool, find_face_model_path
from lightbox.jobs import JobRunner
from lightbox.models import (
    find_face_embedder_model_path,
    find_siglip2_text_path,
    find_siglip2_vision_path,
)
from lightbox.paths import AppPaths
from lightbox.services.ai_connection import (
    MCPClientId,
    ProbeAdapterFactory,
    delete_ai_connection,
    detect_mcp_clients,
    discover_ollama_models,
    generate_mcp_snippet,
    get_ai_connection_status,
    save_ai_connection,
    test_ai_connection,
)
from lightbox.services.app_installer import (
    CommandRunner,
    InstallAppResult,
    install_desktop_app,
)
from lightbox.store import Database, LightboxRepository


@dataclass(frozen=True, slots=True)
class DoctorCheckItem:
    name: str
    status: str  # 'ok' | 'warn' | 'fail'
    detail: str


@dataclass(frozen=True, slots=True)
class DoctorReport:
    overall_ok: bool
    checks: tuple[DoctorCheckItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_ok": self.overall_ok,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }


class SystemService:
    """System diagnostics backing `lightbox doctor` (zero secret values exposed)."""

    def __init__(self, app_paths: AppPaths, db: Database) -> None:
        self.app_paths = app_paths
        self.db = db

    def run_doctor(self) -> DoctorReport:
        checks: list[DoctorCheckItem] = []

        # 1. Python runtime
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        checks.append(
            DoctorCheckItem(
                name="python",
                status="ok" if sys.version_info >= (3, 12) else "fail",
                detail=f"Python {py_ver} ({platform.system()} {platform.machine()})",
            )
        )

        # 2. SQLite + sqlite-vec extension
        vec_status = "ok" if self.db.sqlite_vec_available else "warn"
        vec_detail = (
            f"SQLite {sqlite3.sqlite_version}; sqlite-vec loaded"
            if self.db.sqlite_vec_available
            else f"SQLite {sqlite3.sqlite_version}; sqlite-vec unavailable (numpy fallback active)"
        )
        checks.append(DoctorCheckItem(name="sqlite_vec", status=vec_status, detail=vec_detail))

        # 3. App data dir writability
        try:
            self.app_paths.ensure_dirs()
            writable = os.access(self.app_paths.root, os.W_OK)
            checks.append(
                DoctorCheckItem(
                    name="app_data_dir",
                    status="ok" if writable else "fail",
                    detail=f"{self.app_paths.root} ({'writable' if writable else 'read-only'})",
                )
            )
        except OSError as exc:
            checks.append(
                DoctorCheckItem(
                    name="app_data_dir",
                    status="fail",
                    detail=f"Cannot initialize {self.app_paths.root}: {exc}",
                )
            )

        # 4. ExifTool availability
        exiftool_path = find_exiftool(self.app_paths)
        if exiftool_path is not None:
            checks.append(
                DoctorCheckItem(
                    name="exiftool",
                    status="ok",
                    detail=f"Found at {exiftool_path}",
                )
            )
        else:
            checks.append(
                DoctorCheckItem(
                    name="exiftool",
                    status="warn",
                    detail="Not found in PATH or <APP_DATA_DIR>/bin (fallback Pillow/rawpy/exifread active)",
                )
            )

        # 5. ONNX Runtime & InsightFace model cache
        try:
            import onnxruntime as ort

            providers = ", ".join(ort.get_available_providers())
            checks.append(
                DoctorCheckItem(
                    name="onnxruntime",
                    status="ok",
                    detail=f"onnxruntime {ort.__version__} ({providers})",
                )
            )
        except Exception as exc:
            checks.append(
                DoctorCheckItem(name="onnxruntime", status="warn", detail=f"Unavailable: {exc}")
            )

        face_model = find_face_model_path(self.app_paths)
        rec_model = find_face_embedder_model_path(self.app_paths)
        if face_model is not None and rec_model is not None:
            checks.append(
                DoctorCheckItem(
                    name="insightface_model",
                    status="ok",
                    detail=f"{face_model.name} + {rec_model.name} cached under {face_model.parent}",
                )
            )
        elif face_model is not None:
            checks.append(
                DoctorCheckItem(
                    name="insightface_model",
                    status="ok",
                    detail=f"{face_model.name} cached at {face_model} (glintr100.onnx lazy-downloads on people index)",
                )
            )
        else:
            checks.append(
                DoctorCheckItem(
                    name="insightface_model",
                    status="warn",
                    detail=f"antelopev2 models not yet downloaded under {self.app_paths.models_dir}",
                )
            )

        siglip_vis = find_siglip2_vision_path(self.app_paths)
        siglip_txt, siglip_tok = find_siglip2_text_path(self.app_paths)
        if siglip_vis is not None and siglip_txt is not None and siglip_tok is not None:
            checks.append(
                DoctorCheckItem(
                    name="siglip2_model",
                    status="ok",
                    detail=f"SigLIP 2 vision + text ONNX cached under {siglip_vis.parent.parent}",
                )
            )
        elif siglip_vis is not None:
            checks.append(
                DoctorCheckItem(
                    name="siglip2_model",
                    status="ok",
                    detail=f"SigLIP 2 vision ONNX cached at {siglip_vis} (text tower lazy-loads on search)",
                )
            )
        else:
            checks.append(
                DoctorCheckItem(
                    name="siglip2_model",
                    status="warn",
                    detail=f"SigLIP 2 ONNX not yet downloaded under {self.app_paths.models_dir} (lazy downloads on --embed / search)",
                )
            )

        # 6. LLM API key presence (boolean only; NEVER expose values!)
        key_vars = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
        configured = [k for k in key_vars if os.environ.get(k, "").strip()]
        if configured:
            checks.append(
                DoctorCheckItem(
                    name="api_keys",
                    status="ok",
                    detail=f"Env keys configured: {', '.join(configured)}",
                )
            )
        else:
            checks.append(
                DoctorCheckItem(
                    name="api_keys",
                    status="ok",
                    detail="Zero-LLM offline mode (no cloud API keys set)",
                )
            )

        overall_ok = all(c.status != "fail" for c in checks)
        return DoctorReport(overall_ok=overall_ok, checks=tuple(checks))

    def get_settings(self) -> AppConfig:
        """Load current validated settings from `<APP_DATA_DIR>/config.toml`."""
        return load_config(self.app_paths)

    def update_settings(self, updates: dict[str, Any]) -> AppConfig:
        """Validate and atomically persist updated settings inside the config_io lock (forbids API keys & `ai`)."""
        if "ai" in updates:
            raise ValueError("AI connection settings must be managed via /api/ai/connection.")

        validated_holder: list[AppConfig] = []

        def _transform(existing_text: str | None) -> str:
            base_raw = (
                tomllib.loads(existing_text)
                if existing_text is not None and existing_text.strip()
                else tomllib.loads(DEFAULT_CONFIG_TOML)
            )
            current = AppConfig.model_validate(base_raw).model_dump()
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(current.get(key), dict):
                    current[key] = {**current[key], **value}
                else:
                    current[key] = value
            validated = AppConfig.model_validate(current)
            validated_holder.clear()
            validated_holder.append(validated)
            return _dump_config_toml(validated)

        atomic_update_text(self.app_paths, self.app_paths.config_file, _transform)
        return validated_holder[0]

    # --- AI Connection, Probes, Ollama Discovery & Read-only MCP Clients ---

    def get_ai_connection(self) -> dict[str, Any]:
        """Return current AI connection status derived locally from fingerprint + SQLite calls."""
        return get_ai_connection_status(self.app_paths, self.db)

    def save_ai_connection(
        self,
        *,
        provider: AIProviderName,
        agent_model: str | None = None,
        triage_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reviewer: str = "none",
        is_local_free: bool = False,
        input_cost_per_mtok: float | None = None,
        output_cost_per_mtok: float | None = None,
        triage_input_cost_per_mtok: float | None = None,
        triage_output_cost_per_mtok: float | None = None,
        max_session_calls: int = 20,
        max_triage_photos_unpriced: int = 20,
    ) -> dict[str, Any]:
        """Atomically persist endpoint-bound secret (first) and `[ai]` section in `config.toml` (second)."""
        return save_ai_connection(
            self.app_paths,
            self.db,
            dump_config_fn=_dump_config_toml,
            provider=provider,
            agent_model=agent_model,
            triage_model=triage_model,
            base_url=base_url,
            api_key=api_key,
            reviewer=reviewer,
            is_local_free=is_local_free,
            input_cost_per_mtok=input_cost_per_mtok,
            output_cost_per_mtok=output_cost_per_mtok,
            triage_input_cost_per_mtok=triage_input_cost_per_mtok,
            triage_output_cost_per_mtok=triage_output_cost_per_mtok,
            max_session_calls=max_session_calls,
            max_triage_photos_unpriced=max_triage_photos_unpriced,
        )

    def delete_ai_connection(self, *, delete_credential: bool = True) -> dict[str, Any]:
        """Deactivate `[ai]`, delete non-env stored credentials, and re-resolve remaining source."""
        return delete_ai_connection(
            self.app_paths,
            self.db,
            dump_config_fn=_dump_config_toml,
            delete_credential=delete_credential,
        )

    async def test_ai_connection(
        self,
        *,
        provider: AIProviderName | None = None,
        agent_model: str | None = None,
        triage_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        is_local_free: bool | None = None,
        adapter_factory: ProbeAdapterFactory | None = None,
        http_get_json: Callable[[str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run verifiable capability probes against Agent and Triage models and update fingerprint cache."""
        return await test_ai_connection(
            self.app_paths,
            self.db,
            provider=provider,
            agent_model=agent_model,
            triage_model=triage_model,
            base_url=base_url,
            api_key=api_key,
            is_local_free=is_local_free,
            adapter_factory=adapter_factory,
            http_get_json=http_get_json,
        )

    def discover_ollama_models(
        self,
        *,
        base_url: str | None = None,
        http_get_json: Callable[[str], dict[str, Any]] | None = None,
        http_post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """List local Ollama models and theoretical context limits from `/api/tags` + `/api/show`."""
        return discover_ollama_models(
            self.app_paths,
            base_url=base_url,
            http_get_json=http_get_json,
            http_post_json=http_post_json,
        )

    def detect_mcp_clients(
        self,
        *,
        home_dir: Path | None = None,
        platform_name: str | None = None,
    ) -> dict[str, Any]:
        """Read-only inspection of external MCP client configuration status (never returns raw config)."""
        return detect_mcp_clients(home_dir=home_dir, platform_name=platform_name)

    def generate_mcp_snippet(
        self,
        client: MCPClientId,
        *,
        home_dir: Path | None = None,
        platform_name: str | None = None,
    ) -> dict[str, Any]:
        """Generate copy-paste MCP configuration snippet for a validated client enum."""
        return generate_mcp_snippet(client, home_dir=home_dir, platform_name=platform_name)

    def install_desktop_app(
        self,
        *,
        include_desktop: bool = False,
        target_dir: Path | None = None,
        home_dir: Path | None = None,
        platform_name: str | None = None,
        python_executable: Path | None = None,
        command_runner: CommandRunner | None = None,
    ) -> InstallAppResult:
        """Install native OS desktop launcher (macOS `.app`, Windows `.lnk`, Linux `.desktop`)."""
        return install_desktop_app(
            self.app_paths,
            include_desktop=include_desktop,
            target_dir=target_dir,
            home_dir=home_dir,
            platform_name=platform_name,
            python_executable=python_executable,
            command_runner=command_runner,
        )

    def pick_folder(self) -> str | None:
        """Invoke native OS folder picker dialog; returns selected path or None if canceled."""
        if sys.platform == "darwin":
            script = 'POSIX path of (choose folder with prompt "Select photo folder for Lightbox")'
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            if proc.returncode == 0:
                chosen = proc.stdout.strip()
                return chosen.rstrip("/") or "/" if chosen else None
            return None

        # Cross-platform fallback via tkinter when available
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                chosen = filedialog.askdirectory(title="Select photo folder for Lightbox")
            finally:
                root.destroy()
            return chosen or None
        except Exception:
            return None

    def open_folder(self, folder: Path) -> bool:
        """Open an existing local directory in the OS file manager (Finder / Explorer / xdg-open)."""
        target = folder.expanduser().resolve()
        if not target.is_dir():
            return False
        if sys.platform == "darwin":
            proc = subprocess.run(["open", str(target)], check=False, timeout=15)
            return proc.returncode == 0
        if sys.platform == "win32":
            os.startfile(str(target))  # type: ignore[attr-defined]
            return True
        proc = subprocess.run(["xdg-open", str(target)], check=False, timeout=15)
        return proc.returncode == 0

    def reconcile_orphaned_jobs(
        self,
        *,
        current_instance_id: str | None = None,
        live_instance_ids: Collection[str] | None = None,
    ) -> list[str]:
        """Heal leftover `running`/`queued` jobs whose owning OS process has exited or recycled PID."""
        runner = JobRunner(LightboxRepository(self.db))
        return runner.reconcile_orphaned_jobs(
            current_instance_id=current_instance_id,
            live_instance_ids=live_instance_ids,
        )

    def record_audit_log(
        self,
        *,
        surface: str,
        tool_name: str,
        hook_decision: str,
        args_redacted: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        summary: str | None = None,
    ) -> int:
        """Persist a redacted audit log entry (SPEC §10.2(7))."""
        repo = LightboxRepository(self.db)
        return repo.add_audit_log(
            surface=surface,
            tool_name=tool_name,
            hook_decision=hook_decision,
            args_redacted=args_redacted,
            duration_ms=duration_ms,
            summary=summary,
        )

    def get_llm_traces(
        self,
        *,
        session_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return lightweight `llm_calls` trace ledger and monthly spend summary (PLAN T13)."""
        repo = LightboxRepository(self.db)
        cfg = load_config(self.app_paths)
        calls = repo.list_llm_calls(session_id=session_id, limit=limit)
        now = datetime.now(tz=UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        monthly_spent = repo.sum_llm_cost(since_ts=month_start)
        session_spent = repo.sum_llm_cost(session_id=session_id) if session_id else None

        total_in = sum(c.in_tokens for c in calls)
        total_out = sum(c.out_tokens for c in calls)
        total_cached = sum(c.cached_tokens for c in calls)
        total_cost = sum(c.cost_usd for c in calls)
        cache_hits = sum(1 for c in calls if c.cache_hit)

        return {
            "session_id": session_id,
            "monthly_spent_usd": round(monthly_spent, 6),
            "monthly_budget_usd": cfg.budget.monthly_budget_usd,
            "session_spent_usd": round(session_spent, 6) if session_spent is not None else None,
            "session_budget_usd": cfg.budget.session_budget_usd,
            "model_profile": cfg.model_profile,
            "total_calls": len(calls),
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / len(calls), 4) if calls else 0.0,
            "total_in_tokens": total_in,
            "total_out_tokens": total_out,
            "total_cached_tokens": total_cached,
            "total_cost_usd": round(total_cost, 6),
            "items": [
                {
                    "id": c.id,
                    "role": c.role,
                    "provider": c.provider,
                    "model": c.model,
                    "in_tokens": c.in_tokens,
                    "out_tokens": c.out_tokens,
                    "cached_tokens": c.cached_tokens,
                    "cost_usd": round(c.cost_usd, 6),
                    "latency_ms": c.latency_ms,
                    "cache_hit": c.cache_hit,
                    "status": c.status,
                    "error_class": c.error_class,
                    "connection_fingerprint": c.connection_fingerprint,
                    "session_id": c.session_id,
                    "created_at": c.created_at,
                }
                for c in calls
            ],
        }


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _dump_config_toml(cfg: AppConfig) -> str:
    lines = [
        "# Lightbox general configuration (never store plaintext API keys here)",
        f"language = {_toml_str(cfg.language)}",
        f"sensitivity = {_toml_str(cfg.sensitivity)}",
        f"model_profile = {_toml_str(cfg.model_profile)}",
        f"auto_supervisor = {'true' if cfg.auto_supervisor else 'false'}",
        "",
        "[budget]",
        f"session_budget_usd = {cfg.budget.session_budget_usd:.2f}",
        f"monthly_budget_usd = {cfg.budget.monthly_budget_usd:.2f}",
        "",
        "[quarantine]",
        f"strategy = {_toml_str(cfg.quarantine.strategy)}",
    ]
    if cfg.quarantine.custom_dir is not None:
        lines.append(f"custom_dir = {_toml_str(cfg.quarantine.custom_dir)}")
    lines.extend(
        [
            "",
            "[cache]",
            f"preview_max_gb = {cfg.cache.preview_max_gb:.2f}",
        ]
    )
    if cfg.ai is not None:
        ai = cfg.ai
        lines.extend(
            [
                "",
                "[ai]",
                f"provider = {_toml_str(ai.provider)}",
                f"agent_model = {_toml_str(ai.agent_model)}",
                f"triage_model = {_toml_str(ai.triage_model)}",
            ]
        )
        if ai.base_url is not None:
            lines.append(f"base_url = {_toml_str(ai.base_url)}")
        lines.append(f"reviewer = {_toml_str(ai.reviewer)}")
        lines.append(f"is_local_free = {'true' if ai.is_local_free else 'false'}")
        if ai.input_cost_per_mtok is not None:
            lines.append(f"input_cost_per_mtok = {ai.input_cost_per_mtok}")
        if ai.output_cost_per_mtok is not None:
            lines.append(f"output_cost_per_mtok = {ai.output_cost_per_mtok}")
        if ai.triage_input_cost_per_mtok is not None:
            lines.append(f"triage_input_cost_per_mtok = {ai.triage_input_cost_per_mtok}")
        if ai.triage_output_cost_per_mtok is not None:
            lines.append(f"triage_output_cost_per_mtok = {ai.triage_output_cost_per_mtok}")
        lines.append(f"max_session_calls = {ai.max_session_calls}")
        lines.append(f"max_triage_photos_unpriced = {ai.max_triage_photos_unpriced}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DoctorCheckItem",
    "DoctorReport",
    "InstallAppResult",
    "SystemService",
    "_dump_config_toml",
]
