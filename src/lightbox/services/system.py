"""SystemService: environment health checks for `lightbox doctor` (SPEC §2.4, §12)."""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightbox.config import AppConfig, load_config
from lightbox.core import find_exiftool, find_face_model_path
from lightbox.models import (
    find_face_embedder_model_path,
    find_siglip2_text_path,
    find_siglip2_vision_path,
)
from lightbox.paths import AppPaths
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
        """Validate and persist updated settings to `<APP_DATA_DIR>/config.toml` (forbids API keys)."""
        current = load_config(self.app_paths).model_dump()
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key] = {**current[key], **value}
            else:
                current[key] = value
        validated = AppConfig.model_validate(current)
        self.app_paths.ensure_dirs()
        self.app_paths.config_file.write_text(_dump_config_toml(validated), encoding="utf-8")
        return validated

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
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["DoctorCheckItem", "DoctorReport", "SystemService"]
