"""Typer CLI entrypoint for Lightbox (SPEC §2.4, §10.2(3), PLAN T06)."""

from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
import threading
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
import uvicorn

from lightbox import __version__
from lightbox.api import (
    LOCAL_HOST,
    clear_server_lock,
    create_app,
    find_free_local_port,
    read_server_lock,
    write_server_lock,
)
from lightbox.mcp import DEFAULT_MCP_HTTP_PORT, MCPTransport, run_mcp_server
from lightbox.paths import get_app_paths
from lightbox.services import (
    ApprovalCodeError,
    LightboxServices,
    PlanError,
    build_demo_dataset,
    run_e1_evaluation,
    run_personal_eval,
)

app = typer.Typer(
    name="lightbox",
    help="Lightbox — local-first photo culling and intelligent selection system.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"lightbox {__version__}")
        raise typer.Exit()


@app.callback()
def _root_options(
    version: Annotated[
        bool,
        typer.Option("--version", "-v", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    _ = version


@app.command()
def scan(
    folder: Annotated[Path, typer.Argument(help="Photo directory to scan.")],
    sensitivity: Annotated[
        Literal["conservative", "standard"] | None,
        typer.Option("--sensitivity", help="Detection sensitivity preset."),
    ] = None,
    embed: Annotated[
        bool,
        typer.Option("--embed", help="Enable L2 SigLIP 2 embedding (activates in T11)."),
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option("--workers", min=1, help="Worker process count for CPU analysis."),
    ] = None,
    draft_obvious: Annotated[
        bool,
        typer.Option("--draft-obvious", help="Also draft a one-click obvious-rejects quarantine plan."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON output."),
    ] = False,
) -> None:
    """Run zero-LLM deterministic scan on a photo folder with resumable caching."""
    with LightboxServices.open() as svcs:
        res = svcs.scan.scan_folder(folder, sensitivity=sensitivity, max_workers=workers)
        draft_payload = None
        if draft_obvious:
            draft = svcs.plan.draft_obvious_rejects(res.session_id)
            draft_payload = {
                "plan_id": draft.plan.id if draft.plan else None,
                "plan_hash": draft.plan.plan_hash if draft.plan else None,
                "selected_count": draft.selected_count,
                "tech_reject_count": draft.tech_reject_count,
                "burst_non_pick_count": draft.burst_non_pick_count,
                "manual_reject_count": draft.manual_reject_count,
                "high_ratio_warning": draft.high_ratio_warning,
                "quarantine_dir": draft.plan.quarantine_dir if draft.plan else None,
            }

    payload = {
        "session_id": res.session_id,
        "job_id": res.job_id,
        "root_path": str(res.root_path),
        "sensitivity": res.sensitivity,
        "status": res.status,
        "total_photos": res.total_photos,
        "counts_by_bucket": res.counts_by_bucket,
        "burst_count": res.burst_count,
        "cache_hits": res.cache_hits,
        "skipped_retryable": res.skipped_retryable,
        "elapsed_s": res.elapsed_s,
        "embed_requested": embed,
        "obvious_rejects_plan": draft_payload,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    typer.echo(f"Session : {res.session_id} ({res.root_path})")
    typer.echo(f"Job     : {res.job_id} [{res.status}] in {res.elapsed_s:.2f}s")
    typer.echo(
        f"Photos  : {res.total_photos} total | "
        f"red={res.counts_by_bucket.get('red', 0)} "
        f"yellow={res.counts_by_bucket.get('yellow', 0)} "
        f"normal={res.counts_by_bucket.get('normal', 0)} "
        f"exempt={res.counts_by_bucket.get('exempt', 0)}"
    )
    typer.echo(f"Bursts  : {res.burst_count} | Cache hits: {res.cache_hits}")
    if embed:
        typer.echo("Note: --embed (SigLIP 2 semantic indexing) will be enabled in T11.")
    if draft_payload is not None:
        if draft_payload["plan_id"]:
            typer.echo(
                f"Drafted obvious-rejects plan: {draft_payload['plan_id']} "
                f"({draft_payload['selected_count']} photos -> {draft_payload['quarantine_dir']})"
            )
            typer.echo(f"Approve via terminal: lightbox approve {draft_payload['plan_id']}")
        else:
            typer.echo("No obvious rejects eligible for quarantine plan.")


@app.command()
def demo(
    target_dir: Annotated[
        Path | None,
        typer.Option("--dir", help="Optional directory to materialize demo photos."),
    ] = None,
    sensitivity: Annotated[
        Literal["conservative", "standard"],
        typer.Option("--sensitivity", help="Detection sensitivity preset."),
    ] = "conservative",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON output."),
    ] = False,
) -> None:
    """Run an offline, zero-Key demo scan on lightweight synthetic CC0 photos."""
    paths = get_app_paths()
    paths.ensure_dirs()
    demo_dir = (target_dir or (paths.root / "demo_shoot")).expanduser().resolve()
    specs = build_demo_dataset(demo_dir)

    with LightboxServices(app_paths=paths) as svcs:
        res = svcs.scan.scan_folder(demo_dir, sensitivity=sensitivity, max_workers=1)
        summary = svcs.review.get_session_summary(res.session_id)
        draft = svcs.plan.draft_obvious_rejects(res.session_id, strategy="sibling")

    payload = {
        "demo_dir": str(demo_dir),
        "generated_photos": len(specs),
        "session_id": res.session_id,
        "job_id": res.job_id,
        "elapsed_s": res.elapsed_s,
        "counts_by_bucket": summary.counts_by_bucket,
        "burst_count": summary.burst_count,
        "obvious_reject_plan_id": draft.plan.id if draft.plan else None,
        "obvious_reject_count": draft.selected_count,
        "tech_reject_count": draft.tech_reject_count,
        "burst_non_pick_count": draft.burst_non_pick_count,
        "top_reasons": list(summary.top_reasons),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    typer.echo("=== Lightbox Zero-LLM Demo ===")
    typer.echo(f"Demo Folder     : {demo_dir}")
    typer.echo(f"Session ID      : {res.session_id} (scanned in {res.elapsed_s:.2f}s)")
    typer.echo(
        f"Bucket Summary  : red={summary.counts_by_bucket.get('red', 0)} | "
        f"yellow={summary.counts_by_bucket.get('yellow', 0)} | "
        f"normal={summary.counts_by_bucket.get('normal', 0)} | "
        f"exempt={summary.counts_by_bucket.get('exempt', 0)}"
    )
    typer.echo(f"Burst Groups    : {summary.burst_count} (unconfirmed non-picks kept safe)")
    if draft.plan is not None:
        typer.echo(
            f"Obvious Rejects : {draft.selected_count} photos ready in draft plan {draft.plan.id}"
        )
        typer.echo(f"Quarantine Dir  : {draft.plan.quarantine_dir}")
        typer.echo(f"Next Step       : Run `lightbox approve {draft.plan.id}` to approve.")


def _is_interactive_tty() -> bool:
    """Return True when stdin is attached to an interactive terminal."""
    return sys.stdin.isatty()


@app.command()
def approve(
    plan_id: Annotated[str, typer.Argument(help="Execution plan ID to approve interactively.")],
    apply_now: Annotated[
        bool,
        typer.Option("--apply", help="Immediately apply the quarantine plan after code verification."),
    ] = False,
) -> None:
    """Interactive TTY fallback approval: verify photo count and issue/confirm a 6-digit SHA-256 code."""
    if not _is_interactive_tty():
        typer.echo(
            "Error: `lightbox approve` requires an interactive terminal (TTY) to prevent "
            "unattended or prompt-injected execution (SPEC §10.2(3)).",
            err=True,
        )
        raise typer.Exit(code=1)

    with LightboxServices.open() as svcs:
        try:
            plan, items = svcs.plan.get_plan(plan_id)
        except PlanError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from None

        photo_count = len(items)
        typer.echo(f"Plan ID        : {plan.id}")
        typer.echo(f"Plan Hash      : {plan.plan_hash}")
        typer.echo(f"Status         : {plan.status}")
        typer.echo(f"Photo Groups   : {photo_count}")
        typer.echo(f"Quarantine Dir : {plan.quarantine_dir}")

        if plan.verifier_report_json:
            report = json.loads(plan.verifier_report_json)
            if report.get("high_ratio_warning"):
                typer.echo(
                    f"WARNING: High reject ratio ({report.get('reject_ratio', 0.0):.1%}) exceeds 40%!",
                    err=True,
                )

        entered_count = typer.prompt(
            f"Type the exact number of photo groups ({photo_count}) to confirm"
        ).strip()
        if entered_count != str(photo_count):
            typer.echo("Confirmation count did not match. Aborted without approving.", err=True)
            raise typer.Exit(code=1)

        try:
            issued = svcs.plan.issue_cli_approval_code(plan_id)
        except PlanError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from None

        exp_str = datetime.fromtimestamp(issued.expires_at, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        typer.echo(f"One-time 6-digit approval code: {issued.code} (valid until {exp_str})")

        entered_code = typer.prompt("Enter the 6-digit approval code to approve this plan").strip()
        try:
            approved = svcs.plan.verify_and_approve_with_code(plan_id, entered_code)
        except (ApprovalCodeError, PlanError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from None

        typer.echo(f"Approved plan {approved.id} (status={approved.status}).")

        if apply_now:
            try:
                result = svcs.plan.apply_plan(plan_id)
            except PlanError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(code=1) from None
            typer.echo(
                f"Applied plan {result.plan_id}: moved {result.moved_photo_count} photo groups "
                f"({result.moved_file_count} files) -> {result.quarantine_dir}"
            )


@app.command()
def ui(
    port: Annotated[
        int,
        typer.Option("--port", min=0, max=65535, help="Local TCP port on 127.0.0.1 (0 = auto)."),
    ] = 0,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Do not automatically open the browser."),
    ] = False,
) -> None:
    """Start the local-only (127.0.0.1) review web server and open browser with one-time launch auth."""
    paths = get_app_paths()
    paths.ensure_dirs()

    existing = read_server_lock(app_paths=paths, check_alive=True)
    if existing is not None:
        typer.echo(f"Lightbox UI is already running at {existing.base_url} (pid={existing.pid})")
        if not no_browser:
            webbrowser.open(existing.base_url)
        return

    chosen_port = port if port > 0 else find_free_local_port(LOCAL_HOST)
    launch_token = secrets.token_urlsafe(32)
    instance_id = secrets.token_hex(8)
    fastapi_app = create_app(
        app_paths=paths,
        port=chosen_port,
        bootstrap_token=launch_token,
        instance_id=instance_id,
    )
    launch_url = f"http://{LOCAL_HOST}:{chosen_port}/auth/launch?token={launch_token}"

    write_server_lock(
        chosen_port,
        app_paths=paths,
        host=LOCAL_HOST,
        pid=os.getpid(),
        instance_id=instance_id,
    )
    typer.echo(f"Lightbox UI serving on http://{LOCAL_HOST}:{chosen_port}")

    if not no_browser:
        threading.Timer(0.6, webbrowser.open, args=(launch_url,)).start()

    class _LightboxServer(uvicorn.Server):
        def handle_exit(self, sig: int, frame: Any) -> None:
            # Immediately terminate open SSE (`/api/events`) streams so a single Ctrl+C
            # never blocks waiting on browser EventSource connections to close.
            fastapi_app.state.event_bus.close()
            super().handle_exit(sig, frame)

    config = uvicorn.Config(
        fastapi_app,
        host=LOCAL_HOST,
        port=chosen_port,
        log_level="warning",
        timeout_graceful_shutdown=1,
    )
    server = _LightboxServer(config)
    try:
        server.run()
    finally:
        fastapi_app.state.event_bus.close()
        clear_server_lock(app_paths=paths, only_if_pid=os.getpid())


@app.command()
def mcp(
    transport: Annotated[
        MCPTransport,
        typer.Option("--transport", help="MCP transport mode: 'stdio' (default) or local 'http' (Streamable HTTP)."),
    ] = "stdio",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535, help="Local 127.0.0.1 port when using --transport http."),
    ] = DEFAULT_MCP_HTTP_PORT,
    no_auto_ui: Annotated[
        bool,
        typer.Option("--no-auto-ui", help="Disable auto-starting the embedded local Web UI when server.lock is absent."),
    ] = False,
) -> None:
    """Start the Lightbox MCP Server (14 tools, 4 resources, 3 prompts) on stdio or 127.0.0.1 HTTP."""
    run_mcp_server(
        transport=transport,
        port=port,
        app_paths=get_app_paths(),
        auto_start_ui=not no_auto_ui,
    )


@app.command()
def doctor(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON output."),
    ] = False,
) -> None:
    """Check Python/sqlite-vec, ExifTool, model caches, storage writability, and API key status."""
    with LightboxServices.open() as svcs:
        report = svcs.system.run_doctor()

    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        typer.echo("=== Lightbox Doctor ===")
        for item in report.checks:
            badge = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}.get(item.status, "[INFO]")
            typer.echo(f"{badge} {item.name:18s}: {item.detail}")
        typer.echo(f"Overall: {'HEALTHY' if report.overall_ok else 'ISSUES DETECTED'}")

    if not report.overall_ok:
        raise typer.Exit(code=1)


@app.command(name="eval")
def eval_cmd(
    personal: Annotated[
        bool,
        typer.Option("--personal", help="Run offline regression summary on local overrides table."),
    ] = False,
    sensitivity: Annotated[
        Literal["conservative", "standard"],
        typer.Option("--sensitivity", help="Detection sensitivity preset for E1."),
    ] = "conservative",
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON report output file path."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON output."),
    ] = False,
) -> None:
    """Run deterministic E1 synthetic CI gate or offline personal override summary."""
    if personal:
        with LightboxServices.open() as svcs:
            p_report = run_personal_eval(svcs)
        text = json.dumps(p_report.to_dict(), indent=2, ensure_ascii=False)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
        if as_json:
            typer.echo(text)
        else:
            typer.echo(f"Personal overrides: {p_report.total_overrides} total")
            typer.echo(
                f"Rescued from red: {p_report.rescued_from_red} | "
                f"Rejected from normal: {p_report.rejected_from_normal}"
            )
        return

    with tempfile.TemporaryDirectory(prefix="lightbox_eval_e1_") as tmp:
        report = run_e1_evaluation(Path(tmp), sensitivity=sensitivity, max_workers=1)

    text = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")

    if as_json:
        typer.echo(text)
    else:
        typer.echo("=== Lightbox E1 Engine Gate ===")
        typer.echo(f"Samples        : {report.total_samples} ({report.sensitivity}) in {report.elapsed_s:.2f}s")
        typer.echo(
            f"Red FPR        : {report.red_fpr:.2%} ({report.false_red_count}/{report.keep_and_protected_count}) [gate < 1.00%]"
        )
        typer.echo(
            f"Reject Recall  : {report.obvious_reject_recall:.2%} ({report.true_red_count}/{report.obvious_reject_count}) [gate >= 75.00%]"
        )
        typer.echo(
            f"Exempt Mislabel: {report.exempt_mislabel_rate:.2%} ({report.exempt_mislabel_count}/{report.exempt_count}) [gate == 0.00%]"
        )
        typer.echo(f"Gate Result    : {'PASS' if report.passed else 'FAIL'}")

    if not report.passed:
        raise typer.Exit(code=1)


def main() -> None:
    app()


__all__ = ["app", "main"]
