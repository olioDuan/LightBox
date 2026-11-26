"""Contract & ground-truth culling tests for the dual-mode `lightbox-cull` Skill, AGENTS.md, and CLI fallback (PLAN T09)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from typer.testing import CliRunner

from lightbox.cli import app as cli_app
from lightbox.i18n import TECH_REJECT_REASON_CODES
from lightbox.mcp import LightboxMCPBundle, create_mcp_server
from lightbox.paths import get_app_paths
from lightbox.services import (
    LightboxServices,
    build_demo_dataset,
    restore_dataset_write_bits,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_DIR = REPO_ROOT / "skills" / "lightbox-cull"
SKILL_MD = SKILL_DIR / "SKILL.md"
RUBRIC_MD = SKILL_DIR / "references" / "culling_rubric.md"
FALLBACK_SH = SKILL_DIR / "scripts" / "cli_fallback.sh"
AGENTS_MD = REPO_ROOT / "AGENTS.md"


@asynccontextmanager
async def _open_mcp_client(bundle: LightboxMCPBundle) -> AsyncIterator[ClientSession]:
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        init_opts = bundle.server._lowlevel_server.create_initialization_options()

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                bundle.server._lowlevel_server.run,
                server_read,
                server_write,
                init_opts,
                True,
            )
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                try:
                    yield session
                finally:
                    tg.cancel_scope.cancel()


def _parse_json(res: types.CallToolResult) -> dict[str, Any]:
    assert not res.is_error, f"Unexpected tool error: {res.content}"
    first = res.content[0]
    assert isinstance(first, types.TextContent)
    parsed = json.loads(first.text)
    assert isinstance(parsed, dict)
    return parsed


# ---------------------------------------------------------------------------
# 1. Skill trigger breadth, intent-first structure & rejects-first blur layout
# ---------------------------------------------------------------------------


def test_skill_frontmatter_triggers_broad_photo_culling_and_blur_intents() -> None:
    """Description must trigger on blur detection, reject cleanup, picking, bursts, and local folders."""
    text = SKILL_MD.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    assert m is not None, "SKILL.md must begin with YAML frontmatter"
    frontmatter = m.group(1)

    assert "name: lightbox-cull" in frontmatter
    # Covers finding blurry/bad photos (not just picking N best)
    assert "帮我把糊的照片挑出来" in frontmatter
    assert "帮我去掉废片" in frontmatter
    assert "blurry" in frontmatter.lower()
    # Covers picking best photos & comparing bursts
    assert "帮我挑 3 张" in frontmatter
    assert "连拍" in frontmatter
    # Covers zero-install chat mode and local folder MCP mode
    assert "zero installation" in frontmatter.lower()
    assert "mcp" in frontmatter.lower()


def test_skill_and_rubric_put_obvious_rejects_first_and_required_for_blur_tasks() -> None:
    """When user asks to find blurry/bad photos, obvious rejects section must come first and be required."""
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    rubric_text = RUBRIC_MD.read_text(encoding="utf-8")

    # Zero-install chat mode prerequisites explicitly absent & not blocked by lightbox_scan_folder
    assert "no LightBox installed" in skill_text
    assert "no MCP configured" in skill_text
    assert "no LightBox models downloaded" in skill_text
    assert "no extra API key" in skill_text
    assert "never blocks Mode A" in skill_text

    # In rubric Template 1 (finding blurry/bad photos), obvious rejects MUST appear BEFORE uncertain photos
    t1_start = rubric_text.index("### Template 1: Finding Blurry / Bad Photos")
    t2_start = rubric_text.index("### Template 2: Picking Standout Photos")
    template1 = rubric_text[t1_start:t2_start]

    rejects_pos = template1.index("### 明显废片 / 模糊照片")
    uncertain_pos = template1.index("### 拿不准的照片")
    keepers_pos = template1.index("### 清晰可用照片")
    assert rejects_pos < uncertain_pos < keepers_pos
    assert "MUST come **FIRST** and is **REQUIRED**" in template1


def test_skill_and_agents_md_enforce_local_folder_mcp_safety_rules() -> None:
    """Mode B retains all 4 updated hard rules scoped to local folders."""
    for path in (SKILL_MD, AGENTS_MD):
        text = path.read_text(encoding="utf-8")
        assert "lightbox_scan_folder" in text
        assert "lightbox://memory/preferences" in text
        assert "review_url" in text
        assert "lightbox_create_plan" in text
        assert "lightbox_apply_plan" in text
        assert "lightbox_restore_plan" in text
        assert "Mode A" in text
        assert "Mode B" in text


# ---------------------------------------------------------------------------
# 2. Ground-truth culling correctness tests (blur/reject detection + best picks + bursts)
# ---------------------------------------------------------------------------


def test_ground_truth_blur_detection_and_picking_accuracy_via_mcp(tmp_path: Path) -> None:
    """Verify real image culling accuracy on labeled ground-truth photos (blur, keepers, creative guards, bursts)."""
    home = tmp_path / "lightbox_home"
    shoot_dir = tmp_path / "labeled_shoot"
    specs = build_demo_dataset(shoot_dir)

    try:
        label_by_stem = {Path(s.filename).stem: s.label for s in specs}
        keep_stems = {stem for stem, lbl in label_by_stem.items() if lbl == "keep"}
        reject_stems = {stem for stem, lbl in label_by_stem.items() if lbl == "obvious_reject"}
        protected_stems = {stem for stem, lbl in label_by_stem.items() if lbl == "protected_non_red"}
        exempt_stems = {stem for stem, lbl in label_by_stem.items() if lbl == "exempt"}

        assert len(keep_stems) >= 3
        assert len(reject_stems) >= 3
        assert len(protected_stems) >= 1
        assert len(exempt_stems) >= 1

        paths = get_app_paths(home)
        bundle = create_mcp_server(app_paths=paths, auto_start_ui=False)

        async def _verify() -> None:
            async with _open_mcp_client(bundle) as client:
                scan_data = _parse_json(
                    await client.call_tool(
                        "lightbox_scan_folder",
                        {"folder_path": str(shoot_dir), "wait_seconds": 30},
                    )
                )
                session_id = str(scan_data["session_id"])

                # --- Intent 1: "帮我把糊的照片挑出来 / 去掉废片" ---
                red_data = _parse_json(
                    await client.call_tool(
                        "lightbox_list_photos",
                        {"session_id": session_id, "bucket": "red", "limit": 30},
                    )
                )
                red_items = red_data["items"]
                red_stems = {str(it["stem"]) for it in red_items}

                # All sharp keepers & creative-intent protected & exempt shots MUST have 0 false rejects
                assert red_stems.isdisjoint(keep_stems), (
                    f"False rejects on sharp keepers: {red_stems & keep_stems}"
                )
                assert red_stems.isdisjoint(protected_stems), (
                    f"Creative-intent shots falsely marked red: {red_stems & protected_stems}"
                )
                assert red_stems.isdisjoint(exempt_stems), (
                    f"Exempt shots falsely marked red: {red_stems & exempt_stems}"
                )

                # Obvious blurry / mis-shot rejects MUST be caught with >= 75% recall & valid technical reason codes
                caught_rejects = red_stems & reject_stems
                recall = len(caught_rejects) / float(len(reject_stems))
                assert recall >= 0.75, (
                    f"Obvious reject recall too low: {recall:.2%} ({caught_rejects} vs {reject_stems})"
                )
                for it in red_items:
                    codes = set(it["reason_codes"])
                    assert codes & TECH_REJECT_REASON_CODES, (
                        f"Red photo {it['stem']} missing technical reject reason code: {codes}"
                    )

                # Exempt shots MUST remain in `exempt` bucket
                exempt_data = _parse_json(
                    await client.call_tool(
                        "lightbox_list_photos",
                        {"session_id": session_id, "bucket": "exempt", "limit": 30},
                    )
                )
                found_exempt = {str(it["stem"]) for it in exempt_data["items"]}
                assert exempt_stems.issubset(found_exempt)

                # Draft quarantine plan MUST include ONLY caught obvious rejects (never keepers/protected/exempt/unconfirmed bursts)
                plan_data = _parse_json(
                    await client.call_tool(
                        "lightbox_create_plan",
                        {"session_id": session_id, "action": "quarantine_move"},
                    )
                )
                assert plan_data["verifier_report"]["passed"] is True
                assert plan_data["verifier_report"]["exempt_count"] == 0
                assert plan_data["verifier_report"]["unconfirmed_burst_non_pick_count"] == 0
                assert plan_data["count"] == len(red_items)

                # --- Intent 2: "帮我挑 3 张" & "这几张连拍选哪张" ---
                normal_data = _parse_json(
                    await client.call_tool(
                        "lightbox_list_photos",
                        {"session_id": session_id, "bucket": "normal", "sort": "name", "limit": 30},
                    )
                )
                normal_stems = {str(it["stem"]) for it in normal_data["items"]}
                # None of the obvious blurry/extreme-exposure rejects appear in sharp normal candidates
                assert normal_stems.isdisjoint(reject_stems), (
                    f"Obvious rejects leaked into normal candidates: {normal_stems & reject_stems}"
                )
                # At least 3 distinct sharp keepers are available for "帮我挑 3 张"
                assert len(normal_stems & keep_stems) >= 3

                # Burst comparison ranks the sharpest frame as pick #1 and finds burst siblings via find_similar
                with LightboxServices(app_paths=paths, migrate=False) as svcs:
                    bursts = svcs.burst.list_bursts(session_id)
                assert bursts
                cmp_data = _parse_json(
                    await client.call_tool("lightbox_compare_burst", {"group_id": bursts[0].burst.id})
                )
                frames = cmp_data["frames"]
                assert len(frames) >= 2
                pick_frames = [f for f in frames if f["is_pick"]]
                assert len(pick_frames) == 1
                pick_frame = pick_frames[0]
                assert pick_frame["rank"] == 1
                max_sharp = max(float(f["sharp_p95"] or 0.0) for f in frames)
                assert float(pick_frame["sharp_p95"] or 0.0) == max_sharp

                sim_data = _parse_json(
                    await client.call_tool("lightbox_find_similar", {"photo_id": pick_frame["photo_id"]})
                )
                sibling_ids = {m["photo_id"] for m in sim_data["similar"] if m["relation"] == "burst_sibling"}
                expected_siblings = {f["photo_id"] for f in frames if f["photo_id"] != pick_frame["photo_id"]}
                assert expected_siblings.issubset(sibling_ids)

        asyncio.run(_verify())
    finally:
        restore_dataset_write_bits(shoot_dir)


# ---------------------------------------------------------------------------
# 3. cli_fallback.sh safety & unrelated-pyproject.toml bug regression tests
# ---------------------------------------------------------------------------


def _write_stub_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_cli_fallback_script_syntax_and_no_delete_primitives() -> None:
    assert os.access(FALLBACK_SH, os.X_OK), "cli_fallback.sh must be executable"
    content = FALLBACK_SH.read_text(encoding="utf-8")
    code_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in code_lines:
        assert not re.search(r"(^|[;&|]\s*)(rm|mv)\s+", line), f"Forbidden rm/mv in fallback: {line}"

    proc = subprocess.run(["bash", "-n", str(FALLBACK_SH)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_cli_fallback_does_not_misuse_uv_run_in_unrelated_python_project(tmp_path: Path) -> None:
    """Regression: an unrelated pyproject.toml in CWD must NOT trigger bare `uv run lightbox`."""
    other_project = tmp_path / "other_python_project"
    other_project.mkdir()
    (other_project / "pyproject.toml").write_text(
        '[project]\nname = "some-other-app"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    _write_stub_executable(fake_bin / "uv", '#!/usr/bin/env bash\necho "UV:$*"\n')
    _write_stub_executable(fake_bin / "uvx", '#!/usr/bin/env bash\necho "UVX:$*"\n')
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin"}

    # Case A: Invoking repo's cli_fallback.sh from an unrelated Python project resolves to repo root via --directory
    proc_repo = subprocess.run(
        [str(FALLBACK_SH), "doctor", "--json"],
        cwd=other_project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_repo.returncode == 0, proc_repo.stderr
    assert f"UV:--directory {REPO_ROOT} run lightbox doctor --json" in proc_repo.stdout

    # Case B: Invoking a standalone installed skill copy from an unrelated Python project skips uv run -> uses uvx
    standalone_script = tmp_path / "standalone_skill" / "scripts" / "cli_fallback.sh"
    standalone_script.parent.mkdir(parents=True)
    shutil.copy2(FALLBACK_SH, standalone_script)
    standalone_script.chmod(standalone_script.stat().st_mode | stat.S_IXUSR)

    proc_standalone = subprocess.run(
        [str(standalone_script), "doctor", "--json"],
        cwd=other_project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_standalone.returncode == 0, proc_standalone.stderr
    assert "UVX:--from lightbox-cull lightbox doctor --json" in proc_standalone.stdout


def test_cli_mcp_command_help_is_registered() -> None:
    runner = CliRunner()
    res = runner.invoke(cli_app, ["mcp", "--help"])
    assert res.exit_code == 0
    plain_output = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", res.output)
    assert "--transport" in plain_output
    assert "--port" in plain_output
