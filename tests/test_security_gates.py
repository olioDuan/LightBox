"""Unit tests verifying T01 CI security & architecture boundary gates (SPEC §2.2, §10.2(1))."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_NO_DELETE_SCRIPT = REPO_ROOT / "scripts" / "check_no_delete.py"


def test_ast_no_delete_blocks_path_unlink(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_unlink.py"
    bad_file.write_text(
        'from pathlib import Path\n\ndef nuke() -> None:\n    Path("a").unlink()\n',
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(CHECK_NO_DELETE_SCRIPT), str(bad_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert ".unlink()" in proc.stderr


def test_ast_no_delete_blocks_path_rmdir(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_rmdir.py"
    bad_file.write_text(
        'from pathlib import Path\n\ndef nuke_dir() -> None:\n    Path("a").rmdir()\n',
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(CHECK_NO_DELETE_SCRIPT), str(bad_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert ".rmdir()" in proc.stderr


def test_ruff_tid251_blocks_os_replace(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_replace.py"
    bad_file.write_text(
        'import os\n\ndef overwrite() -> None:\n    os.replace("a", "b")\n',
        encoding="utf-8",
    )

    ruff_bin = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
    proc = subprocess.run(
        [
            ruff_bin,
            "check",
            "--no-cache",
            "--config",
            str(REPO_ROOT / "pyproject.toml"),
            str(bad_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "TID251" in proc.stdout
    assert "os.replace" in proc.stdout

    ast_proc = subprocess.run(
        [sys.executable, str(CHECK_NO_DELETE_SCRIPT), str(bad_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ast_proc.returncode == 1
    assert "os.replace()" in ast_proc.stderr


def test_config_io_allows_os_replace_but_still_blocks_delete_primitives() -> None:
    """Verify check_no_delete whitelists os.replace ONLY for config_io.py while blocking unlink/remove."""
    import ast
    import importlib.util

    config_io_file = REPO_ROOT / "src" / "lightbox" / "config_io.py"
    ok_proc = subprocess.run(
        [sys.executable, str(CHECK_NO_DELETE_SCRIPT), str(config_io_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok_proc.returncode == 0, ok_proc.stderr

    spec = importlib.util.spec_from_file_location("check_no_delete_mod", CHECK_NO_DELETE_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    sample = (
        "import os\n"
        "from pathlib import Path\n"
        "def f() -> None:\n"
        "    os.replace('a', 'b')\n"
        "    Path('a').unlink()\n"
        "    os.unlink('a')\n"
        "    os.remove('a')\n"
    )
    visitor = mod._NoDeleteVisitor(
        config_io_file,
        allow_cache_unlink=False,
        allow_safe_fs_rename=False,
        allow_config_io_replace=True,
    )
    visitor.visit(ast.parse(sample, filename=str(config_io_file)))
    msgs = [v.message for v in visitor.violations]
    assert len(msgs) == 3
    assert any(".unlink()" in m for m in msgs)
    assert any("os.unlink()" in m for m in msgs)
    assert any("os.remove()" in m for m in msgs)
    assert not any("os.replace()" in m for m in msgs)


def test_ast_no_delete_blocks_path_rename(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_rename.py"
    bad_file.write_text(
        'from pathlib import Path\n\ndef overwrite() -> None:\n    Path("a").rename("b")\n',
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(CHECK_NO_DELETE_SCRIPT), str(bad_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert ".rename()" in proc.stderr


def test_ruff_tid251_and_ast_block_os_rename(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_os_rename.py"
    bad_file.write_text(
        'import os\n\ndef overwrite() -> None:\n    os.rename("a", "b")\n',
        encoding="utf-8",
    )

    ruff_bin = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
    ruff_proc = subprocess.run(
        [
            ruff_bin,
            "check",
            "--no-cache",
            "--config",
            str(REPO_ROOT / "pyproject.toml"),
            str(bad_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ruff_proc.returncode != 0
    assert "TID251" in ruff_proc.stdout
    assert "os.rename" in ruff_proc.stdout

    ast_proc = subprocess.run(
        [sys.executable, str(CHECK_NO_DELETE_SCRIPT), str(bad_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ast_proc.returncode == 1
    assert "os.rename()" in ast_proc.stderr


def _setup_isolated_pkg(tmp_path: Path) -> Path:
    shutil.copy2(REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    pkg_root = tmp_path / "src" / "lightbox"
    for sub in (
        "core",
        "jobs",
        "fs",
        "store",
        "models",
        "agent",
        "services",
        "tools",
        "mcp",
        "api",
        "cli",
        "i18n",
    ):
        (pkg_root / sub).mkdir(parents=True, exist_ok=True)
        (pkg_root / sub / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    return pkg_root


def _run_lint_imports(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    lint_imports_bin = shutil.which("lint-imports") or str(
        Path(sys.executable).parent / "lint-imports"
    )
    env = {"PYTHONPATH": str(tmp_path / "src"), "PATH": str(Path(sys.executable).parent)}
    return subprocess.run(
        [lint_imports_bin, "--config", str(tmp_path / "pyproject.toml")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_linter_blocks_core_importing_services(tmp_path: Path) -> None:
    """Verify import-linter contract rejects `lightbox.core` importing `lightbox.services`."""
    pkg_root = _setup_isolated_pkg(tmp_path)
    (pkg_root / "core" / "violation.py").write_text(
        "import lightbox.services\n",
        encoding="utf-8",
    )

    proc = _run_lint_imports(tmp_path)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "BROKEN" in combined
    assert "lightbox.core.violation -> lightbox.services" in combined


def test_import_linter_allows_cli_to_services_to_core_but_blocks_direct(tmp_path: Path) -> None:
    """Verify `cli -> services -> core` is allowed while direct `cli -> core` is blocked."""
    pkg_root = _setup_isolated_pkg(tmp_path)
    (pkg_root / "services" / "scan.py").write_text("import lightbox.core\n", encoding="utf-8")
    (pkg_root / "cli" / "cmd.py").write_text("import lightbox.services.scan\n", encoding="utf-8")

    ok_proc = _run_lint_imports(tmp_path)
    assert ok_proc.returncode == 0, ok_proc.stdout + ok_proc.stderr

    # Now add a direct import `cli -> core`, which must fail
    (pkg_root / "cli" / "bad_cmd.py").write_text("import lightbox.core\n", encoding="utf-8")
    bad_proc = _run_lint_imports(tmp_path)
    assert bad_proc.returncode != 0
    assert "lightbox.cli.bad_cmd -> lightbox.core" in (bad_proc.stdout + bad_proc.stderr)
