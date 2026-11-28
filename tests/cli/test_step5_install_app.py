"""Step 5 unit tests: cross-platform `lightbox install-app` (macOS .app / Windows .lnk / Linux .desktop)."""

from __future__ import annotations

import json
import os
import plistlib
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lightbox.cli import app as cli_app
from lightbox.paths import get_app_paths
from lightbox.services import LightboxServices
from lightbox.services.app_installer import (
    install_desktop_app,
    resolve_desktop_launcher,
)


def _make_fake_venv(base_dir: Path, *, files: tuple[str, ...]) -> Path:
    bin_dir = base_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        p = bin_dir / name
        p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        os.chmod(p, 0o755)
    return bin_dir


def test_resolve_desktop_launcher_preserves_venv_symlink_dir(tmp_path: Path) -> None:
    # Real base Python lives elsewhere (simulating uv managed Python); venv/bin/python symlinks to it
    base_py_dir = tmp_path / "base_python" / "bin"
    base_py_dir.mkdir(parents=True)
    base_py = base_py_dir / "python3.12"
    base_py.write_text("#!/bin/sh\n", encoding="utf-8")

    venv_bin = tmp_path / "my venv" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_py = venv_bin / "python"
    os.symlink(base_py, venv_py)
    lightbox_app = venv_bin / "lightbox-app"
    lightbox_app.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(lightbox_app, 0o755)

    resolved = resolve_desktop_launcher(python_executable=venv_py, platform_name="darwin")
    assert resolved.kind == "gui_script"
    assert resolved.target == lightbox_app.resolve()
    assert resolved.args == ()


def test_resolve_desktop_launcher_windows_priority_order(tmp_path: Path) -> None:
    # 1. lightbox-app.exe preferred (console-less gui-script)
    win_bin_1 = _make_fake_venv(
        tmp_path / "win1",
        files=("python.exe", "pythonw.exe", "lightbox-app.exe"),
    )
    r1 = resolve_desktop_launcher(python_executable=win_bin_1 / "python.exe", platform_name="win32")
    assert r1.kind == "gui_script"
    assert r1.target.name == "lightbox-app.exe"
    assert r1.args == ()

    # 2. pythonw.exe -m lightbox.cli app when lightbox-app.exe absent
    win_bin_2 = _make_fake_venv(
        tmp_path / "win2",
        files=("python.exe", "pythonw.exe"),
    )
    r2 = resolve_desktop_launcher(python_executable=win_bin_2 / "python.exe", platform_name="win32")
    assert r2.kind == "pythonw_module"
    assert r2.target.name == "pythonw.exe"
    assert r2.args == ("-m", "lightbox.cli", "app")

    # 3. python.exe fallback
    win_bin_3 = _make_fake_venv(tmp_path / "win3", files=("python.exe",))
    r3 = resolve_desktop_launcher(python_executable=win_bin_3 / "python.exe", platform_name="win32")
    assert r3.kind == "python_module"
    assert r3.target.name == "python.exe"
    assert r3.args == ("-m", "lightbox.cli", "app")


def test_install_macos_app_bundle_and_desktop_alias(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    # Include a space in the venv path to verify shell quoting
    venv_bin = _make_fake_venv(tmp_path / "Space Env", files=("python", "lightbox-app"))
    app_paths = get_app_paths(tmp_path / "app_data_mac")

    res = install_desktop_app(
        app_paths,
        include_desktop=True,
        home_dir=home,
        platform_name="darwin",
        python_executable=venv_bin / "python",
    )

    assert res.platform == "darwin"
    assert res.primary_path == home / "Applications" / "LightBox.app"
    assert res.desktop_path == home / "Desktop" / "LightBox.app"
    assert res.launcher_kind == "gui_script"
    assert res.icon_path is not None and res.icon_path.is_file()
    assert res.icon_path.name == "LightBox.icns"

    # Info.plist is valid XML plist with expected bundle metadata & macOS TCC descriptions
    plist_path = res.primary_path / "Contents" / "Info.plist"
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["CFBundleExecutable"] == "LightBox"
    assert plist["CFBundleIdentifier"] == "io.lightbox.desktop"
    assert plist["CFBundleDisplayName"] == "LightBox"
    assert plist["CFBundleIconFile"] == "LightBox"
    assert plist["NSHighResolutionCapable"] is True
    assert "NSDesktopFolderUsageDescription" in plist
    assert "NSPicturesFolderUsageDescription" in plist
    assert "NSRemovableVolumesUsageDescription" in plist

    # Executable wrapper script is chmod +x, execs quoted lightbox-app, and has osascript error alert
    exec_script = res.primary_path / "Contents" / "MacOS" / "LightBox"
    assert exec_script.is_file()
    mode = stat.S_IMODE(exec_script.stat().st_mode)
    assert mode & stat.S_IXUSR
    script_text = exec_script.read_text(encoding="utf-8")
    assert script_text.startswith("#!/bin/sh\n")
    assert f"'{(venv_bin / 'lightbox-app').resolve()}'" in script_text
    assert 'exec "$TARGET" "$@"' in script_text
    assert "osascript" in script_text

    # Desktop symlink points to ~/Applications/LightBox.app
    assert res.desktop_path.is_symlink()
    assert Path(os.readlink(res.desktop_path)) == res.primary_path

    # Re-running with an existing unrelated file on Desktop never deletes/overwrites it
    home2 = tmp_path / "home2"
    (home2 / "Desktop").mkdir(parents=True)
    existing_desktop = home2 / "Desktop" / "LightBox.app"
    existing_desktop.write_text("user-kept", encoding="utf-8")
    res2 = install_desktop_app(
        app_paths,
        include_desktop=True,
        home_dir=home2,
        platform_name="darwin",
        python_executable=venv_bin / "python",
    )
    assert res2.desktop_path is None
    assert existing_desktop.read_text(encoding="utf-8") == "user-kept"
    assert any("kept untouched" in n for n in res2.notes)


def test_install_windows_lnk_direct_exe_no_vbs(tmp_path: Path) -> None:
    home = tmp_path / "O'Connor Home"
    home.mkdir()
    venv_bin = _make_fake_venv(
        tmp_path / "win_venv",
        files=("python.exe", "pythonw.exe", "lightbox-app.exe"),
    )
    app_paths = get_app_paths(tmp_path / "app_data_win")

    recorded_cmds: list[list[str]] = []

    def _fake_runner(cmd: Any) -> subprocess.CompletedProcess[str]:
        cmd_list = [str(c) for c in cmd]
        recorded_cmds.append(cmd_list)
        return subprocess.CompletedProcess(args=cmd_list, returncode=0, stdout="", stderr="")

    res = install_desktop_app(
        app_paths,
        include_desktop=True,
        home_dir=home,
        platform_name="win32",
        python_executable=venv_bin / "python.exe",
        command_runner=_fake_runner,
    )

    assert res.platform == "win32"
    assert res.primary_path.name == "LightBox.lnk"
    assert "Start Menu" in str(res.primary_path)
    assert res.desktop_path == home / "Desktop" / "LightBox.lnk"
    assert res.launcher_kind == "gui_script"
    assert res.launcher_target.name == "lightbox-app.exe"
    assert res.icon_path is not None and res.icon_path.is_file() and res.icon_path.suffix == ".ico"

    # Two PowerShell shortcut creation invocations (Start Menu + Desktop), ZERO .vbs files anywhere!
    assert len(recorded_cmds) == 2
    for cmd in recorded_cmds:
        assert cmd[0] == "powershell.exe"
        ps_body = cmd[-1]
        assert "WScript.Shell" in ps_body
        assert "CreateShortcut(" in ps_body
        assert "lightbox-app.exe" in ps_body
        assert "$sc.Arguments = ''" in ps_body
        # Escaped single quote in user home directory name
        assert "O''Connor Home" in ps_body
        assert ".vbs" not in ps_body.lower()

    assert list(tmp_path.rglob("*.vbs")) == [], "Windows installer must NEVER generate .vbs files"


def test_install_windows_lnk_falls_back_to_pythonw_module(tmp_path: Path) -> None:
    home = tmp_path / "win_home_2"
    home.mkdir()
    venv_bin = _make_fake_venv(tmp_path / "win_venv_2", files=("python.exe", "pythonw.exe"))
    app_paths = get_app_paths(tmp_path / "app_data_win_2")

    recorded_scripts: list[str] = []

    def _fake_runner(cmd: Any) -> subprocess.CompletedProcess[str]:
        recorded_scripts.append(str(cmd[-1]))
        return subprocess.CompletedProcess(args=list(cmd), returncode=0, stdout="", stderr="")

    res = install_desktop_app(
        app_paths,
        include_desktop=False,
        home_dir=home,
        platform_name="win32",
        python_executable=venv_bin / "python.exe",
        command_runner=_fake_runner,
    )

    assert res.launcher_kind == "pythonw_module"
    assert res.launcher_target.name == "pythonw.exe"
    assert res.launcher_args == ("-m", "lightbox.cli", "app")
    assert len(recorded_scripts) == 1
    assert "pythonw.exe" in recorded_scripts[0]
    assert "-m lightbox.cli app" in recorded_scripts[0]
    assert ".vbs" not in recorded_scripts[0].lower()


def test_install_linux_desktop_entry(tmp_path: Path) -> None:
    home = tmp_path / "linux_home"
    home.mkdir()
    venv_bin = _make_fake_venv(tmp_path / "My Venv", files=("python", "lightbox-app"))
    app_paths = get_app_paths(tmp_path / "app_data_linux")

    res = install_desktop_app(
        app_paths,
        include_desktop=True,
        home_dir=home,
        platform_name="linux",
        python_executable=venv_bin / "python",
    )

    assert res.platform == "linux"
    assert res.primary_path == home / ".local" / "share" / "applications" / "lightbox.desktop"
    assert res.desktop_path == home / "Desktop" / "lightbox.desktop"
    assert res.icon_path is not None and res.icon_path.is_file() and res.icon_path.suffix == ".png"

    content = res.primary_path.read_text(encoding="utf-8")
    assert "[Desktop Entry]" in content
    assert "Terminal=false" in content
    assert f'Exec="{(venv_bin / "lightbox-app").resolve()}"' in content
    assert f"Icon={res.icon_path}" in content

    assert res.desktop_path.is_file()
    desktop_mode = stat.S_IMODE(res.desktop_path.stat().st_mode)
    assert desktop_mode & stat.S_IXUSR


def test_cli_install_app_command_and_service_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIGHTBOX_HOME", str(tmp_path / "cli_install_home"))
    runner = CliRunner()

    help_res = runner.invoke(cli_app, ["install-app", "--help"])
    assert help_res.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", help_res.output)
    assert "--desktop" in plain
    assert "--dir" in plain

    custom_dir = tmp_path / "custom_apps"
    res = runner.invoke(cli_app, ["install-app", "--dir", str(custom_dir), "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert Path(payload["primary_path"]).exists()
    assert payload["launcher_target"]

    # Service layer wrapper works directly too
    with LightboxServices(app_paths=get_app_paths(tmp_path / "svc_home")) as svcs:
        svc_res = svcs.system.install_desktop_app(
            target_dir=tmp_path / "svc_apps",
            platform_name="linux",
            home_dir=tmp_path / "svc_fake_home",
        )
        assert svc_res.primary_path.is_file()
