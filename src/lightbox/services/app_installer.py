"""Cross-platform desktop shortcut/app-bundle installer (`lightbox install-app`, Step 5).

Generates native GUI launch entries bound to the absolute path of the current environment's
`lightbox-app` executable (avoiding GUI PATH inheritance failures):
- macOS: `~/Applications/LightBox.app` (+ optional Desktop symlink via `--desktop`)
- Windows: Start Menu / Desktop `LightBox.lnk` pointing directly to `lightbox-app.exe`
  (or `pythonw.exe -m lightbox.cli app`); NEVER generates `.vbs` scripts.
- Linux: `~/.local/share/applications/lightbox.desktop` (`Terminal=false`, + optional Desktop copy)

Obeys all Lightbox safety invariants: zero hard deletion (`unlink`/`remove`/`rmtree`/`replace`).
"""

from __future__ import annotations

import importlib.util
import os
import plistlib
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw

from lightbox import __version__
from lightbox.paths import AppPaths

LauncherKind = Literal["gui_script", "pythonw_module", "python_module"]
PlatformName = Literal["darwin", "win32", "linux"]
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class ResolvedLauncher:
    """Resolved absolute desktop executable and arguments for the current Python environment."""

    target: Path
    args: tuple[str, ...]
    kind: LauncherKind


@dataclass(frozen=True, slots=True)
class InstallAppResult:
    """Structured result returned by `install_desktop_app` and `lightbox install-app`."""

    platform: PlatformName
    primary_path: Path
    desktop_path: Path | None
    icon_path: Path | None
    launcher_target: Path
    launcher_args: tuple[str, ...]
    launcher_kind: LauncherKind
    pywebview_installed: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "primary_path": str(self.primary_path),
            "desktop_path": str(self.desktop_path) if self.desktop_path is not None else None,
            "icon_path": str(self.icon_path) if self.icon_path is not None else None,
            "launcher_target": str(self.launcher_target),
            "launcher_args": list(self.launcher_args),
            "launcher_kind": self.launcher_kind,
            "pywebview_installed": self.pywebview_installed,
            "notes": list(self.notes),
        }


def _normalize_platform(platform_name: str | None) -> PlatformName:
    raw = (platform_name or sys.platform).lower()
    if raw == "darwin" or raw.startswith("mac"):
        return "darwin"
    if raw.startswith("win"):
        return "win32"
    return "linux"


def _is_pywebview_installed() -> bool:
    try:
        return importlib.util.find_spec("webview") is not None
    except Exception:
        return False


def _render_app_icon_image(size: int = 256) -> Image.Image:
    """Render a crisp minimalist LightBox viewfinder icon (dark zinc squircle + white lens mark)."""
    scale = 2
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = int(s * 0.06)
    radius = int(s * 0.22)
    # Neutral dark zinc squircle matching LightBox monochrome surface aesthetic
    draw.rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=radius,
        fill=(24, 24, 27, 255),
        outline=(63, 63, 70, 255),
        width=max(2, int(s * 0.014)),
    )

    # Four minimalist viewfinder corner brackets
    inset = int(s * 0.23)
    arm = int(s * 0.14)
    stroke = max(4, int(s * 0.036))
    fg = (250, 250, 250, 255)
    # Top-left
    draw.line([(inset, inset + arm), (inset, inset), (inset + arm, inset)], fill=fg, width=stroke)
    # Top-right
    draw.line(
        [(s - inset - arm, inset), (s - inset, inset), (s - inset, inset + arm)],
        fill=fg,
        width=stroke,
    )
    # Bottom-left
    draw.line(
        [(inset, s - inset - arm), (inset, s - inset), (inset + arm, s - inset)],
        fill=fg,
        width=stroke,
    )
    # Bottom-right
    draw.line(
        [(s - inset - arm, s - inset), (s - inset, s - inset), (s - inset, s - inset - arm)],
        fill=fg,
        width=stroke,
    )

    # Central crisp lens ring + subtle accent dot
    cx, cy = s // 2, s // 2
    r_outer = int(s * 0.15)
    draw.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        outline=fg,
        width=stroke,
    )
    r_dot = int(s * 0.045)
    draw.ellipse(
        [cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot],
        fill=(59, 130, 246, 255),
    )

    return img.resize((size, size), Image.Resampling.LANCZOS)


def _write_icon_file(path: Path, *, fmt: Literal["ICNS", "ICO", "PNG"]) -> Path | None:
    """Write an application icon file without deleting any pre-existing files."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = _render_app_icon_image(size=256)
        if fmt == "ICO":
            img.save(path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
        elif fmt == "ICNS":
            img.save(path, format="ICNS")
        else:
            img.save(path, format="PNG")
        return path
    except Exception:
        return None


def resolve_desktop_launcher(
    *,
    python_executable: Path | None = None,
    platform_name: str | None = None,
) -> ResolvedLauncher:
    """Resolve the absolute `lightbox-app` launcher next to `sys.executable`.

    Crucially avoids calling `.resolve()` on the Python symlink itself so virtual environments
    (e.g. `.venv/bin/python` symlinked to base Python) still locate `.venv/bin/lightbox-app`.
    """
    plat = _normalize_platform(platform_name)
    py_path = Path(python_executable) if python_executable is not None else Path(sys.executable)
    # Keep the venv bin directory intact (resolve parent directory only, never dereference python symlink)
    raw_bin_dir = py_path.parent
    bin_dir = raw_bin_dir.resolve() if raw_bin_dir.exists() else raw_bin_dir
    py_in_bin = bin_dir / py_path.name

    if plat == "win32":
        gui_exe = bin_dir / "lightbox-app.exe"
        if gui_exe.is_file():
            return ResolvedLauncher(target=gui_exe, args=(), kind="gui_script")
        pythonw_exe = bin_dir / "pythonw.exe"
        if pythonw_exe.is_file():
            return ResolvedLauncher(
                target=pythonw_exe,
                args=("-m", "lightbox.cli", "app"),
                kind="pythonw_module",
            )
        return ResolvedLauncher(
            target=py_in_bin,
            args=("-m", "lightbox.cli", "app"),
            kind="python_module",
        )

    gui_bin = bin_dir / "lightbox-app"
    if gui_bin.is_file():
        return ResolvedLauncher(target=gui_bin, args=(), kind="gui_script")
    return ResolvedLauncher(
        target=py_in_bin,
        args=("-m", "lightbox.cli", "app"),
        kind="python_module",
    )


def _default_command_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _ps_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _build_windows_shortcut_ps(
    *,
    shortcut_path: Path,
    target_path: Path,
    arguments: Sequence[str],
    working_dir: Path,
    icon_path: Path | None,
) -> str:
    arg_str = subprocess.list2cmdline(list(arguments)) if arguments else ""
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$ws = New-Object -ComObject WScript.Shell",
        f"$sc = $ws.CreateShortcut({_ps_single_quote(str(shortcut_path))})",
        f"$sc.TargetPath = {_ps_single_quote(str(target_path))}",
        f"$sc.Arguments = {_ps_single_quote(arg_str)}",
        f"$sc.WorkingDirectory = {_ps_single_quote(str(working_dir))}",
        "$sc.Description = 'LightBox — local-first photo culling and intelligent selection'",
    ]
    if icon_path is not None:
        lines.append(f"$sc.IconLocation = {_ps_single_quote(f'{icon_path},0')}")
    lines.append("$sc.Save()")
    return "; ".join(lines)


def _format_freedesktop_exec(target: Path, args: Sequence[str]) -> str:
    """Format a FreeDesktop Desktop Entry `Exec=` value with proper quoting."""
    parts = [str(target), *args]
    formatted: list[str] = []
    special = set(' \t\n"\'\\><~|&;$*?#()`')
    for part in parts:
        if any(ch in special for ch in part) or not part:
            escaped = (
                part.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("$", "\\$")
                .replace("`", "\\`")
            )
            formatted.append(f'"{escaped}"')
        else:
            formatted.append(part)
    return " ".join(formatted)


def _detect_editable_src_root() -> Path | None:
    """Return `<repo>/src` when running from an editable checkout (guards against macOS iCloud UF_HIDDEN `.pth`)."""
    pkg_dir = Path(__file__).resolve().parents[1]  # .../lightbox
    src_root = pkg_dir.parent
    if src_root.name == "src" and (src_root / "lightbox" / "__init__.py").is_file():
        return src_root
    return None


def _build_macos_launcher_script(
    launcher: ResolvedLauncher,
    *,
    icon_path: Path | None = None,
    editable_src_root: Path | None = None,
) -> str:
    quoted_target = shlex.quote(str(launcher.target))
    exec_args = " ".join(shlex.quote(a) for a in launcher.args)
    exec_line = f'exec "$TARGET" {exec_args} "$@"' if exec_args else 'exec "$TARGET" "$@"'
    icon_export = (
        f"export LIGHTBOX_APP_ICON={shlex.quote(str(icon_path))}\n"
        if icon_path is not None
        else ""
    )
    pypath_export = (
        f'export PYTHONPATH={shlex.quote(str(editable_src_root))}"${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        if editable_src_root is not None
        else ""
    )
    return (
        "#!/bin/sh\n"
        f"TARGET={quoted_target}\n"
        'if [ ! -x "$TARGET" ]; then\n'
        '  /usr/bin/osascript -e \'display alert "LightBox 启动失败" '
        'message "找不到 LightBox 运行环境，请在终端重新执行 lightbox install-app。" '
        'as critical buttons {"好"} default button "好"\' >/dev/null 2>&1 || true\n'
        "  exit 1\n"
        "fi\n"
        "export LIGHTBOX_APP_BUNDLE=1\n"
        f"{icon_export}"
        f"{pypath_export}"
        f"{exec_line}\n"
    )


def _install_macos_app(
    *,
    home_dir: Path,
    target_dir: Path | None,
    launcher: ResolvedLauncher,
    include_desktop: bool,
    notes: list[str],
) -> tuple[Path, Path | None, Path | None]:
    apps_dir = target_dir if target_dir is not None else (home_dir / "Applications")
    bundle_dir = apps_dir / "LightBox.app"
    contents_dir = bundle_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    icon_path = _write_icon_file(resources_dir / "LightBox.icns", fmt="ICNS")

    plist_data: dict[str, Any] = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": "LightBox",
        "CFBundleExecutable": "LightBox",
        "CFBundleIdentifier": "io.lightbox.desktop",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "LightBox",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSDesktopFolderUsageDescription": (
            "LightBox needs access to your Desktop folder when you scan photo folders located there."
        ),
        "NSPicturesFolderUsageDescription": (
            "LightBox needs access to your Pictures folder when you scan photo folders located there."
        ),
        "NSRemovableVolumesUsageDescription": (
            "LightBox needs access to external drives and memory cards when you scan photo folders located there."
        ),
    }
    if icon_path is not None:
        plist_data["CFBundleIconFile"] = "LightBox"

    (contents_dir / "Info.plist").write_bytes(plistlib.dumps(plist_data, fmt=plistlib.FMT_XML))

    exec_path = macos_dir / "LightBox"
    exec_path.write_text(
        _build_macos_launcher_script(
            launcher,
            icon_path=icon_path,
            editable_src_root=_detect_editable_src_root(),
        ),
        encoding="utf-8",
    )
    os.chmod(exec_path, 0o755)

    desktop_link: Path | None = None
    if include_desktop:
        desktop_dir = home_dir / "Desktop"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        candidate = desktop_dir / "LightBox.app"
        if candidate.is_symlink() or candidate.exists():
            # Zero hard-deletion invariant: never unlink an existing user Desktop entry
            if candidate.is_symlink() and Path(os.readlink(candidate)) == bundle_dir:
                desktop_link = candidate
            else:
                notes.append(
                    f"Desktop entry already exists at {candidate}; kept untouched without overwriting."
                )
        else:
            os.symlink(bundle_dir, candidate)
            desktop_link = candidate

    return bundle_dir, desktop_link, icon_path


def _install_windows_shortcuts(
    *,
    app_paths: AppPaths,
    home_dir: Path,
    target_dir: Path | None,
    launcher: ResolvedLauncher,
    include_desktop: bool,
    command_runner: CommandRunner,
    notes: list[str],
) -> tuple[Path, Path | None, Path | None]:
    app_paths.ensure_dirs()
    icon_path = _write_icon_file(app_paths.root / "assets" / "LightBox.ico", fmt="ICO")

    appdata_env = os.environ.get("APPDATA", "").strip()
    if target_dir is not None:
        start_menu_dir = target_dir
    elif appdata_env and home_dir == Path.home():
        start_menu_dir = Path(appdata_env) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    else:
        start_menu_dir = (
            home_dir / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        )
    start_menu_dir.mkdir(parents=True, exist_ok=True)

    primary_lnk = start_menu_dir / "LightBox.lnk"
    shortcut_targets: list[Path] = [primary_lnk]
    desktop_lnk: Path | None = None
    if include_desktop:
        desktop_dir = home_dir / "Desktop"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        desktop_lnk = desktop_dir / "LightBox.lnk"
        shortcut_targets.append(desktop_lnk)

    for lnk in shortcut_targets:
        ps_script = _build_windows_shortcut_ps(
            shortcut_path=lnk,
            target_path=launcher.target,
            arguments=launcher.args,
            working_dir=home_dir,
            icon_path=icon_path,
        )
        proc = command_runner(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_script,
            ]
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Failed to create Windows shortcut {lnk}: {err or proc.returncode}")

    if launcher.kind == "python_module":
        notes.append(
            "Neither lightbox-app.exe nor pythonw.exe was found; shortcut points to python.exe "
            "(reinstall lightbox-cull to generate the console-less lightbox-app.exe gui-script)."
        )

    return primary_lnk, desktop_lnk, icon_path


def _install_linux_desktop_entry(
    *,
    app_paths: AppPaths,
    home_dir: Path,
    target_dir: Path | None,
    launcher: ResolvedLauncher,
    include_desktop: bool,
    notes: list[str],
) -> tuple[Path, Path | None, Path | None]:
    apps_dir = (
        target_dir
        if target_dir is not None
        else (home_dir / ".local" / "share" / "applications")
    )
    apps_dir.mkdir(parents=True, exist_ok=True)

    icon_dest = (
        app_paths.root / "assets" / "lightbox.png"
        if target_dir is not None
        else (home_dir / ".local" / "share" / "icons" / "hicolor" / "256x256" / "apps" / "lightbox.png")
    )
    icon_path = _write_icon_file(icon_dest, fmt="PNG")

    exec_value = _format_freedesktop_exec(launcher.target, launcher.args)
    lines = [
        "[Desktop Entry]",
        "Version=1.0",
        "Type=Application",
        "Name=LightBox",
        "GenericName=Photo Culling Assistant",
        "Comment=Local-first photo culling and intelligent selection",
        f"Exec={exec_value}",
        f"Icon={icon_path}" if icon_path is not None else "Icon=camera-photo",
        "Terminal=false",
        "Categories=Graphics;Photography;",
        "StartupNotify=true",
        "",
    ]
    content = "\n".join(lines)

    primary_desktop = apps_dir / "lightbox.desktop"
    primary_desktop.write_text(content, encoding="utf-8")
    os.chmod(primary_desktop, 0o644)

    desktop_copy: Path | None = None
    if include_desktop:
        desktop_dir = home_dir / "Desktop"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        desktop_copy = desktop_dir / "lightbox.desktop"
        desktop_copy.write_text(content, encoding="utf-8")
        os.chmod(desktop_copy, 0o755)

    notes.append(
        "Linux pywebview requires system GTK bindings (`python3-gi gir1.2-webkit2-4.1`) or `PySide6`."
    )
    return primary_desktop, desktop_copy, icon_path


def install_desktop_app(
    app_paths: AppPaths,
    *,
    include_desktop: bool = False,
    target_dir: Path | None = None,
    home_dir: Path | None = None,
    platform_name: str | None = None,
    python_executable: Path | None = None,
    command_runner: CommandRunner | None = None,
) -> InstallAppResult:
    """Install a native OS double-click desktop launcher bound to the current environment."""
    plat = _normalize_platform(platform_name)
    resolved_home = (home_dir if home_dir is not None else Path.home()).expanduser()
    resolved_target = target_dir.expanduser() if target_dir is not None else None
    launcher = resolve_desktop_launcher(
        python_executable=python_executable,
        platform_name=plat,
    )
    pywebview_ok = _is_pywebview_installed()
    notes: list[str] = []

    if plat == "darwin":
        primary_path, desktop_path, icon_path = _install_macos_app(
            home_dir=resolved_home,
            target_dir=resolved_target,
            launcher=launcher,
            include_desktop=include_desktop,
            notes=notes,
        )
    elif plat == "win32":
        primary_path, desktop_path, icon_path = _install_windows_shortcuts(
            app_paths=app_paths,
            home_dir=resolved_home,
            target_dir=resolved_target,
            launcher=launcher,
            include_desktop=include_desktop,
            command_runner=command_runner or _default_command_runner,
            notes=notes,
        )
    else:
        primary_path, desktop_path, icon_path = _install_linux_desktop_entry(
            app_paths=app_paths,
            home_dir=resolved_home,
            target_dir=resolved_target,
            launcher=launcher,
            include_desktop=include_desktop,
            notes=notes,
        )

    if not pywebview_ok:
        notes.append(
            "pywebview is not installed in this environment yet; install desktop extras with "
            "`uv pip install 'lightbox-cull[desktop]'` before double-clicking the app icon."
        )
    notes.append(
        "GUI launches do not inherit shell exports from ~/.zshrc or ~/.bashrc; configure API keys "
        "in the in-app AI connection panel (stored in system keychain)."
    )

    return InstallAppResult(
        platform=plat,
        primary_path=primary_path,
        desktop_path=desktop_path,
        icon_path=icon_path,
        launcher_target=launcher.target,
        launcher_args=launcher.args,
        launcher_kind=launcher.kind,
        pywebview_installed=pywebview_ok,
        notes=tuple(notes),
    )


__all__ = [
    "CommandRunner",
    "InstallAppResult",
    "LauncherKind",
    "PlatformName",
    "ResolvedLauncher",
    "install_desktop_app",
    "resolve_desktop_launcher",
]
