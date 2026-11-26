"""Custom AST static gate forbidding file deletion primitives (SPEC §10.2(1)).

Complements ruff TID251 by intercepting instance method calls `.unlink()` and
`.rmdir()` (which TID251 cannot catch on `Path` instances), as well as direct
calls to banned deletion/overwrite APIs.

Only explicitly whitelisted `<APP_DATA_DIR>/cache/` LRU cleanup modules may
invoke `.unlink()` / `.rmdir()`.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCAN_DIRS = ("src", "scripts", "eval")

# Narrow whitelist for `<APP_DATA_DIR>/cache/` LRU cleanup ONLY (SPEC §10.2(1)).
# T04 should isolate cache eviction into `src/lightbox/core/cache_lru.py` (invoked
# by `core/preview.py`) and enforce a runtime `is_relative_to(cache_dir)` guard
# before unlinking any cached file.
CACHE_CLEANUP_WHITELIST: frozenset[str] = frozenset(
    {
        "src/lightbox/core/cache_lru.py",
    }
)

# Narrow whitelist for SafeFS non-overwriting rename primitive ONLY (SPEC §10.2(2)).
SAFE_FS_RENAME_WHITELIST: frozenset[str] = frozenset(
    {
        "src/lightbox/fs/safe_fs.py",
    }
)

FORBIDDEN_DELETE_ATTRS: frozenset[str] = frozenset({"unlink", "rmdir"})
FORBIDDEN_RENAME_ATTRS: frozenset[str] = frozenset({"rename"})
FORBIDDEN_MODULE_CALLS: dict[str, frozenset[str]] = {
    "os": frozenset({"remove", "unlink", "rmdir", "replace"}),
    "shutil": frozenset({"rmtree", "move"}),
}
SAFE_FS_ONLY_MODULE_CALLS: dict[str, frozenset[str]] = {
    "os": frozenset({"rename"}),
}


@dataclass(frozen=True, slots=True)
class Violation:
    path: Path
    line: int
    col: int
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.message}"


def _relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class _NoDeleteVisitor(ast.NodeVisitor):
    def __init__(
        self,
        file_path: Path,
        *,
        allow_cache_unlink: bool,
        allow_safe_fs_rename: bool,
    ) -> None:
        self.file_path = file_path
        self.allow_cache_unlink = allow_cache_unlink
        self.allow_safe_fs_rename = allow_safe_fs_rename
        self.violations: list[Violation] = []
        # Track module aliases (`import os as o`) and direct symbol imports (`from os import remove as r`)
        self._module_aliases: dict[str, str] = {}
        self._banned_symbol_aliases: dict[str, str] = {}
        self._safe_fs_only_aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        tracked = set(FORBIDDEN_MODULE_CALLS) | set(SAFE_FS_ONLY_MODULE_CALLS)
        for alias in node.names:
            if alias.name in tracked:
                local_name = alias.asname or alias.name
                self._module_aliases[local_name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module in FORBIDDEN_MODULE_CALLS:
            banned = FORBIDDEN_MODULE_CALLS[module]
            for alias in node.names:
                if alias.name in banned:
                    local_name = alias.asname or alias.name
                    self._banned_symbol_aliases[local_name] = f"{module}.{alias.name}"
        if module in SAFE_FS_ONLY_MODULE_CALLS:
            safe_fs_only = SAFE_FS_ONLY_MODULE_CALLS[module]
            for alias in node.names:
                if alias.name in safe_fs_only:
                    local_name = alias.asname or alias.name
                    self._safe_fs_only_aliases[local_name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            # 1. Check `module.banned_func(...)`
            if isinstance(func.value, ast.Name):
                orig_mod = self._module_aliases.get(func.value.id)
                if orig_mod and func.attr in FORBIDDEN_MODULE_CALLS.get(orig_mod, frozenset()):
                    self.violations.append(
                        Violation(
                            path=self.file_path,
                            line=node.lineno,
                            col=node.col_offset,
                            message=f"Forbidden call `{orig_mod}.{func.attr}()` (SPEC §10.2).",
                        )
                    )
                    self.generic_visit(node)
                    return
                if orig_mod and func.attr in SAFE_FS_ONLY_MODULE_CALLS.get(orig_mod, frozenset()):
                    if not self.allow_safe_fs_rename:
                        self.violations.append(
                            Violation(
                                path=self.file_path,
                                line=node.lineno,
                                col=node.col_offset,
                                message=(
                                    f"Forbidden call `{orig_mod}.{func.attr}()` outside SafeFS "
                                    "(SPEC §10.2(2))."
                                ),
                            )
                        )
                    self.generic_visit(node)
                    return

            # 2. Check `.unlink()` / `.rmdir()` on any instance
            if func.attr in FORBIDDEN_DELETE_ATTRS and not self.allow_cache_unlink:
                self.violations.append(
                    Violation(
                        path=self.file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            f"Forbidden `.{func.attr}()` call outside cache LRU whitelist "
                            "(SPEC §10.2)."
                        ),
                    )
                )
            # 3. Check `.rename()` on any instance (overwrites on POSIX)
            elif func.attr in FORBIDDEN_RENAME_ATTRS and not self.allow_safe_fs_rename:
                self.violations.append(
                    Violation(
                        path=self.file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            f"Forbidden `.{func.attr}()` call outside SafeFS whitelist "
                            "(SPEC §10.2(2))."
                        ),
                    )
                )
        elif isinstance(func, ast.Name):
            if func.id in self._banned_symbol_aliases:
                qualified = self._banned_symbol_aliases[func.id]
                self.violations.append(
                    Violation(
                        path=self.file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"Forbidden call `{qualified}()` (SPEC §10.2).",
                    )
                )
            elif func.id in self._safe_fs_only_aliases and not self.allow_safe_fs_rename:
                qualified = self._safe_fs_only_aliases[func.id]
                self.violations.append(
                    Violation(
                        path=self.file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"Forbidden call `{qualified}()` outside SafeFS (SPEC §10.2(2)).",
                    )
                )

        self.generic_visit(node)


def check_file(file_path: Path) -> list[Violation]:
    """Parse and inspect a single Python source file for forbidden deletion/rename calls."""
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    rel = _relative_posix(file_path)
    visitor = _NoDeleteVisitor(
        file_path,
        allow_cache_unlink=(rel in CACHE_CLEANUP_WHITELIST),
        allow_safe_fs_rename=(rel in SAFE_FS_RENAME_WHITELIST),
    )
    visitor.visit(tree)
    return visitor.violations


def collect_python_files(targets: list[Path]) -> list[Path]:
    """Collect all `.py` files from the provided file/directory paths."""
    files: list[Path] = []
    for target in targets:
        if target.is_file() and target.suffix == ".py":
            files.append(target)
        elif target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
    return files


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args:
        targets = [Path(arg) for arg in args]
    else:
        targets = [REPO_ROOT / d for d in DEFAULT_SCAN_DIRS if (REPO_ROOT / d).exists()]

    all_violations: list[Violation] = []
    for py_file in collect_python_files(targets):
        all_violations.extend(check_file(py_file))

    if all_violations:
        for violation in all_violations:
            print(violation.format(), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
