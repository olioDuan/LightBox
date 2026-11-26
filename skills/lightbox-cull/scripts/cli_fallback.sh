#!/usr/bin/env bash
# Safe CLI fallback for Lightbox Mode B (local folder culling without MCP).
# Strictly invokes Lightbox CLI commands only — never runs rm or mv on user photos.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SKILL_REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"

is_lightbox_project_dir() {
  local dir="$1"
  local pyproject="$dir/pyproject.toml"
  [[ -f "$pyproject" ]] && grep -Eq '^[[:space:]]*name[[:space:]]*=[[:space:]]*"lightbox-cull"[[:space:]]*$' "$pyproject"
}

LIGHTBOX_CMD=()
resolve_lightbox_cmd() {
  if command -v lightbox >/dev/null 2>&1; then
    LIGHTBOX_CMD=(lightbox)
  elif command -v uv >/dev/null 2>&1 && is_lightbox_project_dir "$PWD"; then
    LIGHTBOX_CMD=(uv run lightbox)
  elif command -v uv >/dev/null 2>&1 && is_lightbox_project_dir "$SKILL_REPO_ROOT"; then
    LIGHTBOX_CMD=(uv --directory "$SKILL_REPO_ROOT" run lightbox)
  elif command -v uvx >/dev/null 2>&1; then
    LIGHTBOX_CMD=(uvx --from lightbox-cull lightbox)
  else
    echo "Error: Neither 'lightbox', 'uv' (in a lightbox-cull checkout), nor 'uvx' was found in PATH." >&2
    echo "Note: For small photo batches uploaded directly in chat, no installation is needed (Mode A)." >&2
    echo "For local folders (Mode B), install uv (https://docs.astral.sh/uv/) or lightbox-cull." >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage: cli_fallback.sh <command> [args...]

Safe local-folder Lightbox CLI commands (Mode B fallback when MCP is unavailable):
  scan <folder> [--sensitivity conservative|standard] [--draft-obvious] [--json]
  demo [--json]
  ui [--port PORT] [--no-browser]
  approve <plan_id> [--apply]
  doctor [--json]
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CMD="$1"
shift

case "$CMD" in
  scan|demo|ui|approve|doctor)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Error: Unsupported command '$CMD'. Allowed: scan, demo, ui, approve, doctor." >&2
    usage
    exit 1
    ;;
esac

resolve_lightbox_cmd
exec "${LIGHTBOX_CMD[@]}" "$CMD" "$@"
