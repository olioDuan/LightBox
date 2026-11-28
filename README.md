# LightBox

[![PyPI](https://img.shields.io/pypi/v/lightbox-cull?style=flat-square&logo=pypi&logoColor=white&color=18181b)](https://pypi.org/project/lightbox-cull/)
[![CI](https://img.shields.io/github/actions/workflow/status/olioDuan/LightBox/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/olioDuan/LightBox/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-18181b?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Zero Hard Delete](https://img.shields.io/badge/Safety-Zero%20Hard%20Delete-059669?style=flat-square)](scripts/check_no_delete.py)
[![MCP + Skill](https://img.shields.io/badge/MCP-Server%20%2B%20Skill-2563eb?style=flat-square)](#option-2-use-with-claude-cursor-or-codex-skill--mcp)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-52525b?style=flat-square)](LICENSE)

English | [简体中文](README.zh-CN.md)

Local photo culling tool for photographers. Runs offline by default, never deletes original files, and works either as a standalone desktop/web app or through Claude, Cursor, and Codex via Skill and MCP.

## What it does

- **Deterministic culling (offline, no API key)**: flags out-of-focus shots, bad exposure, closed eyes, and groups bursts with a recommended pick. Handles RAW+JPEG and Live Photo pairs together.
- **Reversible quarantine**: never calls hard delete on your photo folders. Moving rejects requires an explicit approval step and can be restored in one click.
- **Local search and people**: SigLIP 2 natural-language photo search, similar-photo lookup, face clustering, and per-person export.
- **Optional AI review**: connect OpenAI, Anthropic, Gemini, any OpenAI-compatible endpoint, or local Ollama in the app drawer (`Cmd+K` / `Ctrl+K`) to review borderline shots or chat through a shoot.

## Option 1: Standalone Desktop / Web App

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

Run immediately in your browser without cloning:

```bash
uvx --from lightbox-cull lightbox ui
```

Or launch the native desktop window:

```bash
uvx --from 'lightbox-cull[desktop]' lightbox app
```

To install permanently and create a double-click app launcher (macOS `~/Applications/LightBox.app`, Windows `.lnk`, Linux `.desktop`):

```bash
uv tool install 'lightbox-cull[desktop]'
lightbox install-app
```

Try it on generated sample photos without touching your own library:

```bash
uvx --from lightbox-cull lightbox demo
```

Optional extras: add `heic` (e.g. `lightbox-cull[desktop,heic]`) for HEIC support. On Linux, desktop mode requires system GTK (`python3-gi gir1.2-webkit2-4.1`) or `PySide6`.

## Option 2: Use with Claude, Cursor, or Codex (Skill + MCP)

### Direct chat culling (zero install)

Load `skills/lightbox-cull/SKILL.md` in your AI client (Codex, Claude Code, Cursor, etc.). You can drop a small batch of photos directly into chat and ask it to pick the best shots or flag blurry ones without installing LightBox.

### Local folder workflows (MCP)

Claude Desktop / Cursor (the in-app **Settings -> Use in Claude / Cursor** tab also generates ready-to-copy snippets for each client):

```json
{
  "mcpServers": {
    "lightbox": {
      "command": "uvx",
      "args": ["--from", "lightbox-cull", "lightbox", "mcp"]
    }
  }
}
```

Claude Code:

```bash
claude mcp add lightbox -- uvx --from lightbox-cull lightbox mcp
```

GUI apps on macOS/Windows may not inherit your shell `PATH`; if needed, replace `"uvx"` with the full path from `which uvx`.

## Development

```bash
git clone https://github.com/olioDuan/LightBox.git
cd LightBox
uv sync
uv run ruff check .
uv run python scripts/check_no_delete.py
uv run lint-imports
uv run mypy src
uv run pytest
```

Frontend source lives in `web/` (`cd web && npm install && npm run build` updates `src/lightbox/web_dist/`).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party component and optional model notices.
