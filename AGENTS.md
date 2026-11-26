# AGENTS.md — Lightbox Agent & Skill Rules

This file provides repository-level instructions for coding and multimodal agents (Codex, Claude Code, Cursor, Antigravity, etc.) and serves as the auto-discovery fallback alongside `skills/lightbox-cull/SKILL.md` (SPEC §8.4, ADR-000 Item 10).

---

## 1. Dual-Mode Photo Culling (`skills/lightbox-cull/SKILL.md`)

Lightbox supports two photo culling entry points. Always route by user input:

### Mode A — Direct Chat Photo Upload (Zero-Install, Immediate Try-Out)
- **When**: The user uploads a small batch of photos directly in chat and asks any photo-review question — e.g. *"帮我把糊的照片挑出来"*, *"帮我去掉废片"*, *"帮我挑 3 张"*, *"这几张连拍选哪张"*, *"pick the best 3 shots"*, *"find blurry photos"*.
- **Requirements**: **Zero installation.** Must work on any client supporting Skills + image input even with no LightBox installed, no MCP configured, no LightBox models downloaded, no extra API key, and without depending on T10 or T13.
- **Behavior**: Understand the user's intent first, and do **NOT** require `lightbox_scan_folder` or MCP setup. Inspect the uploaded images directly using `skills/lightbox-cull/references/culling_rubric.md`, adapting output order to the request:
  - When finding blurry/reject shots: put **明显废片 / 模糊照片** FIRST and REQUIRED (filenames or `#N` + visual reasons), followed by **拿不准的照片** (borderline/creative blur) and usable keepers.
  - When picking standout shots: put **推荐照片** FIRST and REQUIRED (filenames or `#N` + visual reasons), followed by **拿不准的照片** and optional notes on clear misses/burst duplicates.

### Mode B — Local Photo Folder (LightBox MCP Workflow)
- **When**: The user supplies a local directory path or asks to scan an entire shoot folder, process RAW/HEIC/Live Photo pairs, detect bursts at scale, search a library, or batch quarantine / restore files.
- **Four Hard Rules for Local Folder Operations**:
  1. **Scan via MCP first**: Always call `lightbox_scan_folder` first for local photo directories. Never write ad-hoc Shell/Python scripts to walk photo folders or feed an entire folder of images into model context. *(Scoped to local folders only — never blocks Mode A direct chat uploads.)*
  2. **Read preferences first**: Before subjective culling on a local session, read `lightbox://memory/preferences`.
  3. **Respect context budget (`> 8` photos → Web UI)**: For manual visual verification of more than 8 photos, direct the user to open the tokenless `review_url` in the local LightBox Web UI instead of calling `lightbox_get_photo` repeatedly.
  4. **Never hard-delete or manually move photos**: Never run `rm` or `mv` on user photo directories. Always draft an immutable plan via `lightbox_create_plan`, obtain explicit human approval (tokenless `approve_url` in Web UI, MCP Elicitation, or interactive `lightbox approve <plan_id>`), apply via `lightbox_apply_plan`, and remember all moves are reversible via `lightbox_restore_plan`.

---

## 2. Enabling the Skill & MCP in Common Clients

### Skill Installation / Discovery
- **Direct Skill folder**: `skills/lightbox-cull/` (contains `SKILL.md`, `references/culling_rubric.md`, `scripts/cli_fallback.sh`).
- **Codex**: Copy or symlink `skills/lightbox-cull` into `~/.codex/skills/lightbox-cull` (or `.agents/skills/lightbox-cull`); this root `AGENTS.md` provides automatic repo-level fallback.
- **Claude Desktop / Claude Code / Cursor**: Enable or link `skills/lightbox-cull/SKILL.md` in your skills directory. Mode A works immediately with no MCP configuration.

### Optional MCP Server Configuration (for Mode B Local Folders)
- **Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
  ```json
  {
    "mcpServers": {
      "lightbox": {
        "command": "uvx",
        "args": ["--from", "lightbox-cull", "lightbox", "mcp", "--transport", "stdio"]
      }
    }
  }
  ```
  *(Local repo dev: `"command": "uv"`, `"args": ["--directory", "/path/to/LightBox", "run", "lightbox", "mcp"]`. If GUI apps cannot resolve `uv` / `uvx` on `PATH`, use the absolute path from `which uv` / `which uvx`.)*
- **Codex** (`~/.codex/config.toml`):
  ```toml
  [mcp_servers.lightbox]
  command = "uvx"
  args = ["--from", "lightbox-cull", "lightbox", "mcp", "--transport", "stdio"]
  ```
- **Claude Code**: `claude mcp add lightbox -- uvx --from lightbox-cull lightbox mcp`

---

## 3. Repository Engineering Invariants (When Editing Code)

1. **Zero hard deletion**: Never call `os.remove`, `os.unlink`, `os.rmdir`, `os.rename`, `os.replace`, `shutil.rmtree`, `shutil.move`, `.unlink()`, `.rmdir()`, or `.rename()` outside whitelisted `SafeFS` / `cache_lru`. Both `ruff` (`TID251`) and `scripts/check_no_delete.py` enforce this.
2. **Single source of truth (`lightbox.services`)**: `lightbox.api`, `lightbox.cli`, and `lightbox.mcp` must not import `lightbox.store`, `lightbox.fs`, or `lightbox.core` directly (`lint-imports` enforces this).
3. **Tokenless URLs**: Never embed authentication tokens in `review_url` or `approve_url` returned by MCP or CLI.
4. **Zero-LLM ready**: Core deterministic scan, review, Web UI, SafeFS quarantine/restore, and CI test suite must pass completely offline with zero API keys.
