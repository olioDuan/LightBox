---
name: lightbox-cull
description: >-
  Photographer-grade photo culling, blur detection, and shoot review skill for both direct chat
  uploads and local photo folders. Trigger whenever the user asks any photo-selection or
  photo-cleanup question — such as finding blurry or out-of-focus shots ("帮我把糊的照片挑出来",
  "哪些拍糊了", "帮我去掉废片", "find blurry/bad photos"), picking best shots ("帮我挑 3 张",
  "选几张发朋友圈", "pick the best photos"), comparing burst/similar frames ("连拍留哪张",
  "哪张没闭眼"), checking sharpness/focus, or scanning, searching, quarantining, and restoring a
  local photo folder. Works immediately with zero installation for chat-uploaded photos (no
  LightBox install, MCP, downloaded models, or extra API key needed), and uses LightBox MCP for
  full local folders, RAW files, and reversible batch quarantine.
---

# Lightbox Photo Culling (`lightbox-cull`)

Help photographers understand, clean up, and select from their photos — whether they uploaded a handful of shots directly in chat or pointed you at a local folder of thousands of RAWs.

**Core principle:** *First understand what the user is actually asking for, then apply photographer-grade judgment — never mis-discard a borderline or creatively intentional photo.*

---

## Step 1: Understand User Intent & Route by Input

Always do two quick checks before acting:

### 1A. Understand the User's Goal (Do Not Force a Rigid Template)
Photographers ask many different questions about a batch or folder of photos. Identify the user's primary intent first and structure your answer around it:

- **Find blurry / bad photos / clean up rejects** (e.g., *"帮我把糊的照片挑出来"*, *"哪些拍糊了或闭眼了"*, *"帮我去掉废片"*, *"find blurry or unusable shots"*):
  - **Primary answer first (required):** List the clear blurry/unusable rejects with filename or `#N` and exact visual reason.
  - **Borderline / uncertain second (required):** Separate shots with ambiguous focus, possible creative blur, or preview-resolution uncertainty so nothing borderline is wrongly flagged as waste.
  - **Usable keepers summary:** Briefly confirm which shots look clean/usable.
- **Pick best / curate N shots** (e.g., *"帮我挑 3 张"*, *"选几张发朋友圈"*, *"pick the best 3 for an album"*):
  - **Primary answer first (required):** Recommended picks with filename or `#N` and concrete reasons (focus, expression, light, composition, variety, vibe fit).
  - **Borderline / uncertain second (required):** Close runner-ups or shots that need a 100% zoom check.
  - **Clear misses / burst duplicates (optional):** Brief note if helpful to explain exclusions.
- **Compare burst / similar frames** (e.g., *"这几张连拍选哪张"*, *"哪张眼睛最清晰/没闭眼"*, *"deduplicate these"*):
  - **Primary answer first (required):** Best frame(s) and direct frame-by-frame tradeoffs.
  - **Borderline / uncertain second (required):** Any close runner-up worth checking at 100%.
  - **Redundant or flawed frames:** Briefly note why the others lose.
- **Open-ended review / folder Q&A** (e.g., *"帮我看看这批照片怎么样"*, *"整理一下这个文件夹"*, *"这组人像合焦情况如何"*):
  - Answer the specific question directly, summarize overall quality, highlight standout shots and clear blurry/bad shots, and separate uncertain photos.

### 1B. Route by Input Source (Never Block Chat Uploads)

| User Input | Mode | Prerequisites |
| :--- | :--- | :--- |
| User **uploaded photos directly in chat** | **Mode A: Zero-Install Chat Review** | **None.** Works with no LightBox installed, no MCP configured, no LightBox models downloaded, and no extra API key. |
| User provides a **local folder path** or asks to scan a whole directory, handle RAW/HEIC/Live Photo pairs, search a library, batch quarantine, or restore | **Mode B: Local Folder via LightBox MCP** | Local LightBox MCP server (`lightbox_*` tools) or local `lightbox` CLI fallback. |

> **Critical routing rule:** When the user uploaded photos directly in chat, **stay in Mode A** and inspect the images immediately. **Never** demand that the user install LightBox, configure MCP, or run `lightbox_scan_folder` before answering a direct chat upload request.

---

## Mode A: Zero-Install Chat Review (Direct Image Uploads)

Use this mode whenever photos are attached in the conversation. It works on any client supporting Skills + image input, with zero external setup and no dependency on T10 or T13.

### Workflow
1. **Identify every photo unambiguously**: Use the exact filename if visible, and/or the 1-based upload index (`#1`, `#2`, or `#2 (IMG_4821.JPG)`). If filenames are hidden, pair `#N` with a short 3–5 word visual cue so the user never has to guess which photo you mean.
2. **Inspect every uploaded photo directly** using [references/culling_rubric.md](references/culling_rubric.md):
   - **Focus & sharpness**: Distinguish genuine camera shake / missed focus from intentional shallow depth of field, panning, long exposure, or ICM.
   - **Expression & timing**: Check eyes open, natural expression, peak action across similar frames.
   - **Composition & exposure**: Flag accidental extreme black/white misfires or severe framing accidents; respect deliberate low-key silhouettes or high-key minimalism.
   - **Burst deduplication**: Avoid wasting slots on near-identical frames when picking N shots; when finding rejects, distinguish *technical blur/flaws* from *sharp burst duplicates*.
3. **Be honest about preview resolution**: Chat clients often downscale uploaded images. If 100% eye focus or micro-shake cannot be confirmed from the preview, do **not** guess — put the photo in **拿不准的照片 (Uncertain / Borderline)** and tell the user what to check at 100% zoom.
4. **Adapt section order to the user's intent** (templates in [references/culling_rubric.md](references/culling_rubric.md)):
   - **Finding blurry / bad photos** → **明显废片 / 模糊照片（必选，置顶）** → **拿不准的照片（必选）** → 清晰保留片简述。
   - **Picking best photos** → **推荐照片（必选，置顶）** → **拿不准的照片（必选）** → 明显废片 / 连拍重复提醒（可选）。
   - **Comparing bursts / answering specific questions** → Answer the user's exact question first, always including filenames/indices, concrete visual reasons, and any uncertain/borderline shots.
5. **Graceful handoff (only when needed)**: Only if the user asks to process a whole local directory, handle RAW files the chat cannot preview, or safely move/restore files on disk, briefly mention Mode B (LightBox MCP).

---

## Mode B: Local Photo Folder Workflow (LightBox MCP)

Use this mode when the user wants to scan a local folder, find blurry/bad shots across a shoot, pick keepers, handle RAW/HEIC/Live Photo atomic pairs, compare bursts at scale, search a library, or batch quarantine / restore files.

### Four Hard Rules (Local Folder Mode Only)
1. **Always start with `lightbox_scan_folder`**: For any local photo directory, call `lightbox_scan_folder` first. **Never** write ad-hoc Shell/Python scripts to walk photo folders or dump an entire directory of images into model context. *(Scoped strictly to local folder workflows — never blocks Mode A direct chat uploads.)*
2. **Read preferences first**: Before making subjective culling or curation decisions on a local session, read `lightbox://memory/preferences`.
3. **Protect context budget (`> 8` photos → Web UI)**: Inspect at most **8** uncertain photos or burst strips directly via `lightbox_get_photo` / `lightbox_compare_burst`. When more than 8 photos require manual visual verification, **stop fetching images** and direct the user to open the tokenless `review_url` in the local LightBox Web UI.
4. **Never hard-delete or manually move photos**: **Never** run `rm`, `mv`, or deletion scripts on user photo directories. All quarantine moves must go through:
   - `lightbox_create_plan` (drafts immutable `plan_hash` + runs deterministic Verifier checks; moves 0 files),
   - **Explicit human approval** (default: user clicks the tokenless `approve_url` in LightBox Web UI; alternatives: MCP Elicitation prompt if supported, or interactive terminal `lightbox approve <plan_id>`),
   - `lightbox_apply_plan` (same-volume atomic move into `Lightbox_Quarantine/`).
   - Any moved batch or individual photo can always be reversed via `lightbox_restore_plan`.

### Matching User Intent to Local MCP Tools
- **"帮我把糊的照片挑出来 / 去掉废片" (Find blurry / clear obvious rejects)**:
  1. `lightbox_scan_folder(folder_path)` → inspect `red` (obvious rejects) vs `yellow` (uncertain) vs `exempt` (protected).
  2. List obvious rejects via `lightbox_list_photos(session_id, bucket="red")` and borderline shots via `bucket="yellow"`.
  3. Optionally inspect `<= 8` borderline crops (`lightbox_get_photo(..., image="face_crop" | "sharpest_crop")`) or guide `> 8` to `review_url`.
  4. If the user wants to remove rejects, draft a plan via `lightbox_create_plan` and hand off the tokenless `approve_url` (never delete directly).
- **"帮我挑 N 张 / 选发朋友圈" (Pick best / curate)**:
  1. `lightbox_scan_folder` (if not already scanned) + read `lightbox://memory/preferences`.
  2. Shortlist from `normal` / `exempt` via `lightbox_list_photos` / `lightbox_search_photos`, deduplicate with `lightbox_find_similar`, inspect top candidates (`<= 8`) via `lightbox_get_photo`, and present picks + reasons + uncertain runner-ups.
- **"这几组连拍帮我选一下" (Burst comparison)**:
  1. Call `lightbox_compare_burst(group_id)` to inspect numbered contact sheet + 1:1 horizontal face strip and sharpness metrics.
  2. Record non-destructive recommendation via `lightbox_recommend_burst_pick(group_id, photo_id, reason=...)` (unconfirmed burst non-picks are kept safe until human confirmation).
- **"把刚才移走的还原" (Restore)**:
  1. Read `lightbox://plan/{id}` or call `lightbox_restore_plan(plan_id, photo_ids=...)`.

### CLI Fallback (Local Install Without MCP)
If local LightBox is installed (or `uvx` is available) without MCP, use [scripts/cli_fallback.sh](scripts/cli_fallback.sh) for `scan`, `demo`, `ui`, `approve`, or `doctor` (never raw `rm`/`mv`).
