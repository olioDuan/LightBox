# ADR-000: Pre-Implementation Ecosystem Preflight Checks (T00)

- **Date**: 2026-11-24
- **Status**: Accepted
- **Reference**: `SPEC §12` (Pre-Implementation Checks)
- **Verification Script**: [scripts/preflight_check.py](../../scripts/preflight_check.py)
- **Test Environment**: macOS (Apple Silicon `aarch64-apple-darwin`), `uv 0.13.0`, `uv`-managed Python `3.12.15`, SQLite `3.53.1`.

## Summary of Results (Items 1–7)

| # | Check Item | Result | Decision & Fallback Strategy |
| :-: | :--- | :---: | :--- |
| 1 | PyPI package name `lightbox` availability | **启用回退**（已占名） | PyPI 已存在 `Lightbox` (`2.1`, "Lightbox photo display widget")，且 PyPI 名称大小写不敏感。发行包名改为 `lightbox-cull`（保持 Python 导入包名为 `lightbox`、CLI 命令名为 `lightbox`），安装/MCP 文档统一使用 `uvx --from lightbox-cull lightbox`。 |
| 2 | `uv`-managed Python 3.12 loading `sqlite-vec` on macOS | **可用** | 实测 `uv` 托管 Python `3.12.15`（SQLite `3.53.1`）支持 `enable_load_extension(True)`，成功加载 `sqlite-vec v0.1.9` 并通过 `vec0(embedding float[4] distance_metric=cosine)` KNN 查询。`pyproject.toml` 固定 `[tool.uv] python-preference = "only-managed"`，同时在 T02 保留纯 `numpy` `VectorIndex` 回退后端以应对非托管环境。 |
| 3 | ExifTool platform path & Perl dependency | **启用回退**（本机未预装） | 本机 macOS `PATH` 未安装 `exiftool`，系统 `/usr/bin/perl` 可用。T04 按 `PATH` → `<APP_DATA_DIR>/bin/` 顺序探测，缺失时可经用户确认下载便携包，离线或不可用时平滑降级至 `rawpy.extract_thumb()` + `Pillow`/`exifread`（厂商私有字段标 `confidence="low"`）。 |
| 4 | OpenCV Zoo YuNet LICENSE & `pillow-heif` codec licenses | **可用**（按约束隔离） | 实测 `opencv/opencv_zoo` `face_detection_yunet/LICENSE` 为 **MIT License**，可安全用于 T05 人脸检测。`pillow-heif` (`1.9.0`) Python 封装层为 `BSD-3-Clause`，但底层依赖 `libheif`/`libde265`（LGPL-3.0）及可选 `x265`（GPL-2.0），严格隔离在可选 extra `heic` 中，不进入默认依赖，并在 `NOTICE` 中说明。 |
| 5 | Anthropic 5.5 (`output_config.format`/`effort`, `tool_choice`) & Gemini OpenAI-compat multi-image + JSON Schema | **启用回退预案**（当前环境无实时 API Key） | 本机未配置 `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`，实时真 Key 冒烟延后至 T10 执行。代码层面严格遵守 SPEC §6.1 的 5.5 API 约束，并为 Gemini 保留 ~150 行 `google-genai` 原生适配器回退预案（保持 `ChatModel` 接口不变）。 |
| 6 | SigLIP 2 Base ONNX & macOS CoreML EP readiness | **可用**（CPU 默认保底 + 懒加载） | 实测 `onnxruntime 1.32.0` 在 macOS Apple Silicon 上可用，提供 `CoreMLExecutionProvider` 与 `CPUExecutionProvider`。T11 采用视觉塔与文本塔分步懒加载，默认 `CPUExecutionProvider` 保底并在 T11 实测 CoreML EP 算子兼容性及与原版余弦精度偏差。 |
| 7 | macOS `renamex_np(RENAME_EXCL)` atomic no-overwrite rename | **可用** | 实测通过 `ctypes` 调用 `libc.renamex_np(..., RENAME_EXCL=0x00000004)`：新目标返回 `0` 成功移动；已存在目标返回 `-1` 且 `errno == 17 (EEXIST)`，目标与源文件均保持完整不被覆盖。T03 在 macOS 上优先使用 `renamex_np(RENAME_EXCL)`，非 macOS 平台回退至独占隔离目录内先检后移 + 唯一序号后缀。 |

## Deferred Client-Behavior Checks (Items 8–10)

Per `PLAN.md` T00 acceptance criteria, client-specific integration checks are deferred to their respective units:
- **Item 8** (`SameSite=Strict` Cookie behavior when opening `127.0.0.1` links from external apps): verify in **T07 / T08** (fallback: local one-click session confirmation on landing page, never passing tokens in URL).
- **Item 9** (Claude Desktop `PATH` lookup for `uvx` & MCP Elicitation / Async Tasks support): verify in **T09** (fallback: absolute path to `uvx`, short-wait + polling, tokenless Web UI approval).
- **Item 10** (Codex `skills/lightbox-cull/SKILL.md` auto-discovery path): verify in **T09** (fallback: document symlink/install path and provide root `AGENTS.md`).
