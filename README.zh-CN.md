# LightBox

[![CI](https://img.shields.io/github/actions/workflow/status/olioDuan/LightBox/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/olioDuan/LightBox/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-18181b?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Zero Hard Delete](https://img.shields.io/badge/Safety-Zero%20Hard%20Delete-059669?style=flat-square)](scripts/check_no_delete.py)
[![MCP + Skill](https://img.shields.io/badge/MCP-Server%20%2B%20Skill-2563eb?style=flat-square)](#方式二接入-claude--cursor--codexskill--mcp)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-52525b?style=flat-square)](LICENSE)

[English](README.md) | 简体中文

本地运行的选片工具。默认完全离线，不硬删除原片，既可以当独立桌面/网页软件用，也可以通过 Skill 和 MCP 接入 Claude、Cursor、Codex 使用。

## 功能概览

- **本地筛片（离线，无需 API Key）**：检测拍糊、过曝/欠曝、闭眼，自动对连拍分组并推荐最佳张，RAW+JPEG 与 Live Photo 同组联动。
- **可逆隔离**：绝不直接删除照片目录里的文件；整理废片必须经人工确认后移入隔离文件夹，随时可整批或单张还原。
- **本地搜图与人物整理**：内置 SigLIP 2 语义搜图与以图搜相似、人脸聚类命名及按人物导出。
- **可选 AI 复核**：在界面按 `Cmd+K` / `Ctrl+K` 打开助手，可接入 OpenAI、Anthropic、Gemini、OpenAI 兼容接口或本地 Ollama，用来复核存疑照片或对话选片。

## 方式一：作为独立桌面 / 网页软件使用

需要 Python 3.12+ 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/olioDuan/LightBox.git
cd LightBox
uv sync --extra desktop
```

启动桌面窗口：

```bash
uv run lightbox app
```

生成系统双击启动图标（macOS `~/Applications/LightBox.app`、Windows `.lnk`、Linux `.desktop`）：

```bash
uv run lightbox install-app
```

如果不想装 `pywebview`，直接用浏览器打开：

```bash
uv sync
uv run lightbox ui
```

想先体验一下而不扫描自己的照片库：

```bash
uv run lightbox demo
```

可选扩展：HEIC 支持用 `uv sync --extra heic`；Linux 桌面窗口需要系统安装 GTK 绑定（`python3-gi gir1.2-webkit2-4.1`）或 `PySide6`。

## 方式二：接入 Claude / Cursor / Codex（Skill + MCP）

### 聊天直接传图选片（零安装）

在支持 Skill 的客户端（Codex、Claude Code、Cursor 等）中启用 `skills/lightbox-cull/SKILL.md`，直接把几张照片发进对话框即可让模型挑废片或选最佳张，不需要本地安装软件。

### 扫描本地整夹（MCP）

把 MCP 配置指向本地仓库路径（也可以在软件内 **设置 -> 在 Claude / Cursor 中使用** 一键复制对应客户端的配置片段）：

```json
{
  "mcpServers": {
    "lightbox": {
      "command": "uv",
      "args": ["--directory", "/path/to/LightBox", "run", "lightbox", "mcp"]
    }
  }
}
```

如果 GUI 客户端找不到 `uv`，将 `"uv"` 替换为终端 `which uv` 输出的绝对路径即可。

## 本地开发与检查

```bash
uv run ruff check .
uv run python scripts/check_no_delete.py
uv run lint-imports
uv run mypy src
uv run pytest
```

前端源码在 `web/` 目录，修改后执行 `cd web && npm install && npm run build` 更新 `src/lightbox/web_dist/`。

## 许可协议

Apache-2.0。第三方依赖与按需下载模型的许可说明见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
