# LightBox 桌面快速启动与 AI Agent 连接设计方案

> 目标：**好用、方便、安全、不搞大厂过度设计**。在最小改动 `router`、`verifier`、`triage`、`jobs` 与 `app.py` 的前提下，保证未启用 `[ai]` 时现有行为 100% 不变，重点解决两个痛点：
> 1. 不用每次开终端敲命令，双击图标就能弹出独立桌面窗口、安全启停本地服务；
> 2. 点开右上角 AI Agent 后，一眼看清连接状态和凭据来源、填 1 个 Key 即可安全跑通，同时在合适的位置提供外部客户端（MCP）接入引导。

---

## 背景与边界共识

| 模块 | 本次要做的（好用、安全） | 刻意不做（保持简单） |
| :--- | :--- | :--- |
| **本地独立桌面** | `pywebview` 原生窗口 + `[project.gui-scripts]`（`lightbox-app`）与 `lightbox app` 统一入口 + `lightbox install-app` 一键生成图标（macOS `.app` / Windows `lightbox-app.exe` 快捷方式 / Linux `.desktop`）；明确服务归属与带超时/断点续做的安全退出 | 不引入 Electron / Tauri 重构前端；不做 `.dmg` 公证签名或 `.exe` 安装包打包；Windows **不使用已弃用的 `.vbs`** |
| **抽屉 AI 连接** | 抽屉顶部极简状态条（仅面向 **API Key** 与 **本地 Ollama**），只填 1 个 Key 自动配好 Agent + 低成本 Triage 双模型；能力按模型和角色实测；指纹缓存连接状态；Router 级区分「已知免费」与「价格未知」 | 不做高风险的网页订阅 Cookie 逆向；V1 不做本地 `claude -p` / `codex exec` 消费订阅桥接（列为后续合规调研）；不做多厂商按角色混搭的复杂表单 |
| **外部客户端 (MCP)** | **移出抽屉连接区**，放到设置/帮助入口（如「在 Claude / Cursor 中使用」）；V1 只做**状态检测 + 一键复制配置/命令**，绝不回传外部配置原始内容 | 不误导为“给抽屉供能”；V1 **不自动改写**外部客户端配置文件（尤其是持续改写的 `~/.claude.json`） |

---

## 一、本地独立桌面窗口 & 一键双击启动

### 1. 极简架构、统一入口与日志重定向

- **技术选型**：[`pywebview`](https://pywebview.flowrl.com/) 直接套在现有 FastAPI + `src/lightbox/web_dist/` React 前端外层（macOS 用系统 WKWebView、Windows 用 Edge WebView2、Linux 用 WebKitGTK），零前端重写。
- **模块位置与延迟导入**：桌面启动器放在 **`src/lightbox/cli/desktop.py`**（受 `import-linter` 契约保护）。**`webview` 模块仅在 `desktop.main()` 内延迟导入**，保证未安装 `pywebview` 的无头 CI 环境能正常导入与跑测试。
- **统一入口命名**：
  - 在 `pyproject.toml` 增加 **`[project.gui-scripts]`**：`lightbox-app = "lightbox.cli.desktop:main"`，三平台统一生成无控制台桌面启动器。
  - 在 `src/lightbox/cli/app.py` 明确定义 **`lightbox app`** 子命令，与 `lightbox-app` 调用**同一个 `desktop.main()` 函数**。
  - 补一个极简 **`src/lightbox/cli/__main__.py`**，让 `python -m lightbox.cli app` 与 Windows 兜底的 `pythonw.exe -m lightbox.cli app` 都能跑通。
- **跨平台无控制台日志重定向**：Windows 使用 `gui-scripts` 或 `pythonw.exe` 启动时 `sys.stdout` / `sys.stderr` 为 `None`，`typer.echo` 或 `uvicorn` 打日志会直接报错。因此在 **`desktop.main()` 的第一步**由 Python 统一将 `sys.stdout` 与 `sys.stderr` 重定向到 `<APP_DATA_DIR>/logs/desktop.log`（三平台一致）。遵守零删除规则，不做删文件轮转——每次启动前将旧 `desktop.log` 覆盖写入 `desktop.prev.log`（保留上一份副本），再覆写本次 `desktop.log`。
- **可选依赖与优雅降级**：
  ```toml
  [project.optional-dependencies]
  desktop = ["pywebview>=5.3"]
  ```
  未安装 `pywebview` 时，命令行调用提示安装命令并退回浏览器 `lightbox ui`。
- **Linux 系统后端依赖**：写明需安装系统 GTK 绑定（如 `python3-gi gir1.2-webkit2-4.1`）或 Qt（`PySide6`），否则 `pywebview` 无可用渲染后端。

### 2. 服务归属、限定自动登录与断线重连

`src/lightbox/mcp/ui_bridge.py` 在 Claude Desktop 等客户端调用 MCP 时也会启动本地 UI 服务并写 `server.lock`（且**不传 `bootstrap_token`**，并在客户端退出时一起停掉）。因此必须明确：

1. **「谁启动、谁关闭」**：桌面端只停止**自己启动的** Uvicorn 实例，只清理自己写的 `server.lock`（校验 `pid` / `instance_id`）。
2. **仅限桌面模式的免密自动登录（解决注入时序竞争）**：复用已有健康服务时（没有 `bootstrap_token`），前端先用统一的就绪检测等待桌面桥——若 `window.pywebview` 尚未注入，先监听 **`pywebviewready` 事件并设短超时（如 300–500ms）**；确认处于桌面模式后才自动调用 **`POST /api/auth/session`**（携带同源 `Origin` 与 `X-Lightbox-Local-Confirm: 1`）完成免点击登录。**超时未等到或普通浏览器访问时，严格按浏览器模式保持现有 `needs_confirm` 手动确认页不变。**
3. **自启服务**：若无可用实例，桌面端在后台线程启动自己的 Uvicorn（绑定 `127.0.0.1` 随机空闲端口），直接加载一次性 `/auth/launch?token=...` 完成登录（仅内部使用，绝不打印或暴露给 MCP）。
4. **防误报与明确重连路径**：休眠唤醒可能导致 SSE 短暂断开，前端仅在**连续 3 次 `/healthz` 探测失败**后才显示「服务已停止」+「重新连接」按钮。点击后前端通过 `pywebview` 注册的 **`js_api` 通知桌面进程**，由桌面进程启动新端口的 Uvicorn，并调用 **`window.load_url(new_launch_url)`** 重载页面。

### 3. 安全退出（带超时）、断点续做与进程归属自愈

窗口右上角 `×` 与 macOS `Cmd+Q` 走**同一套退出处理逻辑**，并兼顾无法拦截的系统强制退出：

- **文件移动在途保护**：`approve-and-apply` 与 `restore` 是同步文件移动请求（job 表不可见）。后端增加轻量“在途文件操作计数”；计数 > 0 时阻止窗口立即退出（提示「正在移动文件，请稍候完成后关闭」）。
- **保持 `daemon=True` + 协同停止 + 超时即退出**：`Scan` 与 `Triage` 的后台线程**必须保持 `daemon=True`**，额外传入协同停止标志（`threading.Event`）。退出时发停止信号并等待 worker 停到可恢复检查点，**等待必须设超时上限（如 5–10 秒）**；超时后直接关闭 `event_bus`、停 Uvicorn 并退出（若改成非 daemon 线程，卡在网络请求上时 Python 会永远无法退出）。
- **真正的文件安全兜底（断点续做）**：Dock 退出、注销/关机或强杀进程无法完全阻止，最终依靠 `plan` 的 `applying` 状态与现有 **`apply_plan(resume=True)`** 断点续做保证零丢失。
- **基于进程归属的自愈（修复 Windows `os.kill(pid, 0)` 误发 Ctrl+C 问题）**：
  - 现有 `src/lightbox/api/server_lock.py` 的 `is_pid_alive` 使用 `os.kill(pid, 0)`，在 Windows 上 `0` 等价于 `signal.CTRL_C_EVENT`，会直接发送 Ctrl+C 信号而非探测存活（可能误杀 MCP 等其他进程）。必须替换为**跨平台进程存活探测**（如引入 `psutil.pid_exists` + `Process(pid).create_time()`，或 Windows 用 `ctypes` `OpenProcess` + `GetExitCodeProcess`），放在底层公共模块供 `server_lock.py` 与 `jobs/runner.py` 共用（符合 `import-linter`）。若选用 `psutil`，必须放进 **`pyproject.toml` `[project.dependencies]` 主依赖**，不能放在 `desktop` 可选组（CLI/MCP 无桌面环境也依赖它）。
  - 单次带重试的 VLM 调用耗时可能超过心跳间隔，且桌面端、MCP 与 CLI `lightbox scan` 共用同一数据库。因此在 `jobs` 表增加 **`owner_pid`、`owner_pid_started_at`（进程启动时间，直接防 PID 复用，比心跳更可靠）与 `owner_instance_id`**（与 `llm_calls` 新增字段合并在同一次数据库迁移中完成）；启动自愈时**仅当所属进程已不存在或启动时间不一致**才将残留 `running` job 标为 `paused`（可恢复），避免仍在跑的长任务或他进程任务被误标暂停进而引发双 worker 并发。

### 4. 桌面原生目录选择

桌面模式下，先复用上述 `pywebviewready` 就绪等待确认桥可用，再将前端 `pickFolder` 入口桥接到 **`window.create_file_dialog(webview.FOLDER_DIALOG)`**，避免 `osascript` / `tkinter` 子进程弹窗被原生窗口遮挡；浏览器模式及就绪超时时保持 `/api/system/pick-folder`（自动登录、原生选目录、断线重连 `js_api` 三处共用同一套就绪等待）。

### 5. 跨平台图标安装 `lightbox install-app` 与环境变量限制

通过 `sys.executable` 解析当前环境同目录下的 `lightbox-app` 绝对路径，避开 GUI 启动不继承终端 `PATH` 的坑：

| 操作系统 | 生成位置与产物 | 实现方式 |
| :--- | :--- | :--- |
| **macOS**（优先） | `~/Applications/LightBox.app`（支持 `--desktop` 放桌面替身） | `Contents/Info.plist` + `Contents/MacOS/LightBox`（直接 `exec` `lightbox-app` 绝对路径，异常时 `osascript` 弹窗；Dock 图标是否需 PyObjC 调整按实机验收决定） |
| **Windows**（优先） | 桌面 / 开始菜单 `LightBox.lnk` | 直接指向无控制台 **`lightbox-app.exe`**（或 `pythonw.exe -m lightbox.cli app`），**不生成 `.vbs`** |
| **Linux**（顺手） | `~/.local/share/applications/lightbox.desktop` | `Terminal=false`，`Exec=<abs_lightbox_app>` |

> **环境变量限制**：双击桌面图标启动**不会继承** `~/.zshrc` / `~/.bashrc` 中 `export` 的环境变量（如 `ANTHROPIC_API_KEY`）。界面显式展示凭据来源，方便排查。

---

## 二、Agent 面板：AI 连接配置（仅 API Key / 本地 Ollama）

严格遵守 `FrontEndDesignRules.md`（无嵌套框、无多余分割线、无解释废话、90% 中性灰白 + 10% 单强调色、Geist + 4px 栅格 + Reicon），全量文案统一走 `web/src/i18n.ts`（禁止中英混用）。

### 1. 精细状态机、验证指纹与凭据来源

1. **验证指纹持久化 + 后端从 SQLite 调用记录统一推导状态**：
   - 本地保存上次验证结果与配置指纹（`provider` + 规范化 `base_url` + `agent_model` + `triage_model` + Key 脱敏后缀）。
   - 为 `llm_calls` 表与 `LLMCallRecord`（`src/lightbox/store/db.py`、`src/lightbox/store/repos.py`）新增 3 个字段（与 `jobs` 表字段在同一次迁移中升级）：**`status`**（`ok` / `error`）、**`error_class`**（映射后的错误类别）与 **`connection_fingerprint`**（当前连接指纹/端点）。`ModelRouter` 在真实调用失败时也写入一条记录（**`cache_key=None`**、`status="error"`、`cost_usd=0.0`、`cache_hit=False`、`response_json=None`，既不影响预算统计，也不占用 `cache_key` 唯一约束而破坏后续重试成功时的缓存）。
   - 每次打开界面只比对本地指纹（绝不在启动时发网络请求），并**由后端结合指纹与 SQLite 中满足 `connection_fingerprint == 当前指纹` 且 `cache_hit = False` 的真实调用记录统一推导状态**（排除缓存命中误判连通，也隔离旧配置记录干扰；CLI、MCP 进程和后台 Triage 故障均可同步反映到前端）。
   - **区分持久失效与临时故障**：仅**鉴权失败（401/403）、端点或模型不可用、账户余额不足**这类持久性错误才会使「已连接」失效并切为「异常」（充值或修复后重新验证恢复）；**429 限流、网络抖动、单次超时、LightBox 本地预算已用尽**（按月自动重置）只显示临时提示，不把「已连接」打回「异常」。指纹未变且无持久性失败时保持显示「已连接 · 模型名（上次验证时间）」，指纹一变重置为「已配置 · 未验证」。
2. **5 种状态与按厂商错误映射**：
   - `unconfigured`（未连接） / `configured_unverified`（已配置 · 未验证） / `testing`（验证中） / `connected`（已连接 · 模型名 + 上次验证时间） / `error`（持久性异常）。
   - **按厂商映射错误**：为 OpenAI / Anthropic / Gemini / OpenAI 兼容各维护一张轻量错误映射表，区分**持久错误**（鉴权失败 / 模型或端点不可用 / 账户余额不足）与**临时提示**（触发限流 / 网络超时 / LightBox 本地预算已用尽）；未命中映射表的统一显示「请求失败」并附原始错误码。
3. **凭据来源独立展示与操作限制**：
   - 独立显示当前生效凭据来源（`系统钥匙串` / `环境变量` / `本地 .env`）。
   - 若来源为 **`env`（环境变量）**：允许「停用连接」（清除 `[ai]` 激活态），但**禁用「删除凭据」按钮**并提示需在系统环境变量中移除。

### 2. 双模型、可验真探针与 Ollama 说明

点击「测试并连接」时（仅主动点击时触发，走真实 adapter，严守 SPEC 保持 `tool_choice=auto` 不强制调用工具），分别验证两个实际用到的模型：

1. **Agent 模型**（同时负责 `agent/memory.py:304` 的 `consolidate`）：
   - **可验真视觉探针**：动态生成内容已知的小图（例如左红右蓝色块或清晰大数字），校验模型回答是否答对，防止丢图模型蒙混过关；
   - **工具探针**：带一个无副作用的只读工具，验证返回合法工具调用；
   - **结构化输出探针**：验证轻量 JSON Schema 响应（供 `consolidate` 使用）。
2. **Triage 模型**（若与 Agent 模型相同则共享结果，不重复测）：
   - **可验真视觉探针** + **结构化输出探针**。
- **按角色可用性展示与降级**：结果按角色记录（`agent_ready` / `triage_ready`）；例如某模型视觉通但工具调用失败，只禁用抽屉 Agent，批量 Triage 照常可用。
- **Ollama 基址与上下文提示**：用户只填基址（默认 `http://127.0.0.1:11434`），后端自动拼接 `/api/tags`、`/api/show` 与 `/v1`。明确提示：`/api/show` 返回的是模型理论上限而非运行时实际 `num_ctx`，若提示或实测发生截断需调大 Ollama 运行时上下文。

### 3. 「单 Key 双模型」、设置页接管与全局统一路由

1. **单 Key 双预设**：用户只需填 **1 个 Key**，每个主流厂商预设一对默认模型——主模型给 `agent` / `consolidate`，低成本视觉模型给 `triage_vlm`（沿用 `balanced` 思路，也允许用户微调），避免批量 Triage 用贵模型失控。
2. **全局统一有效目录**：在服务层构造统一的「有效模型目录 + 有效 profile」，供估价、Agent Loop、Triage、IndependentVerifier 共用（修复 `verifier.py:183` 绕过 `resolve_spec` 与 `triage.py:226/284/586` 提前展开 `config.model_profile` 的问题）：
   - 解析优先级：`显式传入 profile` → `全局 [ai]` → `config.model_profile`（Triage 未显式指定时传 `None`）。
   - `[ai]` 是全局配置，抽屉、CLI `lightbox chat`、MCP 触发的 Triage 均按此规则生效。
3. **设置页互斥提示**：当 `[ai]` 激活时，设置页明确标注 `model_profile`「由 AI 连接接管」并禁用该下拉框，避免用户切换 `balanced`/`best` 无效产生疑惑。
4. **Layer 2 复核保持可控**：通过 UI 新建的 `[ai]` 连接默认 `reviewer = "none"`（仅跑 Layer 1 确定性检查，界面显示「`复核：仅规则`」）；原有 `balanced`/`best` profile 行为不变。

### 4. Router 级成本分类与「价格未知」防超支护栏

把成本分类下沉到 **`ModelRouter`**（根治 `router.py:391` 把价格为 0 当作免费的漏洞）：
- **`known_free`**：仅限内置本地模型、探测确认为 Ollama（如 `/api/tags` 有响应）、或**用户明确勾选标记为「本地免费模型」**的服务，真正免费，不设费用限制；
- **`known_paid`**：内置目录定价模型，按美元单价正常走 `session_budget_usd` / `monthly_budget_usd` 硬拦截；
- **`price_unknown`**（远程自定义中转、目录外模型，以及**默认情况下的 `localhost` / `127.0.0.1` 自定义端点**——防止本机运行的 LiteLLM / one-api / new-api 付费转发代理绕过预算护栏）：
  - `ActiveAIConfig` 扩展可选字段 **`input_cost_per_mtok`** 与 **`output_cost_per_mtok`**；`ModelSpec.provider` 的 `Literal` 扩展支持 **`"custom"`**；
  - **抽屉单次对话**：标注「费用无法预估」，并在 `ModelRouter` / Agent 层施加**单会话最大调用次数上限**；
  - **批量 Triage（二选一）**：用户可手填估算输入/输出单价（填完后按常规美元预算护栏运行），**或**不填单价按**「单次最多处理 N 张（如默认 20 张）」+ 每次运行前弹窗明确确认**执行。

### 5. 密钥端点绑定、删除后重解析与原子写入

1. **端点随密钥一起存放 + 重新绑定清理旧条目**：规范化端点（如 `https://api.openai.com/v1`）必须**随密钥一起存放**（keyring 条目名中编码端点，或 `.env` 中保存配套的绑定行），绝不只存在 `config.toml` 里。**没有绑定记录的环境变量或历史 Key，仅允许发往该厂商的官方默认端点**；测试、保存、实际推理三条路径统一校验，改 `base_url` 必须重新输入 Key。**切换端点重新绑定时**，必须调用 `keyring.delete_password(...)` 清理旧端点对应的钥匙串条目（不存在则忽略），并同步通过 `config_io` 原子改写 `.env` 清理或替换旧端点的绑定行，避免旧端点密钥长期滞留钥匙串。
2. **Keyring 读回校验 + 旧值清理 + 删除后重解析**：
   - 写入 keyring 后立刻读回校验，成功再从 `.env` 剔除同名旧 Key；失败则平滑回落到 `0o600 .env`。
   - 执行 `DELETE /api/ai/connection` 后，后端**重新解析一次当前有效凭据来源并返回给前端**（例如删掉 keyring 后如果环境变量里还有 Key，前端能如实看到新接管的 `env` 来源，不会悄悄失效或产生误判）。
3. **配置文件全过程原子写入（窄白名单 + 写入顺序）**：
   - 新建专用配置写入模块（如 `src/lightbox/config_io.py`），在 `scripts/check_no_delete.py` 和 `pyproject.toml`（`TID251`）中增加**仅限该模块的 `os.replace` 窄白名单**；
   - 该模块运行时强制校验目标路径位于 `<APP_DATA_DIR>` 内，文件锁必须覆盖完整的**「读取 → 修改 → 写入同目录临时文件 → 保留原权限（`.env` 保持 `0o600`）→ `os.replace` 原子替换」全过程**（不能只锁最后替换一步，防止 CLI 与桌面端并发读写互相覆盖），失败时原文件不动；
   - **Windows 替换重试**：在 Windows 上遇到目标文件被短暂占用/杀软扫描抛出 `PermissionError` 时，执行短暂有限重试；
   - **`PUT /api/ai/connection` 写入顺序**：**先写 Key 与端点绑定，再原子写 `config.toml`**；若写配置失败则把错误返回前端（残留 Key 因为有端点绑定校验且 `[ai]` 未写入，不会被误用）。
4. **设置隔离与 TOML 序列化完整**：`PUT /api/settings` 显式拒绝 `ai` 字段；扩展 `_dump_config_toml` 完整序列化 `[ai]` 段（含可选自定义单价，严禁明文 Key）。

---

## 三、外部客户端接入（MCP）— 独立入口与安全边界

- **位置与定位**：移出 `AgentDrawer` 的 AI 连接面板，放在设置/帮助区域（如「在 Claude / Cursor 中使用」），明确提示 MCP 用于让外部客户端操控 LightBox，不会点亮抽屉自身连接状态。
- **V1 能力（只读检测 + 一键复制）**：
  - 支持检测 **Claude Desktop**、**Cursor**、**Codex**、**Claude Code**（只读解析；解析失败时保留原文件绝不覆盖）。
  - 精确状态判定：检查配置里的命令路径与开发态 `--directory` 仓库路径是否真实存在；未实际验证客户端拉起前仅显示「已配置」。
  - 按客户端原生格式生成一键复制片段（Claude Code 给 `claude mcp add --scope user ...` 命令，不碰运行期频繁改写的 `~/.claude.json`；Codex 给 TOML `[mcp_servers.lightbox]`；Claude Desktop / Cursor 给 JSON `mcpServers`，提示重启生效）。
- **严禁泄露外部配置原文 & 防注入**：
  - 检测接口**只返回推导后的状态布尔值/路径有效性**，**绝不回传外部配置文件原始内容**（其中常含其他 MCP Server 的 API Key 或 `~/.claude.json` 敏感信息），也**绝不写进日志**。
  - 请求体使用 `extra="forbid"` **仅接受客户端枚举**，所有命令与参数由后端生成。
- **后续调研（V1 不做）**：用消费级订阅额度驱动抽屉（如 `claude -p` / `codex exec`）列为后续调研，先确认各厂商服务条款边界。

---

## 四、REST API 清单与 5 步实施路线

### 核心新增 API（受 `LocalSecurityMiddleware` Cookie + CSRF 保护）

- `GET /api/ai/connection` — 返回连接状态（含指纹命中缓存的 `connected` + 上次验证时间）、凭据来源、脱敏后缀、各角色能力状态、复核模式与当前成本类型。
- `POST /api/ai/connection/test` — 校验端点一致性后对 Agent / Triage 模型执行可验真探针，更新验证指纹与结果，返回细分错误类型。
- `PUT /api/ai/connection` — 保存 `[ai]`（含可选估算单价）并原子写入绑定端点的 Key。
- `DELETE /api/ai/connection` — 停用 `[ai]`、清理非 `env` 凭据，并**重解析当前剩余凭据来源返回前端**。
- `GET /api/ai/ollama/models` — 从服务基址拼接 `/api/tags` 与 `/api/show` 返回模型列表与上限提示。
- `GET /api/ai/mcp/clients` & `POST /api/ai/mcp/snippet` — 只读检测外部客户端（零原始内容返回）与生成复制片段（严格枚举入参）。

### 5 步实施路线与验收标准

| 步骤 | 改动范围 | 关键验收项 |
| :--- | :--- | :--- |
| **Step 1**<br>配置原子写、密钥绑定、迁移与 Router 护栏 | `pyproject.toml`（`TID251` 窄白名单）<br>`scripts/check_no_delete.py`<br>`src/lightbox/config_io.py`<br>`src/lightbox/config.py`<br>`src/lightbox/store/db.py`<br>`src/lightbox/store/repos.py`<br>`src/lightbox/models/secrets.py`<br>`src/lightbox/models/router.py`<br>`src/lightbox/agent/verifier.py`<br>`src/lightbox/services/triage.py`<br>`src/lightbox/services/system.py` | 1. **回归基线**：无 `[ai]` 时 `balanced`/`best`/`local` 行为 100% 不变<br>2. **统一 DB 迁移单测**：`llm_calls` 新增 `status`/`error_class`/`connection_fingerprint`，`jobs` 新增 `owner_pid`/`owner_pid_started_at`/`owner_instance_id`；**旧版本数据库升级后能正常打开，且已有 `job` 能正常续做**<br>3. `ModelRouter` 失败时也写 **`cache_key=None`**、`status="error"`、`cost_usd=0.0`、`cache_hit=False` 记录（单测验证同一请求先失败后重试成功仍能正常写入并命中缓存）；`config_io` **全过程读写锁**原子写入（`.env` 保持 `0o600`、Windows `PermissionError` 短暂重试、先写 Key 再写配置、失败不损原文件）与 `_dump_config_toml` 往返单测<br>4. `ModelSpec.provider` 支持 `"custom"`、`ActiveAIConfig` 含可选单价、端点随 Key 存储与校验；切换端点重绑时清理旧 keyring 与 `.env` 绑定行<br>5. Router 级 `known_free`（仅内置/确认的 Ollama/显式勾选本地免费）vs `price_unknown`（`localhost` 默认归此类，单会话调用上限 + Triage 填价或限量确认） |
| **Step 2**<br>REST API、探针与安全校验 | `src/lightbox/api/app.py`<br>`src/lightbox/services/system.py`<br>`tests/api/` | 1. 已知答案合成图视觉探针 + 工具 + JSON 探针分模型测试、厂商错误映射、验证指纹持久化；**连接状态严格按 `connection_fingerprint == 当前指纹` 且 `cache_hit=False` 从 SQLite 记录推导：仅鉴权/端点不可用/账户余额不足切 `error`，429 限流/网络抖动/本地预算用尽仅临时提示**<br>2. `DELETE` 后重解析凭据来源返回；改 `base_url` 必须重输 Key<br>3. `PUT /api/settings` 拒收 `ai`；`/api/ai/mcp/*` 拒收额外字段且**不回传/不记录外部配置原文** |
| **Step 3**<br>AgentDrawer、Traces 兼容 & 独立 MCP 面板 | `web/src/api.ts`<br>`web/src/i18n.ts`<br>`web/src/components/AgentDrawer.tsx`<br>`web/src/views/InsightsView.tsx`<br>设置/帮助视图（含 `model_profile` 接管提示） | 1. 全量中英文走 `i18n.ts`；启动只比对本地指纹不发请求<br>2. 凭据来源展示（`env` 禁删）；**前端展示后端推导状态（鉴权/端点/余额不足切 `error`，限流/抖动/本地预算仅临时提示，CLI/MCP/后台失败可同步可见）**<br>3. `InsightsView` 的调用记录（Traces）页面能正常展示新增失败条目；无单价 Triage 运行前弹窗确认；`npm run typecheck && npm run build` + 浏览器视觉验收 |
| **Step 4**<br>桌面窗口、归属与安全生命周期 | `pyproject.toml`（`psutil` 主依赖 + `gui-scripts` + `desktop`）<br>`src/lightbox/cli/desktop.py`<br>`src/lightbox/cli/__main__.py`<br>`src/lightbox/cli/app.py`<br>`src/lightbox/api/server_lock.py`<br>`src/lightbox/store/db.py`<br>`src/lightbox/store/repos.py`<br>`src/lightbox/jobs/runner.py`<br>`src/lightbox/api/app.py`<br>`web/src/App.tsx`<br>`web/src/api.ts` | 1. `lightbox app` / `lightbox-app` / `-m lightbox.cli app` 同入口；`webview` 延迟导入（无 `pywebview` CI 通过）；`desktop.main()` 首步统一日志重定向<br>2. **单测**：`triage` 保持 `daemon=True` + 停止标志且退出超时不被卡死；假 webview 窗口测在途操作阻止关闭 & 退出等待超时；Apply 中途强杀重启后 `apply_plan(resume=True)` 续做；**Windows 进程存活探测不调用 `os.kill(pid, 0)`，及基于 `owner_pid` + 进程启动时间 `owner_pid_started_at` 防 PID 复用的自愈单测（不误伤长耗时存活任务）**<br>3. 等待 **`pywebviewready`**（带短超时）后再判定桌面模式并自动调用 `/api/auth/session`（超时及浏览器保留 `needs_confirm`）；连续 3 次 `/healthz` 失败才显示重连并调用 `js_api` + `window.load_url`；原生选目录接通 |
| **Step 5**<br>一键图标安装与实机验收 | `src/lightbox/cli/app.py`<br>`src/lightbox/services/system.py` | 1. macOS `.app` / Windows `.lnk`（指向 `lightbox-app.exe`，无 `.vbs`）/ Linux `.desktop` 生成单测<br>2. **macOS 实机验收**：Finder 双击可开、Dock 仅 1 个图标（必要时 PyObjC 修正）、`<APP_DATA_DIR>/logs/desktop.log` 有日志、`~/Desktop` / `~/Pictures` / 外接盘权限正常 |

---

## 五、必须守住的 4 条项目红线

1. **零硬删除与受控原子写**：业务代码禁止 `os.remove`、`os.unlink`、`shutil.rmtree`、`.unlink()` 等删除 API（`ruff TID251` + `scripts/check_no_delete.py`）；`os.replace` 仅允许用于 `SafeFS`、`cache_lru` 之外新增的 `<APP_DATA_DIR>` 专用配置模块 `config_io.py`，绝不触碰用户照片目录。
2. **分层架构不越界**：桌面模块放在 `src/lightbox/cli/desktop.py`，受 `lint-imports` 约束，不得绕过 `services` 直接 import `store`/`fs`/`core`。
3. **密钥与外部配置零泄露**：Key 只进 keyring 或 `0o600 .env`，端点绑定随密钥存放，不进 `config.toml`、不进日志、不回传明文；`PUT /api/settings` 拒收 `ai`；MCP 接口绝不回传外部配置原文、绝不接收前端命令/路径。
4. **零 LLM 离线可用**：无任何 API Key、未安装 `pywebview` 时，确定性扫描、本地评审、Web UI 及现有全量 CI 测试全部照常通过。
