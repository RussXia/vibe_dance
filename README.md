# Vibe Dance Editor

从多人视频中框选目标人物，自动跟踪并输出以该人物为中心的 9:16 裁剪视频。

## 功能

- 本地视频导入（mp4 / mov），浏览器内核预览
- 9:16 取景框：拖动移动，边缘/角拖动缩放（图片编辑器式光标反馈），所见即所得
- 输出分辨率可选：720×1280 / 1080×1920 / 自定义（宽高需为偶数）
- CSRT/KCF 逐帧跟踪 + YOLO 人体检测定期重定位，遮挡自动续跟
- 导出保留原视频音轨；进度条分「跟踪中 / 渲染中」两阶段实时显示
- 导出前选择保存位置，完成后一键「打开所在文件夹」
- 替换音轨：素材B（纯净音乐）替换素材A（现场视频）音轨，自动对齐（DTW/节拍/从头三级降级），波形图拖动微调 + 实时试听，确认后导出

## 结构

- `app/` — Electron + React 桌面客户端
- `engine/` — Python 处理引擎（OpenCV + YOLO 跟踪与裁剪）

## 环境要求

- Node.js 20+
- Python 3.13
- FFmpeg 8.x（系统级）

详见 `docs/superpowers/specs/2026-08-07-dance-video-editor-design.md`。

## 安装发行版

从 [GitHub Releases](https://github.com/RussXia/vibe_dance/releases) 下载安装包：

- macOS：`VibeDance-*.dmg`（Apple Silicon）
- Windows：`VibeDance.Setup.*.exe`

### macOS：首次打开提示「已损坏，无法打开」

安装包是**未签名**的（CI 未配置 Apple 开发者签名/公证），macOS 的 Gatekeeper 会对从网上下载的未签名 App 弹「已损坏」。**文件本身没坏**，是签名校验被拦。

**解决办法（任选其一）：**

1. **右键打开**：在 Finder 中右键点击 `VibeDance.app` → 选「打开」→ 在弹窗中再点「打开」。（双击可能不弹，需用右键）
2. **清除隔离属性**（推荐，最稳定）：

```bash
xattr -cr /Applications/VibeDance.app
```

3. **终端手动启动**（临时）：
```bash
/Applications/VibeDance.app/Contents/MacOS/VibeDance
```

> 系统偏好设置 → 隐私与安全性 → 「仍要打开」也可以。要彻底去掉这个提示（用户开箱即用），需要 Apple 开发者账号对 App 做签名 + 公证，见「发布」章节。

### Windows：SmartScreen 提示「未知发布者」

同理，安装包未签名，Windows SmartScreen 可能提示「Windows 已保护你的电脑」。点「更多信息」→「仍要运行」即可。

## 运行桌面客户端

```bash
cd app
npm install
npm run electron:dev   # 构建并启动 Electron，自动拉起引擎
```

> 若之前有残留的引擎进程占用 8787 端口，先 `lsof -i :8787` 找到并 kill，否则新客户端会连到旧引擎。

## 开发

启动前端 dev server：

```bash
cd app
npm install
npm run dev
```

启动引擎（单独终端，或由 Electron 客户端自动拉起）：

```bash
cd engine
source .venv/bin/activate
pip install -e .   # 首次，安装 opencv-contrib / ultralytics / torch
python -m engine 8787
```

## 打包

### 引擎打包策略

Python 引擎用 PyInstaller 冻结为独立二进制（`engine/dist/engine_bundle/`），随安装包分发，Electron spawn 该二进制（见 `app/electron/main.ts` 的 `startEngine`）。**不依赖目标机 Python**，可移植。

- 入口 `engine/launcher.py` 用绝对导入，避免 PyInstaller 冻结相对导入失败。
- YOLO 权重 `engine/yolov8n.pt` 随引擎打包（spec `datas`），无需联网下载。
- 冻结产物约 45MB（含 torch），已在本机验证可独立启动、完整渲染。

> **注意**：渲染编码仍依赖系统级 `ffmpeg`（见下方「注意事项」），这是唯一的外部运行时依赖。

### 发布（GitHub Actions，推荐）

打 `v*` tag 自动触发 CI 构建双平台安装包并发布为 GitHub Release：

```bash
git tag v0.1.0
git push origin v0.1.0
```

CI 流程（`.github/workflows/release.yml`）：创建引擎 venv → PyInstaller 冻结 → `electron-builder` 打包 → 收集产物发布。macOS 产 `.dmg`，Windows 产 `.exe`。

### 本机打包（调试用）

```bash
# 1. 冻结引擎二进制
cd engine
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pyinstaller
pyinstaller engine.spec --distpath dist --workpath build --noconfirm
# 产物：engine/dist/engine_bundle/vibe_engine（macOS）/ vibe_engine.exe（Windows）

# 2. 打安装包
cd app
npm install
npm run package       # electron-builder --mac（生成 dmg）
npm run package:dir   # 仅生成 .app（不做 dmg，验证更快）
```

产物在 `app/release/` 下：

- macOS：`app/release/VibeDance-0.1.0-arm64.dmg`（`package`）或 `app/release/mac-arm64/VibeDance.app`（`package:dir`）
- Windows：在 Windows 机器或 CI 上执行 `npx electron-builder --win`（`app/release/VibeDance Setup 0.1.0.exe`）

打包配置见 `app/electron-builder.yml`：引擎二进制通过 `extraResources` 随包分发，路径为 macOS `Contents/Resources/engine_bundle` / Windows `resources/engine_bundle`。

### 注意事项

- **FFmpeg**：渲染编码依赖系统级 `ffmpeg`，目标机器需安装 FFmpeg 8.x（仅开发机与 CI 需要；引擎与 App 本身不内置）。
- **音频对齐（替换音轨）**：引擎内置 librosa（随打包分发），无需额外安装。
