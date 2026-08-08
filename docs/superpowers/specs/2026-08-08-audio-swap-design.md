# 替换音轨（音频对齐 + 可交互预览）设计文档

> 日期：2026-08-08
> 状态：已评审
> 目标：在 Vibe Dance Editor 中新增一个**完全独立**的功能——把素材A（原始视频，含嘈杂现场声）的音轨替换为素材B（纯净背景音乐）的音轨，自动对齐，并提供可交互的波形图预览、拖动微调、实时试听，确认后导出下载。

## 1. 背景与目标

跳舞视频的现场录制往往嘈杂（人声、拍手、环境音），用户希望得到**更纯净、更卡点的 BGM** 成品视频。用户已有素材B（就是素材A里那首音乐的纯净版），因此核心诉求是：

- 用素材B 的音轨替换素材A 的音轨；
- 素材B 能与素材A 的画面/音乐**自动对齐**（找到素材B 起点在素材A 中的时间偏移，必要时变速）；
- 对齐结果可**交互微调**（波形图 + 拖动 + 试听），确认后导出下载。

**产品定位**：完全独立的第二个功能入口，与现有"框选裁剪"并列。不经过取景框/跟踪裁剪管线。

**明确不做**（YAGNI）：
- 不做歌曲识别/指纹匹配（素材B 与素材A 背景音乐被假设为**同源**）；
- 不做多段音轨拼接/淡入淡出处理（单音轨替换）；
- 不做手机相册/平台导入（只支持本地文件）。

## 2. 素材关系假设

经与用户确认：
- 素材B 与素材A 的背景音乐是**同一首曲子**；
- 播放速度**基本原速，偶尔微调**（可能 1.2x-1.5x，需 DTW 容忍变速）；
- 素材B 可以是纯音频（mp3/wav/flac）或带音轨的视频（取其音轨）。

## 3. 产品交互与行为

### 3.1 主流程

1. **选素材**：选素材A（视频）、素材B（音频或视频）。
2. **自动对齐**：引擎自动计算对齐偏移 + 变速比，返回波形数据。
3. **交互预览**：前端展示素材A/B 双波形图，自动对齐结果落在波形上（B 波形按偏移对齐到 A 波形）。
   - 可**拖动** B 波形调整偏移量（时间轴刻度，步进精确到 0.1s）；
   - 可**试听**：前端 Web Audio API 实时混音——A 视频静音播放 + B 音频按当前偏移同步播放。
4. **导出**：用户感觉 OK 后点「下载」，引擎按用户调整后的 offset 重新混流输出最终视频。

### 3.2 对齐失败降级策略（三级降级）

用户明确：对齐失败时**不应报错阻断**，而要尽量给出一个"能卡点"的结果。降级链路：

1. **DTW 精确对齐**（首选）：全频段梅尔频谱 + 斜率约束 DTW，找最佳偏移 + 变速比。
2. **节拍粗对齐**（降级）：若 DTW 置信度低，用 BPM/节拍检测，让 B 的强拍尽量对齐 A 的节拍。
3. **从头铺设**（兜底）：节拍也对不上时，B 从 0 秒开始铺，UI 提示可手动拖动微调。

无论走哪条路径，**变速比始终应用**（B 按检测到的 tempo_ratio 变速后铺入，尽量让音乐节奏与画面舞蹈动作一致）。

降级结果在 UI 中明确提示当前对齐方式（`DTW / 节拍 / 从头`），并允许用户通过波形图拖动继续手动微调。

## 4. 架构总览

沿用现有分层（Electron UI ↔ HTTP 引擎），但交互职责分工有明确边界：

```
┌─────────────────────────────────────────────────────┐
│        Electron 桌面壳 (UI 层)                        │
│  React/Vite + 素材选择 + 波形图 + 拖动微调             │
│  + Web Audio 实时混音试听 + 播放/下载按钮              │
└──────────────┬──────────────────────────────────────┘
               │ HTTP (localhost)  —  波形数据 + 对齐结果 + 混流导出
┌──────────────▼──────────────────────────────────────┐
│            Python 处理引擎 (算法层)                    │
│  特征提取 → DTW/节拍对齐 → 波形降采样 → 变速 → 混流     │
│  任务队列 + 进度回传                                  │
└─────────────────────────────────────────────────────┘
```

**职责边界（经用户确认）**：
- **引擎**负责算法对齐（DTW/节拍/变速/波形降采样）+ 最终混流导出；
- **前端**负责波形图可视化、拖动微调、实时混音试听（Web Audio）、播放/下载交互。

## 5. 组件划分

| 单元 | 职责 | 关键依赖 | 接口 |
|---|---|---|---|
| **AudioAligner**（`engine/engine/align.py`） | 特征提取 + DTW 对齐 + 节拍粗对齐 + 变速计算 + 置信度评估 | librosa, numpy, scipy | 返回对齐结果 |
| **WaveformExtractor**（`engine/engine/waveform.py`） | 从 A/B 音轨降采样出低分辨率波形数据，供前端绘图 | numpy, ffmpeg | 返回波形数组 |
| **AudioMixer**（`engine/engine/audiotask.py`） | 变速（atempo/重采样）+ 混流（A 视频流 + B 音轨）→ 输出 MP4 | ffmpeg | 返回输出路径 |
| **`/audio-task` API**（`engine/engine/server.py`） | 音频任务：提交/查询/取消 | 标准库 | JSON |
| **AudioSwap UI**（`app/src/AudioSwap.tsx`） | 素材选择、波形图渲染、拖动微调、Web Audio 混音、播放/下载 | React, Web Audio API | 调用 IPC |
| **IPC 通道**（`app/electron/main.ts` + `preload.ts`） | 音频任务提交/查询/下载 | Electron | Promise |

## 6. Task API（独立接口，与现有裁剪任务隔离）

- `POST /audio-task`：`{ video_a_path, audio_b_path, output_path, params }`
  - `params`：`hop_length`（默认 512）、`n_mels`（默认 128）、`window_seconds`（对齐窗口，默认 60）、`max_slope`（DTW 斜率上限，默认 2.0 容忍 2 倍速）。
- `GET /audio-task/{id}`：轮询 `{ task_id, progress, status, message, align_result, preview }`
  - `align_result`：`{ offset_seconds, tempo_ratio, confidence, method }`
    - `method`：`"dtw" | "beat" | "zero"`（对齐方式）
    - `confidence`：`"high" | "low"`
  - `preview`（对齐完成后返回，供前端试听）：`{ video_a_path, audio_a_path, audio_b_path, waveform_a, waveform_b }`
    - `audio_a_path`：素材A 提取出的试听音轨（m4a/aac）
    - `audio_b_path`：素材B 的音轨（m4a/aac，供前端按偏移播放）
    - `waveform_a` / `waveform_b`：降采样的 RMS 包络数组，前端 Canvas 绘制波形
- `POST /audio-task/{id}/cancel`：取消任务。
- **对齐与导出分离**：`POST /audio-task` 返回对齐结果 + 波形数据后即 `DONE`（不自动混流）；用户调整 offset 后调用 `POST /audio-task/{id}/render` 传入最终 `offset_seconds`（和可选的 `tempo_ratio`）触发混流导出。

状态机：`QUEUED → RUNNING → DONE | FAILED | CANCELLED`（与现有裁剪任务一致）。

## 7. 对齐算法细节（方案A：全频段频谱 DTW）

1. **音轨提取**：ffmpeg 把 A/B 转成 16kHz 单声道 wav（对齐特征只需低采样率）。
2. **特征**：librosa 梅尔频谱（`n_mels=128`、`hop_length=512`）。对 A/B 各取对齐窗口（前 60s，可配）控制内存。
3. **DTW**：`librosa.sequence.dtw`，斜率约束（`max_slope=2`）容忍变速 + 端点惩罚。最佳路径导出：
   - **offset**：B 起点在 A 中的时间位置；
   - **tempo_ratio**：路径平均斜率（变速比）。
4. **置信度**：基于 DTW 路径代价归一化 + 路径一致性。低于阈值 → 降级到节拍粗对齐。
5. **节拍粗对齐**（降级）：`librosa.beat.beat_track` 检测 A/B 的 BPM 和节拍时刻，找强拍对齐的偏移。
6. **变速**：按 tempo_ratio 用 ffmpeg `atempo`（或 librosa `time_stretch`）对 B 变速；基本原速（ratio≈1）时走 `-c:a copy` 免重编码快路径。

## 8. 前端交互细节

### 8.1 波形图

- 前端用 Canvas 绘制双波形（A 在顶、B 在底），共享时间轴。
- 波形数据来自引擎降采样返回（如 10Hz 的 RMS 包络），体积小、绘图快。
- B 波形在 x 轴上的起始位置 = 当前 `offset_seconds`，拖动 B 波形整体移动即调整 offset。

### 8.2 实时混音试听

- 引擎把 A 的音轨和 B 的音轨分别转成前端可播放的格式（如 A 的音轨提取为 m4a/mp3，B 保持原始或转 m4a），随对齐结果一并返回 URL/路径。
- 前端用 Web Audio API：两个 `<audio>` 或 `AudioBufferSourceNode`，B 按 offset 延迟启动，与 A 同步播放。
- **A 视频画面**：前端 `<video>` 播放素材A，视频静音（`muted`），音频走 Web Audio 混合通道。拖动 offset 时实时更新 B 的启动偏移。
- 拖动停止（非连续拖动期间）才重新同步试听，避免拖动中频繁起停。

### 8.3 播放 / 下载按钮

- **播放**：触发上面的实时混音试听（A 视频 + 混音音频同步播放）。
- **下载**：把用户当前调整后的 `offset_seconds`（+ `tempo_ratio`）传给引擎 `POST /audio-task/{id}/render`，引擎混流出最终 MP4，完成后前端弹出「打开所在文件夹」。

## 9. 错误处理

| 场景 | 处理 |
|---|---|
| B 无法识别为音频 / 无音轨 | 报错提示 |
| A 无音轨 | 报错提示 |
| DTW 置信度低 | 降级节拍对齐，`align_result.method='beat'`，UI 提示 |
| 节拍也对不上 | 降级从头铺设，`method='zero'`，UI 提示可拖动微调 |
| 变速比超出容忍范围（>max_slope） | 报错提示可能不同曲 |
| 混流编码失败 | 复用现有 Renderer 的错误处理模式 |

## 10. 测试策略

- **对齐算法单元测试**：合成两段同源但偏移 + 轻微变速的音频，验证 DTW 返回的 `offset_seconds` / `tempo_ratio` 正确。
- **降级测试**：对合成音频叠加白噪声/人声干扰，验证 DTW 低置信时正确降级到 `beat` / `zero`。
- **变速测试**：验证 `atempo` 变速 + 混流后输出时长以素材A 为准（`-shortest`）。
- **UI 测试**：`AudioSwap.test.tsx` — 素材选择、波形图绘制、offset 拖动更新、播放/下载按钮状态。
- **集成验收**：真实舞蹈室录音端到端手动验收（含拖动微调 + 试听 + 下载）。

## 11. 里程碑（后续由 writing-plans 细化）

1. `align.py` 音频对齐核心（特征 + DTW + 节拍 + 变速 + 置信度/降级）
2. `waveform.py` 波形降采样 + 音轨提取（供前端预览/试听）
3. `audiotask.py` 混流导出管线（atempo 变速 + 混流 + 进度）
4. `/audio-task` 接口 + 引擎侧任务（对齐与渲染两阶段）
5. UI：素材选择 + 波形图 + 拖动微调 + Web Audio 试听 + 播放/下载
6. 依赖打包（librosa 进 PyInstaller）+ 测试

## 12. 依赖变更

- 引擎新增：`librosa`（含 `numba`、`scipy`、`soundfile` 等传递依赖）。打包体积增加（预估 +40~60MB）。
- 前端无新增运行时依赖（波形用 Canvas，混音用 Web Audio API）。
