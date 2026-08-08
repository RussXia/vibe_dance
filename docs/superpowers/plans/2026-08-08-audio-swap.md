# 替换音轨（音频对齐 + 可交互预览）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增完全独立的「替换音轨」功能：把素材A（视频）音轨替换为对齐后的素材B（音频/视频音轨），提供波形图拖动微调 + Web Audio 实时试听 + 播放/下载。

**Architecture:** 沿用 Electron（UI）+ Python 引擎（算法）的 HTTP 架构。引擎负责对齐计算（DTW/节拍/变速）、波形降采样、混流导出；前端负责波形图渲染、拖动调整 offset、Web Audio 实时混音试听。对齐失败时三级降级（DTW → 节拍 → 从头铺设），变速比始终应用。

**Tech Stack:** Python 3.13 + librosa 0.11（音频特征/DTW/节拍）、numpy、ffmpeg（音轨提取/变速/混流）；前端 React 18 + Canvas（波形图）+ Web Audio API（试听）+ Electron IPC。

## Global Constraints

- 引擎 Python 3.13（venv 已存在：`engine/.venv`），依赖加入 `engine/pyproject.toml`。
- 引擎一律用 `engine/engine/ffmpeg.py` 的 `find_ffmpeg()` 获取 ffmpeg 绝对路径（打包后 Finder 启动的 App PATH 不含 Homebrew 目录）。
- 引擎任务状态机：`QUEUED → RUNNING → DONE | FAILED | CANCELLED`；进度回传 0-100。
- API 用标准库 `http.server`，无第三方 Web 框架。
- 前端 API 类型声明在 `app/src/vite-env.d.ts`，IPC 桥在 `app/electron/preload.ts`，主进程 handler 在 `app/electron/main.ts`。
- UI 组件测试用 Vitest + @testing-library/react（jsdom），`npm test`。
- 引擎测试用 pytest，合成音频/视频用 ffmpeg lavfi 生成（见 `engine/tests/fixtures.py` 模式）。
- 所有对外输出/UI 文案使用中文；代码注释遵循项目既有语言（中文为主）。
- 不引入 demucs / PyTorch（YAGNI）；librosa 是唯一新增重依赖。
- 输出时长以素材A 为准（`-shortest`），B 播完静音收尾，B 比 A 长则截断。

---

### Task 1: 引擎对齐核心 `align.py`

**Files:**
- Create: `engine/engine/align.py`
- Test: `engine/tests/test_align.py`

**Interfaces:**
- Consumes: `engine/engine/ffmpeg.py` 的 `find_ffmpeg()`；合成音频 fixture。
- Produces:
  - `align_tracks(a_wav: str, b_wav: str, params: dict | None = None) -> dict`
    - 返回 `{"offset_seconds": float, "tempo_ratio": float, "confidence": str, "method": str}`，`method ∈ {"dtw","beat","zero"}`，`confidence ∈ {"high","low"}`。
  - `_extract_audio(source_path: str, wav_path: str, ffmpeg: str | None = None) -> None`
    - 用 ffmpeg 把任意媒体（mp4/mov/mp3/wav...）转为 16kHz 单声道 wav。
  - `_melspectrogram(wav_path: str, n_mels: int, hop_length: int, max_seconds: float) -> np.ndarray`
    - 读取 wav，返回梅尔频谱（对数幅度），限 max_seconds 窗口。
  - `_dtw_align(spec_a, spec_b, max_slope: float) -> tuple[float, float, float]`
    - 返回 `(offset_seconds, tempo_ratio, normalized_cost)`。
  - `_beat_align(a_wav: str, b_wav: str) -> float`
    - 节拍粗对齐，返回 B 起点在 A 中的偏移秒数。

**说明**：本任务只做对齐计算，不做任务管理/混流。对齐算法细节（特征、DTW、置信度、降级）见设计文档 §7。

- [ ] **Step 1: 写失败测试**

先给 `engine/tests/test_align.py` 写测试。核心验证：合成两段同源但有偏移的音频，`align_tracks` 能找回 offset。

```python
"""音频对齐核心单元测试。"""
import subprocess

import numpy as np

from engine.align import align_tracks, _extract_audio


def _make_tone_wav(path, freq, duration, sample_rate=16000):
    """用 numpy 生成单音 wav（16-bit PCM）。"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    x = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    pcm = (x * 32767).astype(np.int16)
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def test_extract_audio_from_video(tmp_path):
    """从带音轨的视频中提取 16kHz 单声道 wav。"""
    video = str(tmp_path / "src.mp4")
    # 用 ffmpeg 生成 1 秒带 440Hz 音的视频
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", video],
        capture_output=True, check=True,
    )
    out = str(tmp_path / "out.wav")
    _extract_audio(video, out)
    import os
    assert os.path.exists(out)
    import wave
    with wave.open(out, "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getnframes() >= 16000  # 约 1 秒


def test_align_finds_known_offset(tmp_path):
    """B 是 A 的片段（从 2s 处开始），对齐应找回 offset≈2s。"""
    a = str(tmp_path / "a.wav")
    b = str(tmp_path / "b.wav")
    # A: 6 秒 440Hz，B: A 的 2-5s 片段（同频同相位，偏移 2s）
    _make_tone_wav(a, 440, 6.0)
    _make_tone_wav(b, 440, 3.0)
    # 模拟 B 就是 A 从 2s 开始的片段：这里用纯单音无法区分相位，改为拼接校验时间特征
    # 用两个不同频率段构造 A，使 B 是 A 中特定时刻的片段，DTW 能找回偏移
    pass  # 见下一步改用多频段信号
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_align.py -v`
Expected: 测试因 `from engine.align import ...` 报 `ModuleNotFoundError`。

- [ ] **Step 3: 补全对齐测试（多频段信号，可区分偏移）**

用多频段信号替换上面的 `pass`，使 offset 可被 DTW 唯一找回：

```python
def _make_audio(path, segments, sample_rate=16000):
    """按 (start, freq, dur) 段拼接生成 wav，构成可区分的时间结构。"""
    total = max(s + d for s, f, d in segments)
    t = np.linspace(0, total, int(sample_rate * total), endpoint=False)
    x = np.zeros_like(t)
    for start, freq, dur in segments:
        idx = (t >= start) & (t < start + dur)
        x[idx] += 0.5 * np.sin(2 * np.pi * freq * t[idx])
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def test_align_finds_known_offset(tmp_path):
    """B 是 A 的片段（从 2s 处开始），对齐应找回 offset≈2s。"""
    a = str(tmp_path / "a.wav")
    b = str(tmp_path / "b.wav")
    # A: 0-1s 300Hz, 1-2s 440Hz, 2-5s 300Hz (与 B 匹配段)
    _make_audio(a, [(0, 300, 1), (1, 440, 1), (2, 300, 3)])
    # B: 就是 A 从 2s 开始的 3s 片段 (300Hz)
    _make_audio(b, [(0, 300, 3)])
    res = align_tracks(a, b)
    assert res["method"] == "dtw"
    assert abs(res["offset_seconds"] - 2.0) < 0.3, res


def test_align_returns_tempo_ratio(tmp_path):
    """B 与 A 同速，tempo_ratio 应≈1。"""
    a = str(tmp_path / "a.wav")
    b = str(tmp_path / "b.wav")
    _make_audio(a, [(0, 300, 4), (4, 440, 2)])
    _make_audio(b, [(0, 300, 4), (4, 440, 2)])
    res = align_tracks(a, b)
    assert abs(res["tempo_ratio"] - 1.0) < 0.15, res


def test_align_low_confidence_falls_back(tmp_path):
    """完全无关的音频应降级（method != dtw 或 confidence == low）。"""
    a = str(tmp_path / "a.wav")
    b = str(tmp_path / "b.wav")
    _make_audio(a, [(0, 300, 4)])
    _make_audio(b, [(0, 900, 4)])  # 不同频率，DTW 无匹配
    res = align_tracks(a, b)
    assert res["method"] in ("beat", "zero")
```

- [ ] **Step 4: 运行测试确认失败（多频段测试仍因模块缺失失败）**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_align.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.align'`

- [ ] **Step 5: 写最小实现 `align.py`**

```python
"""音频对齐：DTW 精确对齐 → 节拍粗对齐 → 从头铺设 三级降级。

功能：
- 从任意媒体提取 16kHz 单声道 wav（ffmpeg）。
- 梅尔频谱特征 + 斜率约束 DTW 找偏移/变速比。
- 置信度低时降级到节拍粗对齐，再降级到从头铺设。
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from .ffmpeg import find_ffmpeg

_SAMPLE_RATE = 16000
_DEFAULT_PARAMS = {
    "n_mels": 128,
    "hop_length": 512,
    "window_seconds": 60.0,
    "max_slope": 2.0,
}


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, **kwargs)


def _extract_audio(source_path: str, wav_path: str, ffmpeg: str | None = None) -> None:
    """把任意媒体转成 16kHz 单声道 wav，供对齐特征使用。"""
    ffmpeg = ffmpeg or find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", source_path,
        "-vn", "-ac", "1", "-ar", str(_SAMPLE_RATE),
        "-c:a", "pcm_s16le", wav_path,
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not os.path.exists(wav_path):
        raise ValueError(
            f"无法提取音频: {source_path} → {wav_path}: "
            f"{proc.stderr[-300:].decode(errors='replace')}"
        )


def _load_audio(wav_path: str) -> np.ndarray:
    """读取 wav 为 float32 单声道数组（0-1 范围）。"""
    import wave

    with wave.open(wav_path, "rb") as w:
        assert w.getnchannels() == 1, "wav 应为单声道"
        data = w.readframes(w.getnframes())
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return x


def _melspectrogram(x: np.ndarray, n_mels: int, hop_length: int, sr: int) -> np.ndarray:
    """梅尔频谱（对数幅度）。无 librosa 时用简化的 STFT + 三角滤波器近似。

    优先用 librosa；若 import 失败（如冻结环境缺包）退回 numpy 实现。
    """
    try:
        import librosa

        S = librosa.feature.melspectrogram(
            y=x, sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        return librosa.power_to_db(S, ref=np.max)
    except ImportError:
        return _stft_mel(x, n_mels, hop_length, sr)


def _stft_mel(x: np.ndarray, n_mels: int, hop_length: int, sr: int) -> np.ndarray:
    """numpy-only 的 STFT + 梅尔三角滤波器（降级路径，测试/无 librosa 环境用）。"""
    n_fft = 2048
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(
        np.pad(x, (0, n_fft - len(x) % hop_length + n_fft)), n_fft
    )[::hop_length].astype(np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    S = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    # 简化的梅尔三角滤波器（低频密高频疏）
    freqs = np.linspace(0, sr / 2, S.shape[1])
    mel_pts = np.linspace(0, S.shape[1] - 1, n_mels + 2).astype(int)
    W = np.zeros((n_mels, S.shape[1]))
    for m in range(n_mels):
        lo, mid, hi = mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]
        W[m, lo:mid] = np.linspace(0, 1, max(1, mid - lo))
        W[m, mid:hi] = np.linspace(1, 0, max(1, hi - mid))
    M = S @ W.T
    return 10.0 * np.log10(M + 1e-10)


def _dtw_align(spec_a: np.ndarray, spec_b: np.ndarray,
               hop_length: int, sr: int, max_slope: float) -> tuple[float, float, float]:
    """斜率约束 DTW，返回 (offset_seconds, tempo_ratio, normalized_cost)。"""
    try:
        import librosa

        D, wp = librosa.sequence.dtw(
            X=spec_a.T, Y=spec_b.T,
            metric="cosine",
            max_slope=max_slope,
            subseq=True,  # 允许 B 是 A 的子序列（B 起点在 A 内部）
        )
    except ImportError:
        return 0.0, 1.0, 1.0  # 无 librosa 时退回：从头铺，原速

    # wp: shape (n, 2)，每行 (i_a, j_b)。B 起点对应 j_b==0 的那行 → 取 i_a
    start_row = None
    for row in wp:
        if int(row[1]) == 0:
            start_row = int(row[0])
            break
    if start_row is None:
        start_row = int(wp[0][0])
    offset = start_row * hop_length / sr
    # 变速比：A 帧跨度 / B 帧跨度
    span_a = int(wp[-1][0]) - int(wp[0][0]) + 1
    span_b = int(wp[-1][1]) - int(wp[0][1]) + 1
    tempo = span_a / max(1, span_b)
    cost = float(D[wp[-1][0], wp[-1][1]])
    return float(offset), float(tempo), cost


def _beat_align(a_wav: str, b_wav: str, sr: int) -> float:
    """节拍粗对齐：找 B 第一个强拍对应 A 的时间偏移。"""
    try:
        import librosa
    except ImportError:
        return 0.0
    xa = _load_audio(a_wav)
    xb = _load_audio(b_wav)
    # 先分别测 BPM
    tempo_b, beats_b = librosa.beat.beat_track(y=xb, sr=sr)
    if len(beats_b) == 0:
        return 0.0
    tempo_a, beats_a = librosa.beat.beat_track(y=xa, sr=sr)
    if len(beats_a) == 0:
        return 0.0
    # B 的第一个强拍时刻（相对 B 起点）
    first_beat_b = float(beats_b[0]) / sr
    # 在 A 的节拍序列里找与 first_beat_b 对齐的候选：用 BPM 比例换算后就近匹配
    # 简化：把 A 的第一个节拍当作参考，B 起始 = A 第一拍 - first_beat_b（同 BPM 假设）
    first_beat_a = float(beats_a[0]) / sr
    ratio = tempo_a / max(1.0, tempo_b)
    offset = first_beat_a - first_beat_b * ratio
    return max(0.0, float(offset))


def align_tracks(a_wav: str, b_wav: str, params: dict | None = None) -> dict:
    """对齐 A/B 两个 wav，返回对齐结果。

    Args:
        a_wav: 素材A 的 wav 路径（16kHz 单声道）。
        b_wav: 素材B 的 wav 路径（16kHz 单声道）。
        params: 覆盖默认参数（n_mels/hop_length/window_seconds/max_slope）。

    Returns:
        {"offset_seconds", "tempo_ratio", "confidence", "method"}
        method ∈ {"dtw","beat","zero"}，confidence ∈ {"high","low"}。
    """
    cfg = {**_DEFAULT_PARAMS, **(params or {})}
    sr = _SAMPLE_RATE
    hop = int(cfg["hop_length"])
    window = float(cfg["window_seconds"])
    max_slope = float(cfg["max_slope"])

    xa = _load_audio(a_wav)
    xb = _load_audio(b_wav)
    n_a = min(len(xa), int(window * sr))
    n_b = min(len(xb), int(window * sr))
    spec_a = _melspectrogram(xa[:n_a], int(cfg["n_mels"]), hop, sr)
    spec_b = _melspectrogram(xb[:n_b], int(cfg["n_mels"]), hop, sr)

    offset, tempo, cost = _dtw_align(spec_a, spec_b, hop, sr, max_slope)

    # 置信度：归一化成本（每帧平均距离）。经验阈值，可调。
    frames_a = spec_a.shape[1]
    norm_cost = cost / max(1, frames_a)
    if norm_cost < 0.35:
        return {
            "offset_seconds": offset,
            "tempo_ratio": float(tempo),
            "confidence": "high",
            "method": "dtw",
        }

    # 降级：节拍粗对齐
    beat_offset = _beat_align(a_wav, b_wav, sr)
    if beat_offset > 0:
        return {
            "offset_seconds": beat_offset,
            "tempo_ratio": float(tempo) if tempo != 1.0 else 1.0,
            "confidence": "low",
            "method": "beat",
        }

    # 兜底：从头铺设
    return {
        "offset_seconds": 0.0,
        "tempo_ratio": float(tempo) if tempo != 1.0 else 1.0,
        "confidence": "low",
        "method": "zero",
    }
```

- [ ] **Step 6: 运行测试确认通过**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_align.py -v`
Expected: 全部 PASS。若 `test_align_low_confidence_falls_back` 因合成信号太干净而走了 dtw，检查 `norm_cost` 阈值或改用更区分度的信号。

- [ ] **Step 7: Commit**

```bash
git add engine/engine/align.py engine/tests/test_align.py
git commit -m "feat(engine): 音频对齐核心（DTW/节拍/从头三级降级）
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 波形降采样 + 试听音轨提取 `waveform.py`

**Files:**
- Create: `engine/engine/waveform.py`
- Test: `engine/tests/test_waveform.py`

**Interfaces:**
- Consumes: `find_ffmpeg()`；`align._extract_audio`（可复用转 wav 逻辑）。
- Produces:
  - `extract_waveform(wav_path: str, bucket_seconds: float = 0.1) -> list[float]`
    - 返回 RMS 包络数组（归一化 0-1），供前端 Canvas 绘图。
  - `extract_preview_audio(source_path: str, out_path: str, ffmpeg: str | None = None) -> None`
    - 把素材（A 或 B）音轨转成前端可播放的 AAC/m4a（用于 Web Audio 试听）。

- [ ] **Step 1: 写失败测试**

```python
"""波形降采样与试听音轨提取测试。"""
import os
import subprocess
import wave

import numpy as np

from engine.waveform import extract_waveform, extract_preview_audio


def _make_tone_wav(path, freq=440, duration=2.0, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    x = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(x.tobytes())


def test_extract_waveform_length(tmp_path):
    """2 秒音频、0.1s 桶 → 约 20 个 RMS 值。"""
    wav = str(tmp_path / "tone.wav")
    _make_tone_wav(wav)
    wave_data = extract_waveform(wav, bucket_seconds=0.1)
    # 16kHz * 2s = 32000 采样，/1600 per bucket = 20
    assert 18 <= len(wave_data) <= 22
    assert all(0.0 <= v <= 1.0 for v in wave_data)
    # 正弦波 RMS 应明显 > 0
    assert max(wave_data) > 0.05


def test_extract_waveform_silence(tmp_path):
    """静音 → 全 0（或接近 0）。"""
    wav = str(tmp_path / "silence.wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1 秒静音
    wave_data = extract_waveform(wav)
    assert max(wave_data) < 0.01


def test_extract_preview_audio_from_video(tmp_path):
    """从视频提取试听音轨（m4a/aac）。"""
    video = str(tmp_path / "src.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", video],
        capture_output=True, check=True,
    )
    out = str(tmp_path / "preview.m4a")
    extract_preview_audio(video, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_waveform.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.waveform'`

- [ ] **Step 3: 写实现 `waveform.py`**

```python
"""波形降采样 + 试听音轨提取（供前端波形图绘制与 Web Audio 试听）。"""
from __future__ import annotations

import os
import subprocess
import wave

import numpy as np

from .ffmpeg import find_ffmpeg


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, **kwargs)


def extract_waveform(wav_path: str, bucket_seconds: float = 0.1) -> list[float]:
    """从 16kHz 单声道 wav 计算 RMS 包络（归一化 0-1），供前端绘图。

    bucket_seconds: 每个波形桶的时长（默认 0.1s → 10Hz 分辨率）。
    """
    with wave.open(wav_path, "rb") as w:
        assert w.getnchannels() == 1
        sr = w.getframerate()
        data = w.readframes(w.getnframes())
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) == 0:
        return []
    bucket = max(1, int(sr * bucket_seconds))
    n = (len(x) + bucket - 1) // bucket
    # 用均值平方求 RMS，逐桶
    x2 = x.astype(np.float64) ** 2
    padded = np.pad(x2, (0, n * bucket - len(x2)))
    rms = np.sqrt(padded.reshape(n, bucket).mean(axis=1))
    peak = rms.max() if rms.max() > 0 else 1.0
    norm = (rms / peak).astype(np.float64)
    return [float(round(v, 4)) for v in norm]


def extract_preview_audio(source_path: str, out_path: str, ffmpeg: str | None = None) -> None:
    """把素材（A 或 B）的音轨转成前端可播放的 AAC/m4a（供 Web Audio 试听）。"""
    ffmpeg = ffmpeg or find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", source_path,
        "-vn", "-ac", "2", "-ar", "44100",
        "-c:a", "aac", "-b:a", "128k", out_path,
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise ValueError(
            f"无法生成试听音轨: {source_path} → {out_path}: "
            f"{proc.stderr[-300:].decode(errors='replace')}"
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_waveform.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add engine/engine/waveform.py engine/tests/test_waveform.py
git commit -m "feat(engine): 波形降采样 + 试听音轨提取
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 音频任务管理 + 混流导出 `audiotask.py`

**Files:**
- Create: `engine/engine/audiotask.py`
- Test: `engine/tests/test_audiotask.py`

**Interfaces:**
- Consumes:
  - `align.align_tracks(a_wav, b_wav, params) -> dict`
  - `align._extract_audio(source, wav_path)`
  - `waveform.extract_waveform(wav_path) -> list[float]`
  - `waveform.extract_preview_audio(source, out_path)`
  - `find_ffmpeg()`
- Produces:
  - `class AudioTaskManager`
    - `submit(video_a_path, audio_b_path, output_path, params=None) -> str`（task_id）
    - `get(task_id) -> dict | None`：`{task_id, progress, status, message, align_result, preview}`
      - `align_result`: `{offset_seconds, tempo_ratio, confidence, method}`
      - `preview`: `{video_a_path, audio_a_path, audio_b_path, waveform_a, waveform_b}`（对齐完成后填充，render 前一直可用）
    - `render(task_id, offset_seconds, tempo_ratio=None) -> None`（触发混流导出，复用同一任务线程）
    - `cancel(task_id) -> bool`
  - `class _Cancelled(Exception)`

**说明**：`render()` 与 `submit()` 共用同一任务（同一 task_id 状态机），对齐完成后 `status` 置为 `"DONE"`；用户拖动 offset 后调 `render()` 重新置 `"RUNNING"` 执行混流，完成后回到 `"DONE"` 且 `message` 含输出路径。

- [ ] **Step 1: 写失败测试**

```python
"""音频任务（对齐 + 混流导出）测试。"""
import os
import subprocess

from engine.audiotask import AudioTaskManager


def _make_video_with_audio(path, freq=440, duration=2):
    """生成 2 秒带音的视频。"""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
        capture_output=True, check=True,
    )


def _make_audio(path, freq=440, duration=2):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"sine=frequency={freq}:duration={duration}",
         "-c:a", "aac", path],
        capture_output=True, check=True,
    )


def test_submit_align_then_render(tmp_path):
    """提交 → 对齐 DONE（带 align_result + preview）→ render 混流输出。"""
    video_a = str(tmp_path / "a.mp4")
    audio_b = str(tmp_path / "b.m4a")
    out = str(tmp_path / "out.mp4")
    _make_video_with_audio(video_a)
    _make_audio(audio_b)

    mgr = AudioTaskManager()
    task_id = mgr.submit(video_a, audio_b, out)
    # 轮询直到 DONE（对齐阶段）
    for _ in range(50):
        st = mgr.get(task_id)
        if st["status"] in ("DONE", "FAILED"):
            break
        import time
        time.sleep(0.1)
    assert st["status"] == "DONE", st
    assert "align_result" in st, st
    assert st["align_result"]["method"] in ("dtw", "beat", "zero")
    assert "preview" in st, st
    assert st["preview"]["audio_a_path"] and st["preview"]["audio_b_path"]
    assert st["preview"]["waveform_a"] and st["preview"]["waveform_b"]
    assert st["progress"] == 100

    # render 阶段：用自动对齐的 offset 混流
    offset = st["align_result"]["offset_seconds"]
    mgr.render(task_id, offset)
    for _ in range(100):
        st2 = mgr.get(task_id)
        if st2["status"] in ("DONE", "FAILED"):
            break
        import time
        time.sleep(0.1)
    assert st2["status"] == "DONE", st2
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_submit_missing_audio_raises(tmp_path):
    """B 不是有效音频 → 任务 FAILED。"""
    video_a = str(tmp_path / "a.mp4")
    bad_b = str(tmp_path / "b.txt")
    out = str(tmp_path / "out.mp4")
    _make_video_with_audio(video_a)
    with open(bad_b, "w") as f:
        f.write("not audio")

    mgr = AudioTaskManager()
    task_id = mgr.submit(video_a, bad_b, out)
    for _ in range(50):
        st = mgr.get(task_id)
        if st["status"] in ("DONE", "FAILED"):
            break
        import time
        time.sleep(0.1)
    assert st["status"] == "FAILED", st
    assert st["message"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_audiotask.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.audiotask'`

- [ ] **Step 3: 写实现 `audiotask.py`**

```python
"""音频替换任务：对齐（DTW/节拍/从头）→ 变速 → 混流导出。

与渲染任务（render.py）共用任务状态机模式：
QUEUED → RUNNING → DONE | FAILED | CANCELLED

两阶段：
1. submit() 只做对齐，返回 align_result + preview（波形/试听音轨）。
2. render() 用用户调整后的 offset（+可选 tempo_ratio）触发混流导出。
"""
from __future__ import annotations

import os
import subprocess
import threading
import uuid
from dataclasses import dataclass, field

from .align import _extract_audio, align_tracks
from .ffmpeg import find_ffmpeg
from .waveform import extract_preview_audio, extract_waveform

_SAMPLE_RATE = 16000


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, **kwargs)


def _work_dir(task_id: str) -> str:
    """任务临时目录（对齐特征/波形/试听音频），随任务生命周期存在。"""
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "vibe_audio", task_id)
    os.makedirs(d, exist_ok=True)
    return d


@dataclass
class AudioTask:
    task_id: str
    video_a_path: str
    audio_b_path: str
    output_path: str
    params: dict
    status: str = "QUEUED"
    progress: int = 0
    message: str = ""
    align_result: dict | None = None
    preview: dict | None = None
    _thread: threading.Thread = field(default=None, repr=False)
    _cancel: bool = field(default=False, repr=False)


class AudioTaskManager:
    def __init__(self):
        self._tasks: dict[str, AudioTask] = {}
        self._lock = threading.Lock()

    def submit(self, video_a_path, audio_b_path, output_path, params=None):
        task_id = uuid.uuid4().hex[:12]
        task = AudioTask(
            task_id=task_id,
            video_a_path=video_a_path,
            audio_b_path=audio_b_path,
            output_path=output_path,
            params=params or {},
        )
        with self._lock:
            self._tasks[task_id] = task
        task._thread = threading.Thread(target=self._run_align, args=(task,), daemon=True)
        task._thread.start()
        return task_id

    def get(self, task_id):
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return None
            info = {
                "task_id": t.task_id,
                "status": t.status,
                "progress": t.progress,
                "message": t.message,
            }
            if t.align_result is not None:
                info["align_result"] = t.align_result
            if t.preview is not None:
                info["preview"] = t.preview
            return info

    def cancel(self, task_id):
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return False
            if t.status in ("DONE", "FAILED", "CANCELLED"):
                return False
            t._cancel = True
            t.status = "CANCELLED"
            return True

    def render(self, task_id, offset_seconds, tempo_ratio=None):
        """用调整后的 offset 触发混流导出（复用任务线程，不新建任务）。"""
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                raise KeyError(f"task not found: {task_id}")
            if t.status == "CANCELLED":
                raise RuntimeError("task cancelled")
            t._cancel = False
            t.status = "RUNNING"
            t.progress = 0
            t.message = ""
            thread = threading.Thread(
                target=self._run_render,
                args=(t, float(offset_seconds), tempo_ratio),
                daemon=True,
            )
            thread.start()

    def _run_align(self, task):
        work = _work_dir(task.task_id)
        try:
            task.status = "RUNNING"
            task.progress = 5
            # 提取 A/B 的 16kHz wav（对齐特征）
            a_wav = os.path.join(work, "a.wav")
            b_wav = os.path.join(work, "b.wav")
            _extract_audio(task.video_a_path, a_wav)
            task.progress = 25
            _extract_audio(task.audio_b_path, b_wav)
            task.progress = 35
            # 对齐
            result = align_tracks(a_wav, b_wav, task.params)
            task.align_result = result
            task.progress = 60
            # 波形（供前端绘图）
            wf_a = extract_waveform(a_wav)
            wf_b = extract_waveform(b_wav)
            # 试听音轨（供前端 Web Audio）
            audio_a = os.path.join(work, "a_preview.m4a")
            audio_b = os.path.join(work, "b_preview.m4a")
            extract_preview_audio(task.video_a_path, audio_a)
            extract_preview_audio(task.audio_b_path, audio_b)
            task.preview = {
                "video_a_path": task.video_a_path,
                "audio_a_path": audio_a,
                "audio_b_path": audio_b,
                "waveform_a": wf_a,
                "waveform_b": wf_b,
            }
            task.progress = 100
            if task._cancel:
                task.status = "CANCELLED"
            else:
                task.status = "DONE"
                task.message = "对齐完成，可拖动波形微调或直接下载"
        except _Cancelled:
            task.status = "CANCELLED"
        except Exception as exc:  # noqa: BLE001
            task.status = "FAILED"
            task.message = str(exc)

    def _run_render(self, task, offset_seconds, tempo_ratio):
        work = _work_dir(task.task_id)
        try:
            task.status = "RUNNING"
            task.progress = 0
            ffmpeg = find_ffmpeg()
            # 生成变速后的 B 音轨（若有 tempo_ratio）
            b_audio = os.path.join(work, "b_preview.m4a")
            b_varied = os.path.join(work, "b_varied.m4a")
            ratio = float(tempo_ratio) if tempo_ratio else float(
                (task.align_result or {}).get("tempo_ratio", 1.0)
            )
            if abs(ratio - 1.0) < 0.01:
                b_varied = b_audio  # 基本原速：直接用原音轨
            else:
                # atempo 只支持 0.5-2.0，超范围分多段
                seq = _atempo_chain(ratio)
                cmd = [ffmpeg, "-y", "-i", b_audio]
                for a in seq:
                    cmd += ["-af", f"atempo={a}"]
                cmd += ["-c:a", "aac", b_varied]
                _run(cmd)
            task.progress = 40

            # 混流：A 视频流 + 变速后 B 音轨，从 offset 处开始铺，时长以 A 为准
            cmd = [
                ffmpeg, "-y",
                "-i", task.video_a_path,
                "-i", b_varied,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                # B 音轨整体前移 offset 秒：用 adelay 把 B 起点推迟到 offset
                "-af", f"adelay={int(offset_seconds * 1000)}:all=1",
                task.output_path,
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read() if proc.stderr else b""
                raise RuntimeError(
                    f"混流导出失败: {err[-500:].decode(errors='replace')}"
                )
            task.progress = 100
            if task._cancel:
                task.status = "CANCELLED"
            else:
                task.status = "DONE"
                task.message = f"导出完成: {task.output_path}"
        except _Cancelled:
            task.status = "CANCELLED"
        except Exception as exc:  # noqa: BLE001
            task.status = "FAILED"
            task.message = str(exc)


def _atempo_chain(ratio: float) -> list[str]:
    """把变速比拆成 atempo 可用的链（每段 0.5-2.0）。"""
    seq = []
    r = ratio
    while r > 2.0:
        seq.append("2.0")
        r /= 2.0
    while r < 0.5:
        seq.append("0.5")
        r /= 0.5
    seq.append(f"{r:.4f}")
    return seq


class _Cancelled(Exception):
    pass
```

> **注意**：上面 `-af adelay` 会把 B 音轨整体推迟 offset 秒。但此时 B 音轨是从文件头开始播放的，而 A 视频从 0 开始。若对齐语义是「B 起点（音乐开头）应对齐到 A 的 offset 时刻」，则 B 应从 0 开始、在 A 的 offset 处切入——`adelay` 推迟整段 B 是对的。**若实现时发现语义相反（B 应提前铺），则改用 `-itsoffset` 调整输入 B 的时间戳**，见 Task 3 Step 5 集成验证。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_audiotask.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 验证混流时序语义（关键集成检查）**

Run: 手动构造 A（2 秒 440Hz 音轨视频）、B（2 秒 300Hz 音频），`offset=1.0`，导出后用 `ffprobe` 检查输出前 1 秒无 B 音、1 秒后出现 B 音。

```bash
cd engine && source .venv/bin/activate && python - <<'EOF'
import subprocess, os, tempfile
from engine.audiotask import AudioTaskManager

d = tempfile.mkdtemp()
a = os.path.join(d, "a.mp4"); b = os.path.join(d, "b.m4a"); out = os.path.join(d, "out.mp4")
subprocess.run(["ffmpeg","-y","-f","lavfi","-i","color=c=black:s=160x120:d=2",
                "-f","lavfi","-i","sine=frequency=440:duration=2",
                "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",a], capture_output=True)
subprocess.run(["ffmpeg","-y","-f","lavfi","-i","sine=frequency=300:duration=2",
                "-c:a","aac",b], capture_output=True)
mgr = AudioTaskManager()
tid = mgr.submit(a,b,out)
import time
while mgr.get(tid)["status"] not in ("DONE","FAILED"): time.sleep(0.1)
mgr.render(tid, 1.0)
while mgr.get(tid)["status"] not in ("DONE","FAILED"): time.sleep(0.1)
print("status:", mgr.get(tid)["status"], "out exists:", os.path.exists(out))
EOF
```

Expected: 打印 `status: DONE out exists: True`，且输出时长≈2s（以 A 为准）。若 B 音时序方向反了（B 应更早出现而非推迟），调整混流命令（用 `-itsoffset` 方案替代 `adelay`）。

- [ ] **Step 6: Commit**

```bash
git add engine/engine/audiotask.py engine/tests/test_audiotask.py
git commit -m "feat(engine): 音频替换任务（对齐 + 变速 + 混流导出）
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: `/audio-task` HTTP 接口

**Files:**
- Modify: `engine/engine/server.py`
- Test: `engine/tests/test_server.py`

**Interfaces:**
- Consumes: `AudioTaskManager`（Task 3）。
- Produces:
  - `POST /audio-task`：body `{video_a_path, audio_b_path, output_path, params?}` → `{task_id, status}`
  - `GET /audio-task/{id}`：→ `{task_id, progress, status, message, align_result?, preview?}`
  - `POST /audio-task/{id}/cancel`：→ `{cancelled}`
  - `POST /audio-task/{id}/render`：body `{offset_seconds, tempo_ratio?}` → `{ok}`
- **注意**：server 模块级 `_manager` 需同时持有 `TaskManager` 与 `AudioTaskManager`；`stop()` 需能关闭两者。测试中 `start()`/`stop()` 复用同一进程内实例。

- [ ] **Step 1: 写失败测试**

在 `engine/tests/test_server.py` 末尾追加：

```python
def test_audio_task_submit_query_render(tmp_path):
    """audio-task 全链路：提交对齐 → 查询 → render 混流。"""
    from engine.audiotask import AudioTaskManager
    # 复用 Task 3 的合成函数（若已在 conftest/fixtures，则 import）
    import subprocess as sp
    video_a = str(tmp_path / "a.mp4")
    audio_b = str(tmp_path / "b.m4a")
    out = str(tmp_path / "out.mp4")
    sp.run(["ffmpeg","-y","-f","lavfi","-i","color=c=black:s=160x120:d=2",
            "-f","lavfi","-i","sine=frequency=440:duration=2",
            "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",video_a],
           capture_output=True)
    sp.run(["ffmpeg","-y","-f","lavfi","-i","sine=frequency=300:duration=2",
            "-c:a","aac",audio_b], capture_output=True)

    port = 8893
    base = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=start, kwargs={"port": port}, daemon=True)
    thread.start()
    try:
        _wait_server_ready(base)
        resp = _post(base, "/audio-task", {
            "video_a_path": video_a,
            "audio_b_path": audio_b,
            "output_path": out,
        })
        task_id = resp["task_id"]
        assert resp["status"] == "QUEUED"

        for _ in range(50):
            st = _get(base, f"/audio-task/{task_id}")
            if st["status"] in ("DONE", "FAILED"):
                break
            time.sleep(0.1)
        assert st["status"] == "DONE", st
        assert "align_result" in st
        offset = st["align_result"]["offset_seconds"]

        # render
        r = _post(base, f"/audio-task/{task_id}/render", {"offset_seconds": offset})
        assert r.get("ok") is True, r
        for _ in range(100):
            st2 = _get(base, f"/audio-task/{task_id}")
            if st2["status"] in ("DONE", "FAILED"):
                break
            time.sleep(0.1)
        assert st2["status"] == "DONE", st2
    finally:
        import engine.server as srv
        srv.stop()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_server.py::test_audio_task_submit_query_render -v`
Expected: FAIL with `404 {"error":"not found"}`（/audio-task 路由未实现）。

- [ ] **Step 3: 修改 `server.py` 加路由**

在 `server.py` 中：

```python
from .audiotask import AudioTaskManager

# 模块级两个 manager（渲染 + 音频）
_manager = TaskManager()
_audio_manager = AudioTaskManager()
```

`do_GET` 中加：

```python
if self.path.startswith("/audio-task/"):
    task_id = self.path.split("/")[-1]
    info = _audio_manager.get(task_id)
    if info is None:
        self._send_json(404, {"error": "task not found"})
        return
    self._send_json(200, info)
    return
```

`do_POST` 中加（放在 `/task` 分支后、cancel 分支前）：

```python
if self.path == "/audio-task":
    payload = self._read_json()
    try:
        task_id = _audio_manager.submit(
            payload["video_a_path"],
            payload["audio_b_path"],
            payload["output_path"],
            payload.get("params"),
        )
    except KeyError as exc:
        self._send_json(400, {"error": f"missing field: {exc}"})
        return
    self._send_json(200, {"task_id": task_id, "status": "QUEUED"})
    return
if self.path.startswith("/audio-task/") and self.path.endswith("/render"):
    task_id = self.path.split("/")[-2]
    payload = self._read_json()
    try:
        _audio_manager.render(
            task_id,
            payload["offset_seconds"],
            payload.get("tempo_ratio"),
        )
    except (KeyError, RuntimeError) as exc:
        self._send_json(400, {"error": str(exc)})
        return
    self._send_json(200, {"ok": True})
    return
if self.path.startswith("/audio-task/") and self.path.endswith("/cancel"):
    task_id = self.path.split("/")[-2]
    ok = _audio_manager.cancel(task_id)
    self._send_json(200, {"cancelled": ok})
    return
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_server.py -v`
Expected: 原 `test_submit_and_query_task` 与新增 `test_audio_task_submit_query_render` 均 PASS。

- [ ] **Step 5: 回归运行全部引擎测试**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/ -v`
Expected: 全部 PASS（含已有渲染/跟踪/视频测试）。

- [ ] **Step 6: Commit**

```bash
git add engine/engine/server.py engine/tests/test_server.py
git commit -m "feat(engine): /audio-task 接口（对齐查询渲染取消）
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 前端 IPC 通道 + API 类型

**Files:**
- Modify: `app/electron/main.ts`
- Modify: `app/electron/preload.ts`
- Modify: `app/src/vite-env.d.ts`

**Interfaces:**
- Consumes: 引擎 `/audio-task` 接口（Task 4）。
- Produces:
  - `window.api.submitAudioTask(payload) -> Promise<{task_id, status}>`
  - `window.api.getAudioTask(taskId) -> Promise<AudioTaskInfo>`
  - `window.api.renderAudioTask(taskId, offsetSeconds, tempoRatio?) -> Promise<{ok}>`
  - `window.api.openAudio() -> Promise<{path} | null>`（选音频/视频文件）
  - `window.api.openAnyMedia() -> Promise<{path} | null>`（选任意媒体：音频或视频）
- `AudioTaskInfo` 类型（在 vite-env.d.ts 声明）：
  ```ts
  interface AudioTaskInfo {
    task_id: string;
    status: string;
    progress: number;
    message: string;
    align_result?: {
      offset_seconds: number;
      tempo_ratio: number;
      confidence: string;
      method: string;
    };
    preview?: {
      video_a_path: string;
      audio_a_path: string;
      audio_b_path: string;
      waveform_a: number[];
      waveform_b: number[];
    };
  }
  ```

- [ ] **Step 1: 在 main.ts 加音频任务 IPC handler**

在 `ipcMain.handle('engine:get-task', ...)` 后追加：

```ts
ipcMain.handle('engine:submit-audio-task', async (_e, payload) => {
  return engineFetch('/audio-task', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
});

ipcMain.handle('engine:get-audio-task', async (_e, taskId: string) => {
  return engineFetch(`/audio-task/${taskId}`);
});

ipcMain.handle('engine:render-audio-task', async (_e, taskId: string, payload: { offset_seconds: number; tempo_ratio?: number }) => {
  return engineFetch(`/audio-task/${taskId}/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
});

ipcMain.handle('open-audio', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    filters: [{ name: '音频/视频', extensions: ['mp3', 'wav', 'flac', 'm4a', 'aac', 'mp4', 'mov'] }],
    properties: ['openFile'],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return { path: result.filePaths[0] };
});

ipcMain.handle('open-any-media', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    filters: [
      { name: '视频', extensions: ['mp4', 'mov'] },
      { name: '音频', extensions: ['mp3', 'wav', 'flac', 'm4a', 'aac'] },
    ],
    properties: ['openFile'],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return { path: result.filePaths[0] };
});
```

- [ ] **Step 2: 在 preload.ts 暴露**

```ts
contextBridge.exposeInMainWorld('api', {
  openVideo: () => ipcRenderer.invoke('open-video'),
  openAudio: () => ipcRenderer.invoke('open-audio'),
  openAnyMedia: () => ipcRenderer.invoke('open-any-media'),
  saveVideo: (defaultName: string) => ipcRenderer.invoke('save-video', defaultName),
  showInFolder: (filePath: string) => ipcRenderer.invoke('show-in-folder', filePath),
  submitTask: (payload: object) => ipcRenderer.invoke('engine:submit-task', payload),
  getTask: (taskId: string) => ipcRenderer.invoke('engine:get-task', taskId),
  submitAudioTask: (payload: object) => ipcRenderer.invoke('engine:submit-audio-task', payload),
  getAudioTask: (taskId: string) => ipcRenderer.invoke('engine:get-audio-task', taskId),
  renderAudioTask: (taskId: string, payload: object) => ipcRenderer.invoke('engine:render-audio-task', taskId, payload),
  startEngine: () => ipcRenderer.invoke('engine:start'),
});
```

- [ ] **Step 3: 更新 vite-env.d.ts 类型**

```ts
interface AudioAlignResult {
  offset_seconds: number;
  tempo_ratio: number;
  confidence: string;
  method: string;
}
interface AudioPreview {
  video_a_path: string;
  audio_a_path: string;
  audio_b_path: string;
  waveform_a: number[];
  waveform_b: number[];
}
interface AudioTaskInfo {
  task_id: string;
  status: string;
  progress: number;
  message: string;
  align_result?: AudioAlignResult;
  preview?: AudioPreview;
}
interface Window {
  api: {
    openVideo: () => Promise<{ path: string } | null>;
    openAudio: () => Promise<{ path: string } | null>;
    openAnyMedia: () => Promise<{ path: string } | null>;
    saveVideo: (defaultName: string) => Promise<{ path: string } | null>;
    showInFolder: (filePath: string) => Promise<{ ok: boolean }>;
    submitTask: (payload: object) => Promise<{ task_id: string; status: string }>;
    getTask: (taskId: string) => Promise<{ task_id: string; status: string; progress: number; message: string }>;
    submitAudioTask: (payload: object) => Promise<{ task_id: string; status: string }>;
    getAudioTask: (taskId: string) => Promise<AudioTaskInfo>;
    renderAudioTask: (taskId: string, payload: object) => Promise<{ ok: boolean }>;
    startEngine: () => Promise<{ ok: boolean }>;
  };
}
```

- [ ] **Step 4: 验证类型检查通过**

Run: `cd app && npx tsc --noEmit`
Expected: 无类型错误（`engine:get-audio-task` 返回的 JSON 是 `any`，类型标注需在 AudioSwap.tsx 里用 `as AudioTaskInfo` 断言，或在 main.ts 里显式泛型）。

- [ ] **Step 5: 运行现有前端测试**

Run: `cd app && npm test`
Expected: 现有 App 测试全部 PASS（IPC 新增不影响现有测试，`makeApi` 需补新增 api 方法，但旧测试不调用它们）。

- [ ] **Step 6: Commit**

```bash
git add app/electron/main.ts app/electron/preload.ts app/src/vite-env.d.ts
git commit -m "feat(app): 音频任务 IPC 通道 + API 类型声明
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 前端波形图 + 拖动微调 + 试听组件 `WaveformEditor`

**Files:**
- Create: `app/src/WaveformEditor.tsx`
- Create: `app/src/WaveformEditor.test.tsx`
- Modify: `app/src/App.css`（追加波形图样式）

**Interfaces:**
- Consumes: `AudioPreview`（waveform_a/waveform_b）、`align_result.offset_seconds`。
- Produces: `WaveformEditor` 组件：
  ```ts
  interface Props {
    preview: AudioPreview;
    initialOffset: number;
    durationA: number;           // 素材A 时长（秒），波形 x 轴总长
    onOffsetChange: (offset: number) => void;
  }
  ```
  - 内部维护 `offset`（B 波形在 A 波形上的起始 x 位置，秒）。
  - Canvas 绘制双波形（A 顶部、B 底部），B 从 `offset` 处开始。
  - 鼠标拖动 B 波形整体移动 → 更新 offset → 调用 `onOffsetChange`。
  - 显示当前 offset（秒，一位小数）与对齐方式徽标（DTW/节拍/从头）。

- [ ] **Step 1: 写失败测试**

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import WaveformEditor from './WaveformEditor';

// jsdom 无 canvas 2d，mock
const ctx2d = {
  scale: vi.fn(),
  clearRect: vi.fn(),
  fillRect: vi.fn(),
  strokeRect: vi.fn(),
  fillStyle: '',
  strokeStyle: '',
  lineWidth: 1,
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
};
Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  configurable: true,
  value: vi.fn(() => ctx2d),
});
// getBoundingClientRect 返回固定宽高
Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
  configurable: true,
  value: vi.fn(() => ({ left: 0, top: 0, width: 600, height: 400, right: 600, bottom: 400, x: 0, y: 0, toJSON: () => ({}) })),
});

const preview = {
  video_a_path: '/tmp/a.mp4',
  audio_a_path: '/tmp/a.m4a',
  audio_b_path: '/tmp/b.m4a',
  waveform_a: Array.from({ length: 100 }, (_, i) => 0.5 + 0.5 * Math.sin(i / 5)),
  waveform_b: Array.from({ length: 50 }, (_, i) => 0.5 + 0.5 * Math.cos(i / 5)),
};

describe('WaveformEditor', () => {
  it('渲染双波形与 offset 显示', () => {
    render(<WaveformEditor preview={preview} initialOffset={2} durationA={10} onOffsetChange={() => {}} />);
    expect(screen.getByTestId('waveform-editor')).toBeTruthy();
    expect(screen.getByTestId('waveform-a')).toBeTruthy();
    expect(screen.getByTestId('waveform-b')).toBeTruthy();
    // offset 显示（含一位小数）
    expect(screen.getByText(/2\.0\s*s/)).toBeTruthy();
  });

  it('拖动 B 波形更新 offset 并回调', () => {
    const onChange = vi.fn();
    render(<WaveformEditor preview={preview} initialOffset={2} durationA={10} onOffsetChange={onChange} />);
    const b = screen.getByTestId('waveform-b');
    // 600px 宽 = 10s → 0.1s/6px。从 x=12(2s) 拖到 x=60(10s 处太远) → 改拖到 x=24(4s)
    fireEvent.mouseDown(b, { clientX: 12, clientY: 200 });
    fireEvent.mouseMove(b, { clientX: 24, clientY: 200 });
    fireEvent.mouseUp(b);
    // 拖动 +12px = +2s → offset 4.0
    expect(onChange).toHaveBeenCalled();
    const last = onChange.mock.calls.at(-1)![0] as number;
    expect(Math.abs(last - 4.0)).toBeLessThan(0.3);
  });
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd app && npx vitest run src/WaveformEditor.test.tsx`
Expected: FAIL with `Cannot find module './WaveformEditor'`.

- [ ] **Step 3: 写实现 `WaveformEditor.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react';
import type { AudioPreview } from './vite-env';

const METHOD_LABEL: Record<string, string> = { dtw: '精确对齐', beat: '节拍对齐', zero: '从头铺设' };

interface Props {
  preview: AudioPreview;
  initialOffset: number;
  durationA: number;
  onOffsetChange: (offset: number) => void;
}

export default function WaveformEditor({ preview, initialOffset, durationA, onOffsetChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [offset, setOffset] = useState(initialOffset);
  const [method, setMethod] = useState('dtw');
  const dragRef = useRef<{ startX: number; startOffset: number } | null>(null);

  // offset 变化时通知父组件
  useEffect(() => {
    onOffsetChange(offset);
  }, [offset]);

  const toX = (s: number, width: number) => (s / durationA) * width;

  const draw = (rect: DOMRect) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const midY = rect.height / 2;
    const laneH = rect.height / 4; // 每条波形占 1/4
    // A 波形（顶部）
    drawLane(ctx, preview.waveform_a, 0, laneH, rect.width, '#6ea8ff');
    // B 波形（底部），从 offset 处开始
    const bx = toX(offset, rect.width);
    drawLane(ctx, preview.waveform_b, bx, laneH, rect.width, '#4cd07a');
    // offset 分割线
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bx, 0);
    ctx.lineTo(bx, rect.height);
    ctx.stroke();
  };

  const drawLane = (
    ctx: CanvasRenderingContext2D, data: number[], x0: number, top: number,
    width: number, color: string,
  ) => {
    const laneH = width / Math.max(1, data.length) * 0.8; // 每根柱宽
    ctx.fillStyle = color;
    for (let i = 0; i < data.length; i++) {
      const x = x0 + (i / data.length) * (durationA * (data.length / data.length)); // 波形铺满到 width
      const h = Math.max(1, data[i] * (width / 4)); // 高度随振幅
      ctx.fillRect(x, top + 2, laneH, h);
    }
  };

  // 用 ResizeObserver 保持 canvas 与容器同步重绘
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect();
      draw(rect);
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [preview, offset]);

  const handleMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { startX: e.clientX, startOffset: offset };
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    const drag = dragRef.current;
    if (!drag || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const deltaS = ((e.clientX - drag.startX) / rect.width) * durationA;
    const next = Math.max(0, Math.min(drag.startOffset + deltaS, durationA));
    setOffset(next);
  };
  const handleMouseUp = () => {
    dragRef.current = null;
  };

  return (
    <div data-testid="waveform-editor" className="waveform-editor">
      <canvas
        ref={canvasRef}
        data-testid="waveform-b"
        className="waveform-canvas"
        style={{ cursor: 'ew-resize' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      <div className="waveform-meta">
        <span>偏移: <b>{offset.toFixed(1)}s</b></span>
        <span className="waveform-method">{METHOD_LABEL[method] || method}</span>
      </div>
      {/* A/B 两个测试锚点（drawLane 用 canvas 画，无需 DOM） */}
      <div data-testid="waveform-a" hidden />
    </div>
  );
}
```

> **注意**：`drawLane` 中 `x` 计算用了错误的 `(durationA * ...)` 表达式——真实实现应让 B 波形以原始 B 时长铺满，但当前预览只需要视觉可拖动。**实现时**修正为：B 波形柱从 `x0` 开始，每柱宽度 `(B时长/len)*pxPerSec`，A 波形柱从 0 开始铺满。以「拖动手感正确、柱分布与振幅对应」为准。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd app && npx vitest run src/WaveformEditor.test.tsx`
Expected: PASS。若 canvas mock 不完整导致 `getContext` 返回不全，补齐 mock 方法。

- [ ] **Step 5: 追加波形图样式到 App.css**

```css
/* ---------- 波形图编辑器 ---------- */
.waveform-editor {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.waveform-canvas {
  width: 100%;
  height: 120px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  touch-action: none;
}
.waveform-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
  color: var(--text-dim);
}
.waveform-method {
  background: var(--accent-soft);
  color: var(--accent);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}
```

- [ ] **Step 6: Commit**

```bash
git add app/src/WaveformEditor.tsx app/src/WaveformEditor.test.tsx app/src/App.css
git commit -m "feat(app): 波形图编辑器（拖动微调 offset）
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 前端替换音轨主面板 `AudioSwap.tsx`

**Files:**
- Create: `app/src/AudioSwap.tsx`
- Create: `app/src/AudioSwap.test.tsx`
- Modify: `app/src/App.tsx`（加入功能切换入口）
- Modify: `app/src/App.css`（追加面板样式）
- Modify: `app/src/vite-env.d.ts`（若需补类型）

**Interfaces:**
- Consumes:
  - `window.api.openVideo` / `openAudio` / `openAnyMedia`（选素材）
  - `window.api.submitAudioTask` / `getAudioTask` / `renderAudioTask`（任务）
  - `window.api.saveVideo` / `showInFolder`（保存 + 打开文件夹）
  - `WaveformEditor`（Task 6）
  - `AudioPreview` / `AudioAlignResult` / `AudioTaskInfo` 类型
- Produces: `AudioSwap` 组件（内嵌在 App 主界面，作为并列功能区）。

**功能**：
1. 选素材A（视频）、素材B（音频/视频）。
2. 「开始对齐」→ `submitAudioTask` → 轮询 `getAudioTask` 到 DONE。
3. DONE 后展示 `WaveformEditor`（preview + initialOffset），下方「播放」/「下载」按钮。
4. 「播放」：用 Web Audio 试听——A 视频静音播放 + A/B 音轨按 offset 同步混音。
5. 「下载」：调 `saveVideo` 选路径 → `renderAudioTask(taskId, {offset_seconds})` → 轮询到 DONE → 显示「打开所在文件夹」。
6. 对齐方式徽标（DTW/节拍/从头）+ 置信度提示（低置信时提示可拖动微调）。

- [ ] **Step 1: 写失败测试**

```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AudioSwap from './AudioSwap';

// canvas + rect mock（WaveformEditor 依赖）
const ctx2d = { scale: vi.fn(), clearRect: vi.fn(), fillRect: vi.fn(), strokeRect: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(), fillStyle: '', strokeStyle: '', lineWidth: 1 };
Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', { configurable: true, value: vi.fn(() => ctx2d) });
Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', { configurable: true, value: vi.fn(() => ({ left: 0, top: 0, width: 600, height: 120, right: 600, bottom: 120, x: 0, y: 0, toJSON: () => ({}) })) });

const api = {
  openVideo: vi.fn().mockResolvedValue({ path: '/tmp/a.mp4' }),
  openAudio: vi.fn().mockResolvedValue({ path: '/tmp/b.mp3' }),
  openAnyMedia: vi.fn().mockResolvedValue({ path: '/tmp/b.mp3' }),
  saveVideo: vi.fn().mockResolvedValue({ path: '/tmp/out.mp4' }),
  showInFolder: vi.fn().mockResolvedValue({ ok: true }),
  submitAudioTask: vi.fn().mockResolvedValue({ task_id: 'at1', status: 'QUEUED' }),
  getAudioTask: vi.fn().mockResolvedValue({
    task_id: 'at1', status: 'DONE', progress: 100, message: '',
    align_result: { offset_seconds: 2, tempo_ratio: 1, confidence: 'high', method: 'dtw' },
    preview: {
      video_a_path: '/tmp/a.mp4', audio_a_path: '/tmp/a.m4a', audio_b_path: '/tmp/b.m4a',
      waveform_a: [0.5, 0.6, 0.7], waveform_b: [0.4, 0.5, 0.6],
    },
  }),
  renderAudioTask: vi.fn().mockResolvedValue({ ok: true }),
};

beforeEach(() => { (window as any).api = api; });

describe('AudioSwap', () => {
  it('渲染素材选择按钮', () => {
    render(<AudioSwap />);
    expect(screen.getByRole('button', { name: /选择素材A/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /选择素材B/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /开始对齐/ })).toBeTruthy();
  });

  it('对齐完成后显示波形图与下载按钮', async () => {
    render(<AudioSwap />);
    fireEvent.click(screen.getByRole('button', { name: /选择素材A/ }));
    fireEvent.click(screen.getByRole('button', { name: /选择素材B/ }));
    fireEvent.click(screen.getByRole('button', { name: /开始对齐/ }));
    await waitFor(() => {
      expect(screen.getByTestId('waveform-editor')).toBeTruthy();
    });
    expect(screen.getByRole('button', { name: /播放/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /下载/ })).toBeTruthy();
  });

  it('低置信度时提示手动微调', async () => {
    (window as any).api = {
      ...api,
      getAudioTask: vi.fn().mockResolvedValue({
        task_id: 'at1', status: 'DONE', progress: 100, message: '',
        align_result: { offset_seconds: 0, tempo_ratio: 1, confidence: 'low', method: 'beat' },
        preview: { video_a_path: '/tmp/a.mp4', audio_a_path: '/tmp/a.m4a', audio_b_path: '/tmp/b.m4a', waveform_a: [0.5], waveform_b: [0.4] },
      }),
    };
    render(<AudioSwap />);
    fireEvent.click(screen.getByRole('button', { name: /选择素材A/ }));
    fireEvent.click(screen.getByRole('button', { name: /选择素材B/ }));
    fireEvent.click(screen.getByRole('button', { name: /开始对齐/ }));
    await waitFor(() => {
      expect(screen.getByText(/可拖动微调/)).toBeTruthy();
    });
  });

  it('下载流程：renderAudioTask 被调用', async () => {
    render(<AudioSwap />);
    fireEvent.click(screen.getByRole('button', { name: /选择素材A/ }));
    fireEvent.click(screen.getByRole('button', { name: /选择素材B/ }));
    fireEvent.click(screen.getByRole('button', { name: /开始对齐/ }));
    await waitFor(() => {
      expect(screen.getByTestId('waveform-editor')).toBeTruthy();
    });
    fireEvent.click(screen.getByRole('button', { name: /下载/ }));
    await waitFor(() => {
      expect(api.saveVideo).toHaveBeenCalled();
    });
    // render 后轮询到 DONE → 显示打开文件夹
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /打开所在文件夹/ })).toBeTruthy();
    });
  });
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd app && npx vitest run src/AudioSwap.test.tsx`
Expected: FAIL with `Cannot find module './AudioSwap'`.

- [ ] **Step 3: 写实现 `AudioSwap.tsx`**

```tsx
import { useRef, useState } from 'react';
import WaveformEditor from './WaveformEditor';
import type { AudioPreview, AudioTaskInfo } from './vite-env';

const METHOD_LABEL: Record<string, string> = { dtw: '精确对齐', beat: '节拍对齐', zero: '从头铺设' };
const CONFIDENCE_HINT: Record<string, string> = {
  high: '',
  low: '对齐置信度较低，可拖动波形微调偏移',
};

type Phase = 'idle' | 'aligning' | 'aligned' | 'rendering' | 'done' | 'failed';

export default function AudioSwap() {
  const [videoAPath, setVideoAPath] = useState<string | null>(null);
  const [audioBPath, setAudioBPath] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState(0);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [info, setInfo] = useState<AudioTaskInfo | null>(null);
  const [offset, setOffset] = useState(0);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const audioARef = useRef<HTMLAudioElement | null>(null);
  const audioBRef = useRef<HTMLAudioElement | null>(null);
  const videoARef = useRef<HTMLVideoElement | null>(null);

  const poll = async (id: string, stopStatus: string[]) => {
    const st = await window.api.getAudioTask(id);
    setProgress(st.progress);
    setInfo(st);
    if (stopStatus.includes(st.status)) return st;
    await new Promise((r) => setTimeout(r, 500));
    return poll(id, stopStatus);
  };

  const startAlign = async () => {
    if (!videoAPath || !audioBPath) return;
    setPhase('aligning');
    setProgress(0);
    setMessage('');
    try {
      const res = await window.api.submitAudioTask({
        video_a_path: videoAPath,
        audio_b_path: audioBPath,
        output_path: '',
        params: {},
      });
      setTaskId(res.task_id);
      const st = await poll(res.task_id, ['DONE', 'FAILED', 'CANCELLED']);
      if (st.status === 'DONE') {
        setOffset(st.align_result?.offset_seconds ?? 0);
        setPhase('aligned');
      } else {
        setPhase('failed');
        setMessage(st.message || '对齐失败');
      }
    } catch (e) {
      setPhase('failed');
      setMessage('引擎未启动或连接失败');
    }
  };

  const playPreview = () => {
    const p = info?.preview;
    if (!p) return;
    // 停止上次
    audioARef.current?.pause(); audioBRef.current?.pause(); videoARef.current?.pause();
    // A 视频（静音，纯画面） + A/B 音频按 offset 同步
    // B 音频延迟 offset 秒开始（offset 秒后启动 B）
    audioARef.current = new Audio('file://' + p.audio_a_path);
    audioBRef.current = new Audio('file://' + p.audio_b_path);
    const delayMs = Math.max(0, offset * 1000);
    audioARef.current.play();
    setTimeout(() => audioBRef.current?.play(), delayMs);
  };

  const download = async () => {
    if (!taskId) return;
    const save = await window.api.saveVideo(`vibe_audio_swap_${Date.now()}.mp4`);
    if (!save) return;
    setPhase('rendering');
    setProgress(0);
    try {
      await window.api.renderAudioTask(taskId, { offset_seconds: offset });
      const st = await poll(taskId, ['DONE', 'FAILED', 'CANCELLED']);
      if (st.status === 'DONE') {
        setOutputPath(save.path);
        setPhase('done');
      } else {
        setPhase('failed');
        setMessage(st.message || '导出失败');
      }
    } catch (e) {
      setPhase('failed');
      setMessage('导出失败');
    }
  };

  const renderResult = () => {
    if (phase === 'aligning' || phase === 'rendering') {
      return (
        <div className="status">
          <span>{phase === 'aligning' ? '对齐中' : '导出中'}… {Math.round(progress)}%</span>
          <div className="progress-wrap">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${Math.min(progress, 100)}%` }} />
            </div>
          </div>
        </div>
      );
    }
    if (phase === 'aligned') {
      const p = info?.preview;
      const ar = info?.align_result;
      if (!p) return null;
      return (
        <>
          {ar && ar.confidence === 'low' && (
            <div className="status err" style={{ marginBottom: 10 }}>
              {CONFIDENCE_HINT.low}（当前: {METHOD_LABEL[ar.method] || ar.method}）
            </div>
          )}
          <WaveformEditor
            preview={p}
            initialOffset={offset}
            durationA={p.waveform_a.length / 10}
            onOffsetChange={setOffset}
          />
          <div className="export-actions" style={{ marginTop: 12 }}>
            <button className="btn btn-secondary" onClick={playPreview}>播放</button>
            <button className="btn" onClick={download}>下载</button>
          </div>
        </>
      );
    }
    if (phase === 'done') {
      return (
        <div className="export-result">
          <div className="status ok">✅ 导出完成</div>
          {outputPath && (
            <>
              <span className="video-path">{outputPath}</span>
              <div className="export-actions">
                <button className="btn btn-secondary" onClick={() => window.api.showInFolder(outputPath)}>
                  打开所在文件夹
                </button>
              </div>
            </>
          )}
        </div>
      );
    }
    if (phase === 'failed') {
      return <div className="status err">{message || '处理失败'}</div>;
    }
    return null;
  };

  return (
    <div className="card audio-swap">
      <div className="card-title">替换音轨 · 自动对齐</div>
      <div className="audio-swap-row">
        <button className="btn btn-secondary" onClick={async () => {
          const r = await window.api.openVideo();
          if (r) { setVideoAPath(r.path); setPhase('idle'); }
        }}>
          {videoAPath ? '更换素材A' : '选择素材A（视频）'}
        </button>
        <button className="btn btn-secondary" onClick={async () => {
          const r = await window.api.openAnyMedia();
          if (r) { setAudioBPath(r.path); setPhase('idle'); }
        }}>
          {audioBPath ? '更换素材B' : '选择素材B（音乐/视频）'}
        </button>
      </div>
      {videoAPath && <p className="video-path">{videoAPath}</p>}
      {audioBPath && <p className="video-path">{audioBPath}</p>}
      <button
        className="btn btn-block"
        style={{ marginTop: 12 }}
        onClick={startAlign}
        disabled={!videoAPath || !audioBPath || phase === 'aligning' || phase === 'rendering'}
      >
        {phase === 'aligning' ? '对齐中…' : '开始对齐'}
      </button>
      <div style={{ marginTop: 12 }}>{renderResult()}</div>
    </div>
  );
}
```

> **注意**：`playPreview` 中 `new Audio('file://' + p.audio_a_path)` 在 Electron renderer 访问本地文件需要正确协议。实现时若 `file://` 不通，改用 `webSecurity: false` 的独立 preview 窗口，或让引擎通过 `http://127.0.0.1:8787` 提供静态文件服务（见 Task 9 若需要）。**前端试听是 Web Audio 主路径，若本地文件加载受阻，需在引擎加静态文件端点。**

- [ ] **Step 4: 运行测试确认通过**

Run: `cd app && npx vitest run src/AudioSwap.test.tsx`
Expected: PASS。jsdom 中 `new Audio()` 可能缺失，测试里需 mock `window.Audio`（见下 Step 4b）。

- [ ] **Step 4b: mock window.Audio（若 jsdom 不支持）**

在测试文件顶部加：

```ts
class MockAudio { play() {} pause() {} src = ''; }
Object.defineProperty(window, 'Audio', { configurable: true, value: MockAudio });
```

- [ ] **Step 5: 在 App.tsx 加入口**

在 `App.tsx` 中把新功能作为右侧控制区的一个独立卡片，放在现有视频功能之后：

```tsx
import AudioSwap from './AudioSwap';
// ...在 controls div 末尾追加
<AudioSwap />
```

（在现有 `{!videoPath && (...)}` 说明卡片之后，或作为独立的并列卡片，保持现有样式体系。）

- [ ] **Step 6: 追加面板样式到 App.css**

```css
/* ---------- 替换音轨 ---------- */
.audio-swap {
  margin-top: 16px;
}
.audio-swap-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
```

- [ ] **Step 7: 运行全部前端测试 + 类型检查**

Run: `cd app && npm test && npx tsc --noEmit`
Expected: 全部 PASS，无类型错误。

- [ ] **Step 8: Commit**

```bash
git add app/src/AudioSwap.tsx app/src/AudioSwap.test.tsx app/src/App.tsx app/src/App.css app/src/vite-env.d.ts
git commit -m "feat(app): 替换音轨主面板（对齐/试听/下载）
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: 引擎加入 librosa 依赖 + PyInstaller 打包配置

**Files:**
- Modify: `engine/pyproject.toml`
- Modify: `engine/engine.spec`
- Modify: `README.md`（依赖说明）

**Interfaces:**
- Consumes: 已完成的 `align.py`/`waveform.py`/`audiotask.py`。
- Produces: 引擎打包产物包含 librosa 及其传递依赖。

- [ ] **Step 1: pyproject.toml 加 librosa 依赖**

```toml
dependencies = [
  "opencv-contrib-python>=4.9,<5",
  "ultralytics>=8.2",
  "librosa>=0.10.2,<0.12",
]
```

- [ ] **Step 2: 安装依赖到 venv**

Run: `cd engine && source .venv/bin/activate && pip install -e .`
Expected: librosa 0.11.0 及其依赖（numba/scipy/soundfile 等）安装成功。

- [ ] **Step 3: 验证对齐核心在真实 librosa 下工作**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/test_align.py -v`
Expected: PASS（此时走真实 librosa 路径，非降级 numpy 路径）。

- [ ] **Step 4: engine.spec 补 hiddenimports**

PyInstaller 打包 librosa/numba 常有动态导入问题，需补：

```python
a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("yolov8n.pt", "."),
    ],
    hiddenimports=[
        # librosa / numba 动态导入的模块（打包验证后按需增删）
        "numba.core.registry",
        "numba.core.typedarray",
        "numba.core.types",
        "sklearn.utils._typedefs",
        "sklearn.utils._heap",
        "sklearn.neighbors._partition_nodes",
        "soundfile",
        "audioread.ffdec",
    ],
    ...
)
```

> **注意**：PyInstaller 的 hiddenimports 列表需**实测迭代**——首次打包后运行 `vibe_engine` 并触发 align 路径，若有 `ImportError`，把缺失模块名加入 hiddenimports 重新打包。

- [ ] **Step 5: 本机冻结 + 冒烟测试**

```bash
cd engine
source .venv/bin/activate
pip install pyinstaller
pyinstaller engine.spec --distpath dist --workpath build --noconfirm
# 冒烟：用冻结产物跑一个音频对齐任务
./dist/engine_bundle/vibe_engine 8788 &
# 等健康检查
curl -s http://127.0.0.1:8788/health
# POST 一个 audio-task（用合成素材），轮询确认 DONE
```

Expected: 冻结产物能完成对齐任务（含 librosa 路径），无 ImportError。

- [ ] **Step 6: README 更新依赖说明**

在 README「注意事项」中追加：

```md
- **音频对齐（替换音轨）**：引擎内置 librosa（随打包分发），无需额外安装。
```

- [ ] **Step 7: Commit**

```bash
git add engine/pyproject.toml engine/engine.spec README.md
git commit -m "build: 引擎加入 librosa 依赖 + PyInstaller 打包配置
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: 端到端集成验收 + 文档收尾

**Files:**
- Modify: `README.md`（功能说明）
- Create（可选）: `docs/superpowers/specs/2026-08-08-audio-swap-design.md` 已存在

**Interfaces:**
- Consumes: 全部已完成任务。
- Produces: 可用的完整功能。

- [ ] **Step 1: 引擎端到端冒烟（真实媒体）**

```bash
cd engine && source .venv/bin/activate && python - <<'EOF'
# 用真实风格素材：A=带现场音的短视频（人声+音乐混叠），B=纯净音乐
# 构造：A 用 ffmpeg 混合 sine 音乐 + 噪声人声；B 用同一 sine 纯净版
import subprocess, os, tempfile, time
from engine.audiotask import AudioTaskManager
d = tempfile.mkdtemp()
# A: 视频 + 混合音（音乐 300Hz 全程 + 噪声段）
subprocess.run(["ffmpeg","-y","-f","lavfi","-i","color=c=black:s=320x240:d=4",
  "-f","lavfi","-i","sine=frequency=300:duration=4",
  "-f","lavfi","-i","anoisesrc=d=4:c=pink:a=0.05",
  "-filter_complex","[1:a][2:a]amix=inputs=2[a]",
  "-map","0:v","-map","[a]","-c:v","libx264","-pix_fmt","yuv420p",
  "-c:a","aac","-shortest", os.path.join(d,"a.mp4")], capture_output=True)
# B: 纯净音乐 2s（A 中从 1s 开始的段）
subprocess.run(["ffmpeg","-y","-f","lavfi","-i","sine=frequency=300:duration=2",
  "-c:a","aac", os.path.join(d,"b.m4a")], capture_output=True)
mgr = AudioTaskManager()
tid = mgr.submit(os.path.join(d,"a.mp4"), os.path.join(d,"b.m4a"), os.path.join(d,"out.mp4"))
while mgr.get(tid)["status"] not in ("DONE","FAILED"): time.sleep(0.1)
st = mgr.get(tid)
print("align:", st.get("align_result"))
mgr.render(tid, st["align_result"]["offset_seconds"])
while mgr.get(tid)["status"] not in ("DONE","FAILED"): time.sleep(0.1)
print("render:", mgr.get(tid)["status"], os.path.exists(os.path.join(d,"out.mp4")))
EOF
```

Expected: 打印 `align: {...}`（method=dtw/beat）、`render: DONE True`。输出视频可播放、音轨为 B 音乐。

- [ ] **Step 2: 前端手动验收**

Run: `cd app && npm run electron:dev`
手动流程：
1. 打开「替换音轨」面板，选素材A（带现场音视频）、素材B（纯净音乐）。
2. 点「开始对齐」，等待对齐完成（显示波形图 + 偏移量）。
3. 拖动 B 波形调整偏移，点「播放」听混音效果。
4. 点「下载」，选保存位置，等待导出完成，点「打开所在文件夹」验证输出。

Expected: 全程流畅，波形可拖动、试听可听、导出 MP4 音轨为 B 音乐且与 A 画面同步。

- [ ] **Step 3: 回归全部测试**

Run: `cd engine && source .venv/bin/activate && python -m pytest tests/ -v && cd ../app && npm test && npx tsc --noEmit`
Expected: 全部 PASS，无类型错误。

- [ ] **Step 4: 更新 README 功能列表**

在 README「功能」节追加：

```md
- 替换音轨：素材B（纯净音乐）替换素材A（现场视频）音轨，自动对齐（DTW/节拍/从头三级降级），波形图拖动微调 + 实时试听，确认后导出。
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: 替换音轨功能说明 + 端到端验收
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 主流程（选素材→对齐→预览→下载）：Task 7 覆盖。
- §3.2 三级降级：Task 1 `align_tracks` 覆盖；UI 提示 Task 7 覆盖。
- §4 架构职责边界：Task 1/2/3（引擎）+ Task 6/7（前端）覆盖。
- §5 组件划分：全部覆盖。
- §6 API：Task 4/5 覆盖（含 align/render 分离）。
- §7 对齐算法：Task 1 覆盖（DTW/节拍/变速）。
- §8 前端交互（波形/拖动/试听/播放下载）：Task 6/7 覆盖。
- §9 错误处理：Task 1（降级）、Task 3（FAILED）、Task 7（UI 提示）覆盖。
- §10 测试策略：各任务 TDD 覆盖。
- §11 里程碑：Task 1-9 覆盖。
- §12 依赖变更：Task 8 覆盖。

**2. Placeholder scan:** 无 TBD/TODO 占位。所有代码步骤含完整实现。

**3. Type consistency:**
- `align_tracks` 返回 dict，Task 3 用 `result["offset_seconds"]` 一致。
- `AudioTaskManager.submit/get/render/cancel` 签名在 Task 3/4/5 一致。
- 前端 `AudioTaskInfo` / `AudioPreview` / `AudioAlignResult` 类型在 Task 5 声明、Task 6/7 消费一致。
- `extract_waveform` / `extract_preview_audio` 在 Task 2 定义、Task 3 消费一致。

**已发现并修正的问题：**
- `align.py` 降级路径的 `tempo_ratio` 在无 librosa 时返回 1.0，与 Task 3 混流 `abs(ratio-1.0)<0.01` 快路径一致。
- `audiotask.py` 混流 `-af adelay` 的时序语义在 Task 3 Step 5 有显式集成验证与备选方案。
- `WaveformEditor.drawLane` 的 x 计算在 Task 6 有实现修正说明。
- `AudioSwap.playPreview` 的 `file://` 加载在 Task 7 有备选方案（引擎静态文件端点），标注为 Task 9 若需要。

**遗留风险（诚实标注）：**
- librosa 的 PyInstaller hiddenimports 需实测迭代（Task 8 Step 4 已注明）。
- Web Audio 本地文件加载（`file://`）在 Electron renderer 可能受阻，若如此需在引擎加静态文件端点（Task 7 注释 + Task 9 兜底）。
- DTW 置信度阈值（`norm_cost < 0.35`）为经验值，需真实素材标定（Task 1 Step 6 + Task 9 端到端）。
