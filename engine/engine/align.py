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
    """numpy-only 的 STFT + 梅尔三角滤波器（降级路径，测试/无 librosa 环境用）。

    返回 (n_mels, n_frames) 以匹配 librosa.feature.melspectrogram 的格式。
    """
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
    M = W @ S.T  # (n_mels, n_frames)
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
    except ImportError:
        # 无 librosa 时用 numpy 实现简化 DTW（允许子序列对齐）
        return _dtw_numpy(spec_a.T, spec_b.T, hop_length, sr, max_slope)


def _dtw_numpy(X: np.ndarray, Y: np.ndarray,
               hop_length: int, sr: int, max_slope: float) -> tuple[float, float, float]:
    """numpy 实现的简化对齐（无 librosa 的降级路径）。

    使用窗口化相关性找 Y 在 X 中的最佳对齐位置。

    Args:
        X: A 的频谱转置 (n_frames_a, n_mels)
        Y: B 的频谱转置 (n_frames_b, n_mels)

    Returns:
        (offset_seconds, tempo_ratio, cost)
        cost 是平均平方误差（方便与 DTW 成本比较）
    """
    n_a, n_mels = X.shape
    n_b, _ = Y.shape

    if n_b > n_a:
        # Y 比 X 还长，无法是子序列，返回零
        return 0.0, 1.0, float('inf')

    # 窗口化：滑动 Y 在 X 上，找最小距离窗口
    best_offset_frames = 0
    best_cost = float('inf')

    for start_frame in range(n_a - n_b + 1):
        window = X[start_frame:start_frame + n_b, :]  # (n_b, n_mels)
        # 平均平方误差（MSE）
        mse = np.mean((window - Y) ** 2)
        if mse < best_cost:
            best_cost = mse
            best_offset_frames = start_frame

    offset_seconds = best_offset_frames * hop_length / sr
    # 变速比：都是全长对齐，所以是 1.0
    tempo_ratio = 1.0

    return float(offset_seconds), float(tempo_ratio), float(best_cost)


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
