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
