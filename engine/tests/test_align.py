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
    # A: 0-2s 300Hz (独特前缀), 2-5s 440Hz (B 匹配段)
    _make_audio(a, [(0, 300, 2), (2, 440, 3)])
    # B: 就是 A 从 2s 开始的 3s 片段 (440Hz)
    _make_audio(b, [(0, 440, 3)])
    res = align_tracks(a, b)
    assert res["method"] == "dtw", res
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
