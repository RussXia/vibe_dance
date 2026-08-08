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
