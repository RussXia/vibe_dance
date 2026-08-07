import os

import cv2
import numpy as np

from engine.render import Renderer
from engine.tracker import PersonTracker
from engine.video import VideoReader
from .fixtures import add_audio_track, make_synthetic_video


def _make_scene(tmp_path, frames=20):
    """画面左侧固定一个静止人物方块，居中便于稳定裁剪。"""
    path = str(tmp_path / "scene.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    assert writer.isOpened()
    for _ in range(frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 60), (180, 200), (200, 200, 200), -1)  # 静止人物
        writer.write(frame)
    writer.release()
    return path


def _make_occluded_scene(tmp_path, frames=20):
    """目标方块在第 10–14 帧消失（全黑），验证丢帧仍输出全部帧。"""
    path = str(tmp_path / "occluded.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        if not (10 <= i <= 14):
            cv2.rectangle(frame, (100, 60), (180, 200), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def test_render_skipped_frames_still_outputs_all(tmp_path):
    """中间段目标消失（跟踪丢帧）时，输出帧数 = 输入帧数，而非被截断。"""
    path = _make_occluded_scene(tmp_path)
    reader = VideoReader(path)
    try:
        tracker = PersonTracker(reader, 0, (100, 60, 80, 140))
        viewport = (100, 60, 80, 142)
        out = str(tmp_path / "out.mp4")
        progress = []

        renderer = Renderer(reader, tracker, 0, viewport, (180, 320))
        renderer.render(out, on_progress=lambda p: progress.append(p))

        assert os.path.exists(out)
        cap = cv2.VideoCapture(out)
        assert cap.isOpened()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # 输出帧数 = 输入帧数：PNG 序列无断号，FFmpeg 不会在缺口处丢弃后续帧
        assert n == reader.frame_count == 20
        assert progress and progress[-1] == 100
    finally:
        reader.release()


def test_render_produces_output_video(tmp_path):
    path = _make_scene(tmp_path)
    reader = VideoReader(path)
    try:
        tracker = PersonTracker(reader, 0, (100, 60, 80, 140))
        # 9:16 取景框：宽 80，高 142 (≈80*16/9)
        viewport = (100, 60, 80, 142)
        out = str(tmp_path / "out.mp4")
        progress = []

        renderer = Renderer(reader, tracker, 0, viewport, (180, 320))
        renderer.render(out, on_progress=lambda p: progress.append(p))

        assert os.path.exists(out)
        cap = cv2.VideoCapture(out)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert (w, h) == (180, 320)
        assert n == 20
        assert progress and progress[-1] == 100
    finally:
        reader.release()


def test_render_preserves_audio_track(tmp_path):
    """渲染输出应保留原视频音轨。"""
    scene = _make_scene(tmp_path, frames=20)
    # 给合成视频加音轨（2 秒 440Hz 正弦波）
    audio_path = str(tmp_path / "tone.aac")
    video_with_audio = add_audio_track(scene, audio_path, 2)

    reader = VideoReader(video_with_audio)
    try:
        tracker = PersonTracker(reader, 0, (100, 60, 80, 140))
        viewport = (100, 60, 80, 142)
        out = str(tmp_path / "out.mp4")
        renderer = Renderer(reader, tracker, 0, viewport, (180, 320))
        renderer.render(out)

        # 用 ffprobe 检查输出是否有音频流
        import subprocess
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-select_streams", "a",
             "-of", "json", out],
            capture_output=True, text=True, check=True,
        )
        import json
        streams = json.loads(probe.stdout).get("streams", [])
        assert len(streams) >= 1, "输出视频缺少音轨"
        assert streams[0]["codec_type"] == "audio"
    finally:
        reader.release()
