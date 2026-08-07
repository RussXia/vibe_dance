import numpy as np
import pytest

from engine.video import VideoReader
from .fixtures import make_synthetic_video


def test_metadata(tmp_path):
    path = str(tmp_path / "test.mp4")
    make_synthetic_video(path, width=320, height=240, fps=10, frames=20)
    reader = VideoReader(path)
    try:
        assert reader.width == 320
        assert reader.height == 240
        assert reader.frame_count == 20
    finally:
        reader.release()


def test_read_random_frame(tmp_path):
    path = str(tmp_path / "test.mp4")
    make_synthetic_video(path)
    reader = VideoReader(path)
    try:
        frame = reader.read_frame(0)
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
    finally:
        reader.release()


def test_read_out_of_range_returns_none(tmp_path):
    path = str(tmp_path / "test.mp4")
    make_synthetic_video(path, frames=20)
    reader = VideoReader(path)
    try:
        assert reader.read_frame(999) is None
        assert reader.read_frame(-1) is None
    finally:
        reader.release()


def test_iter_frames_sequential(tmp_path):
    """顺序流式读取：从指定帧起逐帧返回，帧数正确。"""
    path = str(tmp_path / "test.mp4")
    make_synthetic_video(path, frames=20)
    reader = VideoReader(path)
    try:
        frames = list(reader.iter_frames())
        assert len(frames) == 20
        assert all(f is not None for f in frames)
        assert frames[0].shape == (240, 320, 3)
    finally:
        reader.release()


def test_iter_frames_from_start_index(tmp_path):
    """从指定帧开始顺序读取。"""
    path = str(tmp_path / "test.mp4")
    make_synthetic_video(path, frames=20)
    reader = VideoReader(path)
    try:
        frames = list(reader.iter_frames(start_index=15))
        assert len(frames) == 5  # 15..19
    finally:
        reader.release()
