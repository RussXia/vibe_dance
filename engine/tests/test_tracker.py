"""PersonTracker（BoT-SORT + 位置锁定）测试。

新方案用 ultralytics BoT-SORT 逐帧检测，合成视频的方块不会被 YOLO
识别为 person，故测试用 FakeModel mock 检测结果，聚焦锁定/连续性/
丢失逻辑本身。
"""
import numpy as np

from engine.tracker import PersonTracker
from engine.video import VideoReader
from .fixtures import make_synthetic_video


def _make_fake_model(boxes_by_frame):
    """构造一个逐帧返回指定 person 检测框的假 YOLO 模型。

    boxes_by_frame: list，按帧顺序每帧的 person 框列表（可少于视频帧数）。
    """
    class FakeDet:
        def __init__(self, box):
            x, y, w, h = box
            self.cls = 0  # person
            self.xyxy = np.array([(x, y, x + w, y + h)])

    class FakeResult:
        def __init__(self, dets):
            # boxes 需支持 .id（None）和 .xyxy（FakeDet 提供）
            self.boxes = FakeBoxes(dets)

    class FakeBoxes:
        id = None
        def __init__(self, dets):
            self._dets = dets
            # xyxy 模拟 ultralytics tensor：Nx4
            self.xyxy = np.array(
                [d.xyxy[0] for d in dets], dtype=float,
            ) if dets else np.zeros((0, 4), dtype=float)
        def __iter__(self):
            return iter(self._dets)

    class FakeModel:
        def __init__(self, frames):
            self._frames = frames
            self._call = 0
        def track(self, frame, persist=True, tracker=None, verbose=False):
            fi = min(self._call, len(self._frames) - 1)
            self._call += 1
            dets = [FakeDet(b) for b in self._frames[fi]]
            return [FakeResult(dets)]

    return FakeModel(boxes_by_frame)


def _build_reader(tmp_path, frames=20):
    path = str(tmp_path / "track.mp4")
    make_synthetic_video(path, frames=frames)
    return VideoReader(path)


def test_track_follows_nearest_detection(tmp_path):
    """目标持续出现在同一位置时，跟踪应稳定输出该框。"""
    reader = _build_reader(tmp_path)
    frames = [[(100, 100, 50, 80)] for _ in range(20)]
    try:
        tracker = PersonTracker(reader, 0, (100, 100, 50, 80))
        tracker._model = _make_fake_model(frames)
        boxes = tracker.track()
        assert len(boxes) == 20
        for box in boxes:
            assert box == (100, 100, 50, 80)
    finally:
        reader.release()


def test_track_outputs_none_on_missing_detection(tmp_path):
    """无检测时应返回 None，且连续丢失超过阈值后 abandon。"""
    reader = _build_reader(tmp_path)
    # 前 2 帧有目标，之后全无
    frames = [[(100, 100, 50, 80)], [(100, 100, 50, 80)]] + [[] for _ in range(8)]
    try:
        tracker = PersonTracker(
            reader, 0, (100, 100, 50, 80), params={"lose_threshold": 3},
        )
        tracker._model = _make_fake_model(frames)
        boxes = tracker.track(max_frames=10)
        assert boxes[0] is not None
        assert boxes[1] is not None
        # 帧2,3,4 连续丢失达阈值 → abandon，后续全 None
        assert boxes[2] is None and boxes[3] is None
        assert all(b is None for b in boxes[4:])
    finally:
        reader.release()


def test_track_follows_moving_target(tmp_path):
    """目标每帧移动时，跟踪应跟随（选最近的框）。"""
    reader = _build_reader(tmp_path)
    # 目标从 (100,100) 每帧右移 5px
    frames = [[(100 + i * 5, 100, 50, 80)] for i in range(20)]
    try:
        tracker = PersonTracker(reader, 0, (100, 100, 50, 80))
        tracker._model = _make_fake_model(frames)
        boxes = tracker.track()
        # 每帧应跟随目标移动
        for i, box in enumerate(boxes):
            assert box[0] == 100 + i * 5
    finally:
        reader.release()


def test_track_reports_progress(tmp_path):
    """track() 按已处理帧数上报进度（0-100），单调不减。"""
    reader = _build_reader(tmp_path)
    frames = [[(100, 100, 50, 80)] for _ in range(20)]
    try:
        tracker = PersonTracker(reader, 0, (100, 100, 50, 80))
        tracker._model = _make_fake_model(frames)
        progress = []
        tracker.track(on_progress=lambda p: progress.append(p))
        assert progress and progress[0] >= 0
        assert progress[-1] == 100
        assert all(progress[i] <= progress[i + 1] for i in range(len(progress) - 1))
    finally:
        reader.release()
