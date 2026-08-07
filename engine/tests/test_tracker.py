from engine.tracker import PersonTracker
from engine.video import VideoReader
from .fixtures import make_synthetic_video


def _build_tracker(tmp_path, frames=20, init_frame=0):
    path = str(tmp_path / "track.mp4")
    make_synthetic_video(path, frames=frames)
    reader = VideoReader(path)
    # 合成视频里移动方块从 (10,10) 出发，每帧右移 5px
    bbox = (10, 10, 50, 80)
    return PersonTracker(reader, init_frame, bbox), reader


def test_track_returns_one_box_per_frame(tmp_path):
    tracker, reader = _build_tracker(tmp_path, frames=20)
    try:
        boxes = tracker.track()
        assert len(boxes) == 20
        for box in boxes:
            assert box is not None
            x, y, w, h = box
            assert w > 0 and h > 0
    finally:
        reader.release()


def test_track_starts_at_init_box(tmp_path):
    tracker, reader = _build_tracker(tmp_path, frames=20, init_frame=0)
    try:
        boxes = tracker.track()
        # 第一帧输出应接近初始框（允许 tracker 初始化偏移）
        x0, y0, w0, h0 = boxes[0]
        assert abs(x0 - 10) < 20 and abs(y0 - 10) < 20
    finally:
        reader.release()


def test_track_respects_max_frames(tmp_path):
    tracker, reader = _build_tracker(tmp_path, frames=20)
    try:
        boxes = tracker.track(max_frames=5)
        assert len(boxes) == 5
    finally:
        reader.release()


def test_lose_threshold_abandons_after_consecutive_loss(tmp_path, monkeypatch):
    """lose_threshold=3 时，连续 3 帧丢失后 abandon，后续全返回 None。"""
    import cv2

    path = str(tmp_path / "track.mp4")
    make_synthetic_video(path, frames=20)
    reader = VideoReader(path)

    class FakeTracker:
        def init(self, frame, bbox):
            return True

        def update(self, frame):
            return False, None

    monkeypatch.setattr(cv2, "TrackerCSRT_create", lambda: FakeTracker())
    try:
        tracker = PersonTracker(
            reader, 0, (10, 10, 50, 80), params={"lose_threshold": 3},
        )
        boxes = tracker.track(max_frames=10)
        # 帧0: init 成功 → 有框
        assert boxes[0] is not None
        # 帧1,2: 丢失但未达阈值 → None；帧3: 达阈值 abandon → 此后全 None
        assert boxes[1] is None and boxes[2] is None
        assert all(b is None for b in boxes[3:])
    finally:
        reader.release()


def test_track_reports_progress(tmp_path):
    """track() 按已处理帧数上报进度（0-100）。"""
    path = str(tmp_path / "track.mp4")
    make_synthetic_video(path, frames=20)
    reader = VideoReader(path)
    try:
        tracker = PersonTracker(reader, 0, (10, 10, 50, 80))
        progress = []
        tracker.track(on_progress=lambda p: progress.append(p))
        # 进度应从 0 走到 100，单调不减
        assert progress and progress[0] >= 0
        assert progress[-1] == 100
        assert all(progress[i] <= progress[i + 1] for i in range(len(progress) - 1))
    finally:
        reader.release()
