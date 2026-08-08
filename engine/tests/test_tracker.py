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


def test_redetect_prefers_trajectory_prediction(tmp_path):
    """CSRT 漂到他人身上后，重定位应基于轨迹预测回到原目标，而非固化跳变。

    场景：目标 A 持续左移（历史中心 (125,140)->(105,140)->(85,140)），
    遮挡后 CSRT 框漂到另一人 B（(180,100,50,80)）身上。
    修复前 _redetect 按 IoU 匹配选中 B（固化跳变）；
    修复后应按轨迹预测位置选中 A。
    """
    import types

    tracker = PersonTracker.__new__(PersonTracker)
    tracker._model = None
    tracker._size_tolerance = 0.5
    tracker._init_bbox = (50, 100, 50, 80)  # 目标基准尺寸
    tracker._last_box = (180, 100, 50, 80)  # CSRT 已漂到 B
    tracker._history_centers = [(125, 140), (105, 140), (85, 140)]  # 轨迹显示 A 在左移

    # YOLO 检测到 A 和 B 两个人
    class FakeDet:
        def __init__(self, b):
            x, y, w, h = b
            self.cls = 0
            self.xyxy = [(x, y, x + w, y + h)]

    class FakeResult:
        boxes = None

    class FakeModel:
        def __call__(self, frame, verbose=False):
            r = FakeResult()
            r.boxes = [FakeDet(b) for b in [(50, 100, 50, 80), (180, 100, 50, 80)]]
            return [r]

    tracker._model = FakeModel()

    person_a = (50, 100, 50, 80)
    chosen = tracker._redetect(None)
    assert chosen == person_a, f"应回到原目标 A，实际选中 {chosen}"


def test_redetect_falls_back_to_iou_without_history(tmp_path):
    """无历史轨迹时（如刚开始跟踪），重定位退化为 IoU 匹配 + 最近中心。"""
    tracker = PersonTracker.__new__(PersonTracker)
    tracker._model = None
    tracker._size_tolerance = 0.5
    tracker._init_bbox = (100, 100, 50, 80)
    tracker._last_box = (100, 100, 50, 80)
    tracker._history_centers = []  # 无历史

    class FakeDet:
        def __init__(self, b):
            x, y, w, h = b
            self.cls = 0
            self.xyxy = [(x, y, x + w, y + h)]

    class FakeResult:
        boxes = None

    class FakeModel:
        def __call__(self, frame, verbose=False):
            r = FakeResult()
            r.boxes = [FakeDet(b) for b in [(100, 100, 50, 80)]]
            return [r]

    tracker._model = FakeModel()
    chosen = tracker._redetect(None)
    assert chosen == (100, 100, 50, 80)

