"""目标跟踪器：CSRT/KCF 逐帧跟踪 + YOLO 定期重定位。"""
from __future__ import annotations

from pathlib import Path

import cv2

_DEFAULT_PARAMS = {
    "redetect_interval": 30,
    "tracker_type": "CSRT",
    "lose_threshold": 30,
    "track_scale": 0.4,  # 跟踪时把帧缩小的比例：CSRT 在 1080p 上 ~60ms/帧太慢，
                         # 缩到 0.4x 后 ~20ms/帧（3x 提速），坐标输出时换算回原分辨率。
}


def _model_path() -> Path:
    """返回随引擎分发的 YOLO 权重路径（优先本地，避免打包后联网下载）。

    候选位置：
    - 本文件同目录（engine/engine/yolov8n.pt）
    - 引擎包根（engine/yolov8n.pt，开发时 CWD 为 engine/ 也覆盖）
    都找不到时交由 ultralytics 按默认位置下载。
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / "yolov8n.pt", here.parent / "yolov8n.pt", Path("yolov8n.pt")):
        if candidate.exists():
            return candidate
    return Path("yolov8n.pt")  # 找不到本地权重时交由 ultralytics 下载


class PersonTracker:
    """对单个目标人物的跟踪器。

    - 帧间使用 OpenCV 单目标跟踪器（CSRT 或 KCF），在缩小后的帧上运行
      （track_scale，默认 0.4）以大幅提速；输出的 bbox 换算回原分辨率。
    - 每 redetect_interval 帧用 YOLOv8n 人体检测重定位，取与当前框 IoU 最高者。
    - 跟踪连续丢失超过 lose_threshold 帧时，后续帧返回 None（彻底丢失）。
    """

    def __init__(self, reader, init_frame, bbox, params=None):
        self._reader = reader
        self._init_frame = init_frame
        self._params = {**_DEFAULT_PARAMS, **(params or {})}
        self.tracker_type = self._params["tracker_type"]
        self._scale = float(self._params.get("track_scale", 0.4))
        self._init_bbox = tuple(int(v) for v in bbox)
        self._last_box = self._init_bbox
        self._tracker = None
        self._model = None
        self._lose_threshold = int(self._params.get("lose_threshold", 30))
        self._consecutive_loss = 0
        self._abandoned = False
        # 历史跟踪框中心（原始分辨率），用于轨迹外推预测目标当前位置，
        # 供重定位时区分"真正要跟踪的人"与"遮挡/交叉时临时靠近的其他人"。
        self._history_centers: list[tuple[float, float]] = []
        # 重定位候选筛选时，允许的尺寸偏离比例（相对目标历史尺寸）
        self._size_tolerance = float(self._params.get("size_tolerance", 0.5))

    def _create_tracker(self):
        if self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        return cv2.TrackerCSRT_create()

    def _scale_frame(self, frame):
        """把帧缩到 track_scale 比例，供跟踪/检测加速。"""
        if self._scale >= 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(frame, (max(1, int(w * self._scale)), max(1, int(h * self._scale))))

    def _scale_box(self, box):
        """原始分辨率 bbox → 缩小帧坐标。"""
        x, y, w, h = (float(v) for v in box)
        return (int(x * self._scale), int(y * self._scale),
                max(1, int(w * self._scale)), max(1, int(h * self._scale)))

    def _unscale_box(self, box):
        """缩小帧 bbox → 原始分辨率坐标（输出用）。"""
        x, y, w, h = (float(v) for v in box)
        return (int(x / self._scale), int(y / self._scale),
                int(w / self._scale), int(h / self._scale))

    def track(self, max_frames=None, on_progress=None):
        total = self._reader.frame_count - self._init_frame
        results = []
        # 预加载 YOLO 模型：首次 _redetect 才懒加载会导致进度卡在 0%
        # 直到第 redetect_interval 帧（约 1-5s，含 torch 初始化/权重加载/下载）。
        # 前置到这里，模型加载计入"准备"阶段（进度仍为 0 但更早完成）。
        self._ensure_model()
        for frame_index, frame in enumerate(
            self._reader.iter_frames(self._init_frame),
            start=self._init_frame,
        ):
            if max_frames is not None and len(results) >= max_frames:
                break
            # 跟踪在缩小帧上进行
            sframe = self._scale_frame(frame)
            box_out = None
            if not self._abandoned:
                if frame_index == self._init_frame:
                    sbox = self._scale_box(self._init_bbox)
                    ok, sbox_out = self._start_tracking(sframe, sbox)
                else:
                    ok, sbox_out = self._continue_tracking(sframe, frame_index)
                if not ok:
                    self._consecutive_loss += 1
                    if self._consecutive_loss >= self._lose_threshold:
                        self._abandoned = True
                else:
                    self._consecutive_loss = 0
                    box_out = self._unscale_box(sbox_out) if sbox_out else None
                    if box_out is not None:
                        self._push_history(self._center(box_out))
            results.append(box_out)
            # 进度上报：按已处理帧数占比（0-100），每帧都调（render 负责节流）。
            # 用浮点避免 int() 取整导致前 1% 帧显示 0（看起来卡在 0%）。
            if on_progress is not None and total > 0:
                pct = round(len(results) / total * 100, 1)
                on_progress(min(pct, 100.0))
        return results

    def _start_tracking(self, frame, bbox):
        self._tracker = self._create_tracker()
        # OpenCV 4.x contrib 绑定中 Tracker.init() 不返回 bool:成功返回 None,
        # 无效输入直接抛 C++ 异常。以「不抛异常」作为成功判据,与 C++ bool 语义一致。
        try:
            self._tracker.init(frame, tuple(bbox))
        except cv2.error:
            self._last_box = bbox
            return False, None
        self._last_box = bbox
        # 初始化帧直接返回初始框
        return True, tuple(int(v) for v in bbox)

    def _continue_tracking(self, frame, frame_index):
        ok, box = self._tracker.update(frame)
        if not ok:
            return False, None
        x, y, w, h = (int(v) for v in box)
        # 定期重定位（在缩小帧上做 YOLO 检测）
        if frame_index % self._params["redetect_interval"] == 0:
            new_box = self._redetect(frame)
            if new_box is not None:
                x, y, w, h = new_box
                try:
                    self._tracker.init(frame, (x, y, w, h))
                except cv2.error:
                    pass
        self._last_box = (x, y, w, h)
        return True, (x, y, w, h)

    def _ensure_model(self):
        """确保 YOLO 模型已加载（torch 初始化 + 权重加载 + 首次推理 warmup）。"""
        from ultralytics import YOLO

        if self._model is None:
            self._model = YOLO(str(_model_path()))

    def _push_history(self, center: tuple[float, float]):
        """把一帧有效跟踪框的中心推入历史轨迹，最多保留最近 10 个。"""
        self._history_centers.append(center)
        if len(self._history_centers) > 10:
            self._history_centers.pop(0)

    def _predict_center(self):
        """用历史中心线性外推预测目标当前位置。

        用最近两个历史中心的速度外推下一帧位置；历史不足 2 个时退化为
        最后一个历史中心（无速度信息，预测=当前位置）。
        """
        if not self._history_centers:
            return None
        if len(self._history_centers) == 1:
            return self._history_centers[-1]
        (x1, y1), (x2, y2) = self._history_centers[-2], self._history_centers[-1]
        vx, vy = x2 - x1, y2 - y1
        return (x2 + vx, y2 + vy)

    def _target_size(self):
        """目标历史尺寸估计：最近历史框宽高（取最后一个，原始分辨率）。"""
        # _history_centers 只存中心，尺寸从 _last_box 估。为更稳，用 _init_bbox 的宽高
        # 作为目标基准尺寸（人在跟踪过程中大小变化有限）。
        return self._init_bbox[2], self._init_bbox[3]

    def _redetect(self, frame):
        """YOLO 检测所有人，返回与目标轨迹最匹配的 person 框。

        优先按「离轨迹预测中心最近」选人，避免 CSRT 漂移被 IoU 匹配固化
        （遮挡/交叉时跟踪框跳到他人身上，IoU 匹配反而确认错误目标）。
        仅当轨迹预测不可用时（无历史），退化为 IoU 匹配 + 最近中心降级。
        """
        self._ensure_model()
        dets = self._model(frame, verbose=False)[0]
        persons = []
        for det in dets.boxes:
            if int(det.cls) != 0:  # 只取 person 类
                continue
            x1, y1, x2, y2 = [float(v) for v in det.xyxy[0]]
            box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            persons.append(box)

        if not persons:
            return None

        predicted = self._predict_center()
        if predicted is not None:
            # 尺寸约束：过滤掉与目标基准尺寸偏离过大的检测（如把两人框一起的大框）
            tw, th = self._target_size()
            filtered = []
            for box in persons:
                bw, bh = box[2], box[3]
                if (abs(bw - tw) / max(tw, 1) <= self._size_tolerance and
                        abs(bh - th) / max(th, 1) <= self._size_tolerance):
                    filtered.append(box)
            candidates = filtered if filtered else persons
            # 选离轨迹预测中心最近的人
            px, py = predicted
            return min(
                candidates,
                key=lambda b: (self._center(b)[0] - px) ** 2
                + (self._center(b)[1] - py) ** 2,
            )

        # 无历史轨迹（退化路径）：IoU 匹配 + 最近中心降级
        best_box, best_iou = None, 0.0
        cx, cy = self._center(self._last_box)
        for box in persons:
            iou = self._iou(self._last_box, box)
            if iou > best_iou:
                best_iou, best_box = iou, box
        if best_iou < 0.1:
            nearest = None
            min_dist = float("inf")
            for b in persons:
                d = (self._center(b)[0] - cx) ** 2 + (self._center(b)[1] - cy) ** 2
                if d < min_dist:
                    min_dist, nearest = d, b
            best_box = nearest if nearest is not None else best_box
        return best_box

    @staticmethod
    def _center(box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
