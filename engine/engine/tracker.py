"""目标跟踪器：用 ultralytics BoT-SORT 逐帧检测 + 初始框锁定目标。

对比旧方案（CSRT 帧间跟踪 + 手动 YOLO 重定位）：
- 旧方案在目标快速移动/交叉时，CSRT 会漂移到别人身上，重定位
  IoU 匹配又固化错误目标。
- 本方案用 BoT-SORT 每帧检测所有 person（检测框位置稳定），用
  「用户初始框锁定 + 位置连续性」持续选目标，显著减少漂移。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_DEFAULT_PARAMS = {
    "redetect_interval": 30,  # 保留参数名兼容；本方案每帧检测，不依赖此间隔
    "tracker_type": "CSRT",   # 保留参数名兼容；实际用 BoT-SORT
    "lose_threshold": 30,
    "track_scale": 0.4,       # CSRT 兜底跟踪在缩放帧上跑，提速
    "max_jump": 30,           # 检测框与 CSRT 中心偏差超此值视为检测跳变，信 CSRT
}


def _model_path() -> Path:
    """返回随引擎分发的 YOLO 权重路径（优先本地，避免打包后联网下载）。"""
    here = Path(__file__).resolve().parent
    for candidate in (here / "yolov8n.pt", here.parent / "yolov8n.pt", Path("yolov8n.pt")):
        if candidate.exists():
            return candidate
    return Path("yolov8n.pt")


class PersonTracker:
    """对单个目标人物的跟踪器（BoT-SORT 检测 + 初始框锁定）。

    - 逐帧用 ultralytics BoT-SORT 检测所有 person，检测框位置稳定。
    - 首帧用用户初始框匹配最近检测框，锁定目标。
    - 后续帧选「离上一帧目标位置最近」的检测框，保持目标跟随。
    """

    def __init__(self, reader, init_frame, bbox, params=None):
        self._reader = reader
        self._init_frame = init_frame
        self._params = {**_DEFAULT_PARAMS, **(params or {})}
        self._init_bbox = tuple(int(v) for v in bbox)
        self._last_box = self._init_bbox
        self._model = None
        self._lose_threshold = int(self._params.get("lose_threshold", 30))
        self._consecutive_loss = 0
        self._abandoned = False
        # 目标当前位置（原始分辨率，中心），首帧锁定后持续更新
        self._target_cx = self._init_bbox[0] + self._init_bbox[2] / 2
        self._target_cy = self._init_bbox[1] + self._init_bbox[3] / 2
        # 历史目标中心，用于位置连续性校验
        self._history_centers: list[tuple[float, float]] = []
        # CSRT 兜底跟踪器：BoT-SORT 检测跳变/波动时，用 CSRT 保持位置连续。
        # 检测正常时每帧用确认框重初始化，使 CSRT 始终跟随已确认目标。
        self._csrt = None
        # 检测跳变判定阈值（像素）：检测框与 CSRT 中心偏差超此值视为检测跳变
        self._max_jump = float(self._params.get("max_jump", 30))
        # 用缩放帧跑 CSRT（提速），坐标换算回原始分辨率
        self._scale = float(self._params.get("track_scale", 0.4))
        # 是否启用 CSRT 兜底（测试可关闭，聚焦检测路径）
        self._csrt_enabled = bool(self._params.get("enable_csrt_fallback", True))

    def _ensure_model(self):
        """确保 YOLO 模型已加载。"""
        from ultralytics import YOLO

        if self._model is None:
            self._model = YOLO(str(_model_path()))

    def _push_history(self, center: tuple[float, float]):
        """把一帧有效目标中心推入历史轨迹，最多保留最近 10 个。"""
        self._history_centers.append(center)
        if len(self._history_centers) > 10:
            self._history_centers.pop(0)

    def track(self, max_frames=None, on_progress=None):
        """逐帧跟踪，返回每帧目标 bbox 列表（原始分辨率）。

        主路径 BoT-SORT 检测（检测框位置准）+ 兜底 CSRT 帧间跟踪
        （检测跳变/波动时保持位置连续，避免跳到旁边的人）。
        """
        self._ensure_model()
        results = []
        total = self._reader.frame_count - self._init_frame
        cur_cx, cur_cy = self._target_cx, self._target_cy

        for frame_index, frame in enumerate(
            self._reader.iter_frames(self._init_frame),
            start=self._init_frame,
        ):
            if max_frames is not None and len(results) >= max_frames:
                break
            if self._abandoned:
                results.append(None)
                continue

            # 缩放帧用于 CSRT 兜底（提速）
            sframe = self._scale_frame(frame)

            # ---- CSRT 兜底：每帧更新，保持跟随已确认目标 ----
            csrt_box = None
            if self._csrt is not None and self._csrt_enabled:
                try:
                    ok_csrt, sbox = self._csrt.update(sframe)
                    if ok_csrt:
                        csrt_box = self._unscale_box(
                            tuple(int(v) for v in sbox)
                        )
                except cv2.error:
                    csrt_box = None

            # ---- 主路径：BoT-SORT 检测最近框 ----
            res = self._model.track(
                frame, persist=True, tracker="botsort.yaml", verbose=False,
            )
            best_box = None
            best_d = float("inf")
            if (
                res is not None
                and res[0].boxes is not None
                and len(res[0].boxes.xyxy) > 0
            ):
                for b in np.asarray(res[0].boxes.xyxy, dtype=float):
                    x1, y1, x2, y2 = [float(v) for v in b]
                    box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    cx, cy = self._center(box)
                    d = (cx - cur_cx) ** 2 + (cy - cur_cy) ** 2
                    if d < best_d:
                        best_d, best_box = d, box

            # ---- 决策：检测框 vs CSRT 兜底 ----
            chosen_box = None
            if best_box is not None:
                bx, by, bw, bh = best_box
                b_cx, b_cy = self._center(best_box)
                # 检测框与 CSRT 框的一致性：都可用时比较中心距离
                # 一致（距离小）→ 用检测框（可信）；不一致 → 检测可能跳变，用 CSRT
                if csrt_box is not None:
                    c_cx, c_cy = self._center(csrt_box)
                    disagree = ((b_cx - c_cx) ** 2 + (b_cy - c_cy) ** 2) ** 0.5
                    if disagree <= self._max_jump:
                        # 检测与 CSRT 一致：用检测框，并重初始化 CSRT
                        chosen_box = best_box
                        if self._csrt_enabled:
                            self._csrt = self._create_tracker()
                            sbox = self._scale_box(best_box)
                            try:
                                self._csrt.init(sframe, tuple(sbox))
                            except cv2.error:
                                self._csrt = None
                    else:
                        # 检测与 CSRT 不一致（检测可能跳变）：信 CSRT
                        chosen_box = csrt_box
                else:
                    # 无 CSRT（首帧/刚丢失恢复）：用检测框，初始化 CSRT
                    chosen_box = best_box
                    if self._csrt_enabled:
                        self._csrt = self._create_tracker()
                        sbox = self._scale_box(best_box)
                        try:
                            self._csrt.init(sframe, tuple(sbox))
                        except cv2.error:
                            self._csrt = None
            elif csrt_box is not None:
                # 无检测：用 CSRT 兜底保持位置，但检测仍视为丢失（计入丢失计数）
                chosen_box = csrt_box

            if chosen_box is not None:
                cur_cx, cur_cy = self._center(chosen_box)
                self._last_box = chosen_box
                if best_box is not None:
                    # 检测成功才重置丢失计数
                    self._consecutive_loss = 0
                else:
                    # 检测丢失（用 CSRT 兜底）→ 计入丢失，达阈值放弃
                    self._consecutive_loss += 1
                    if self._consecutive_loss >= self._lose_threshold:
                        self._abandoned = True
                        self._csrt = None
                self._push_history((cur_cx, cur_cy))
                results.append(chosen_box)
            else:
                # 无检测也无 CSRT：连续丢失计数，达阈值后放弃
                self._consecutive_loss += 1
                if self._consecutive_loss >= self._lose_threshold:
                    self._abandoned = True
                results.append(None)

            if on_progress is not None and total > 0:
                pct = round(len(results) / total * 100, 1)
                on_progress(min(pct, 100.0))

        self._target_cx, self._target_cy = cur_cx, cur_cy
        return results

    def _scale_frame(self, frame):
        """把帧缩到 track_scale 比例，供 CSRT 兜底加速。"""
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

    def _create_tracker(self):
        """创建 CSRT 兜底跟踪器。"""
        return cv2.TrackerCSRT_create()

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
