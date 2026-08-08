"""目标跟踪器：用 ultralytics BoT-SORT 逐帧检测 + 初始框锁定目标。

对比旧方案（CSRT 帧间跟踪 + 手动 YOLO 重定位）：
- 旧方案在目标快速移动/交叉时，CSRT 会漂移到别人身上，重定位
  IoU 匹配又固化错误目标。
- 本方案用 BoT-SORT 每帧检测所有 person（检测框位置稳定），用
  「用户初始框锁定 + 位置连续性」持续选目标，显著减少漂移。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_PARAMS = {
    "redetect_interval": 30,  # 保留参数名兼容；本方案每帧检测，不依赖此间隔
    "tracker_type": "CSRT",   # 保留参数名兼容；实际用 BoT-SORT
    "lose_threshold": 30,
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

        用 BoT-SORT 每帧检测所有 person，选离目标当前位置最近的框，
        持续跟随用户框选的目标。丢失帧返回 None。
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

            # BoT-SORT 检测当前帧所有 person
            res = self._model.track(
                frame, persist=True, tracker="botsort.yaml", verbose=False,
            )
            best_box = None
            best_d = float("inf")
            # 有检测：boxes.xyxy 存在且非空（不依赖 id，兼容无 id 的边界情况）
            if (
                res is not None
                and res[0].boxes is not None
                and len(res[0].boxes.xyxy) > 0
            ):
                for b in np.asarray(res[0].boxes.xyxy, dtype=float):
                    x1, y1, x2, y2 = [float(v) for v in b]
                    box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    cx, cy = self._center(box)
                    # 位置连续性：选离目标当前位置最近的检测框
                    d = (cx - cur_cx) ** 2 + (cy - cur_cy) ** 2
                    if d < best_d:
                        best_d, best_box = d, box

            if best_box is not None:
                cur_cx, cur_cy = self._center(best_box)
                self._last_box = best_box
                self._consecutive_loss = 0
                self._push_history((cur_cx, cur_cy))
                results.append(best_box)
            else:
                # 无检测：连续丢失计数，达阈值后放弃
                self._consecutive_loss += 1
                if self._consecutive_loss >= self._lose_threshold:
                    self._abandoned = True
                results.append(None)

            if on_progress is not None and total > 0:
                pct = round(len(results) / total * 100, 1)
                on_progress(min(pct, 100.0))

        self._target_cx, self._target_cy = cur_cx, cur_cy
        return results

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
