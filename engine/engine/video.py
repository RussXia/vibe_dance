"""视频读取器：基于 OpenCV 的随机访问帧读取。"""
from __future__ import annotations

import cv2
import numpy as np


class VideoReader:
    """对视频文件的随机访问读取器。

    帧索引为 0-based；越界访问返回 None。
    帧为 BGR 三通道 uint8。
    """

    def __init__(self, path: str):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.path}")
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read_frame(self, index: int) -> np.ndarray | None:
        """读取第 index 帧（0-based）。越界或失败返回 None。

        注意：对 H.264 等帧间压缩视频，随机访问（set+read）极慢
        （实测比顺序读慢 60 倍+）。需要逐帧处理时优先用 iter_frames()。
        """
        if index < 0 or index >= self._frame_count:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def iter_frames(self, start_index: int = 0):
        """顺序流式读取：从 start_index 起逐帧 yield（0-based）。

        用 cap.read() 一次顺序解码，避免随机访问的跳帧解码开销。
        帧为 BGR 三通道 uint8；读尽或失败时停止。
        """
        if start_index < 0 or start_index >= self._frame_count:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return
            yield frame

    def release(self) -> None:
        self._cap.release()
