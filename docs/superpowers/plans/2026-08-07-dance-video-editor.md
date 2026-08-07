# 视频剪辑软件（Vibe Dance Editor）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 macOS/Windows 桌面客户端，从多人视频中框选目标人物，自动跟踪并输出以该人物为中心的 9:16 裁剪视频。

**Architecture:** Electron + React 提供 UI（视频预览 + 9:16 取景框 + 进度），Python 处理引擎负责逐帧跟踪（CSRT/KCF + YOLO 重定位）与裁剪编码（FFmpeg）。两者通过 localhost HTTP/WebSocket 通信。算法引擎是纯 Python，不依赖 UI，独立可测。

**Tech Stack:**
- UI: Electron 33+, React 18, Vite, TypeScript
- 引擎: Python 3.13, OpenCV (opencv-contrib-python >=4.9,<5 — CSRT/KCF 单目标跟踪器只在 contrib 中提供), ultralytics YOLOv8n, FFmpeg (系统级, 8.x)
- 打包: electron-builder + PyInstaller

## Global Constraints

- Python 版本用 **3.13**（不用 3.14；torch/ultralytics 对 3.14 支持可能滞后）。命令用 `python3.13`。
- FFmpeg 用系统已装版本（`ffmpeg 8.1.1`），引擎通过 subprocess 调用，不打包 FFmpeg 二进制。
- YOLOv8n 模型首次运行自动下载（约 6MB），缓存到 `~/.cache/ultralytics`。
- 输出宽高必须为**偶数**（H.264 编码器要求）。
- 所有坐标约定：框选/引擎内部均为**视频原始分辨率像素坐标**，UI 只负责在 canvas 上换算。
- 语言：UI 文案与代码注释用中文；代码标识符用英文。
- 每个任务以「测试 → 提交」结束，提交信息前缀 `feat:` / `fix:` / `test:` / `chore:`。

---

### Task 1: 项目脚手架（monorepo 结构 + 引擎骨架）

**Files:**
- Create: `package.json`
- Create: `engine/pyproject.toml`
- Create: `engine/engine/__init__.py`
- Create: `engine/engine/version.py`
- Create: `engine/tests/__init__.py`
- Create: `engine/tests/test_version.py`
- Create: `.gitignore`
- Create: `README.md`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: `engine/engine/version.py` 暴露 `__version__: str`；`engine/` 可用 `python3.13 -m pytest` 跑测试。

- [ ] **Step 1: 根 package.json**（git 仓库已由控制器初始化于 `main` 分支的 `feat/dance-video-editor` 分支，**跳过 `git init`**）

```bash
cd /Users/RuzZ/personalspace/vibe_dance
```

写根 `package.json`：

```json
{
  "name": "vibe-dance",
  "private": true,
  "version": "0.1.0",
  "description": "视频剪辑软件：多人视频中框选目标人物，自动跟踪并裁剪输出",
  "workspaces": ["app"]
}
```

- [ ] **Step 2: 创建引擎 Python 包结构**

写 `engine/pyproject.toml`：

```toml
[project]
name = "vibe-dance-engine"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
  "opencv-contrib-python>=4.9,<5",
  "ultralytics>=8.2",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

写 `engine/engine/__init__.py`：

```python
"""Vibe Dance 处理引擎。"""
from .version import __version__
```

写 `engine/engine/version.py`：

```python
__version__ = "0.1.0"
```

写 `engine/tests/__init__.py`（空文件）。

写 `engine/tests/test_version.py`：

```python
from engine.version import __version__


def test_version():
    assert __version__ == "0.1.0"
```

- [ ] **Step 3: 创建 venv 并安装最小依赖**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pytest opencv-contrib-python
```

Expected: pip 安装成功，无报错。

- [ ] **Step 4: 跑测试验证**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest -v
```

Expected: `test_version` PASS。

- [ ] **Step 5: 写 .gitignore 与 README**

写 `.gitignore`：

```gitignore
node_modules/
dist/
app/dist/
app/release/
engine/.venv/
engine/.pytest_cache/
engine/__pycache__/
engine/engine/__pycache__/
engine/tests/__pycache__/
*.pyc
.DS_Store
output/
```

写 `README.md`：

```markdown
# Vibe Dance Editor

从多人视频中框选目标人物，自动跟踪并输出以该人物为中心的 9:16 裁剪视频。

## 结构

- `app/` — Electron + React 桌面客户端
- `engine/` — Python 处理引擎（OpenCV + YOLO 跟踪与裁剪）

## 环境要求

- Node.js 20+
- Python 3.13
- FFmpeg 8.x（系统级）

详见 `docs/superpowers/specs/2026-08-07-dance-video-editor-design.md`。
```

- [ ] **Step 6: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add -A
git commit -m "chore: scaffold project structure (electron app + python engine)"
```

---

### Task 2: 引擎核心 — 视频读取器

**Files:**
- Create: `engine/engine/video.py`
- Test: `engine/tests/test_video.py`
- Create: `engine/tests/fixtures.py`

**Interfaces:**
- Consumes: Task 1 的 `engine` 包结构。
- Produces: `engine.engine.video.VideoReader`：
  - `__init__(self, path: str) -> None`
  - `width: int` / `height: int` / `fps: float` / `frame_count: int`（属性）
  - `read_frame(self, index: int) -> numpy.ndarray | None`（0-based 随机访问，BGR 三通道 uint8）
  - `release(self) -> None`

- [ ] **Step 1: 写失败测试**

写 `engine/tests/fixtures.py`（合成视频工具，供多任务复用）：

```python
import cv2
import numpy as np


def make_synthetic_video(path, width=320, height=240, fps=10, frames=20):
    """生成 20 帧含移动方块的合成视频，用于跟踪测试。"""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "合成视频写入失败"
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + i * 5, 10), (60 + i * 5, 90), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
```

写 `engine/tests/test_video.py`：

```python
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
```

- [ ] **Step 2: 跑测试验证失败**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest tests/test_video.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'engine.video'`。

- [ ] **Step 3: 实现 VideoReader**

写 `engine/engine/video.py`：

```python
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
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"无法打开视频文件: {path}")
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
        """读取第 index 帧（0-based）。越界或失败返回 None。"""
        if index < 0 or index >= self._frame_count:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        self._cap.release()
```

- [ ] **Step 4: 跑测试验证通过**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest tests/test_video.py -v
```

Expected: 3 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add engine/engine/video.py engine/tests/test_video.py engine/tests/fixtures.py
git commit -m "feat: add video reader with random frame access"
```

---

### Task 3: 引擎核心 — 目标跟踪器（CSRT + 重定位）

**Files:**
- Create: `engine/engine/tracker.py`
- Test: `engine/tests/test_tracker.py`

**Interfaces:**
- Consumes: Task 2 的 `VideoReader` 与 `make_synthetic_video` fixture。
- Produces:
  - `engine.engine.tracker.PersonTracker`：
    - `__init__(self, reader: VideoReader, init_frame: int, bbox: tuple[int, int, int, int], params: dict | None = None) -> None`（bbox = `(x, y, w, h)`，视频原始分辨率）
    - `track(self, max_frames: int | None = None) -> list[tuple[int, int, int, int] | None]`（返回每帧 bbox，跟踪丢失时该帧为 None，长度 = min(max_frames, 剩余帧)）
    - `tracker_type: str`（`"CSRT"` / `"KCF"`）
    - `params`：`{"redetect_interval": int=30, "tracker_type": str="CSRT"}`

- [ ] **Step 1: 写失败测试**

写 `engine/tests/test_tracker.py`：

```python
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
```

- [ ] **Step 2: 跑测试验证失败**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest tests/test_tracker.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'engine.tracker'`。

- [ ] **Step 3: 实现 PersonTracker**

写 `engine/engine/tracker.py`：

```python
"""目标跟踪器：CSRT/KCF 逐帧跟踪 + YOLO 定期重定位。"""
from __future__ import annotations

import cv2

from .video import VideoReader

_DEFAULT_PARAMS = {
    "redetect_interval": 30,
    "tracker_type": "CSRT",
    "lose_threshold": 30,
}


class PersonTracker:
    """对单个目标人物的跟踪器。

    - 帧间使用 OpenCV 单目标跟踪器（CSRT 或 KCF）。
    - 每 redetect_interval 帧用 YOLOv8n 人体检测重定位，取与当前框 IoU 最高者。
    - 跟踪连续丢失超过 lose_threshold 帧时，后续帧返回 None（彻底丢失）。
    """

    def __init__(self, reader, init_frame, bbox, params=None):
        self._reader = reader
        self._init_frame = init_frame
        self._params = {**_DEFAULT_PARAMS, **(params or {})}
        self.tracker_type = self._params["tracker_type"]
        self._init_bbox = tuple(int(v) for v in bbox)
        self._last_box = self._init_bbox
        self._tracker = None
        self._model = None
        self._lose_threshold = int(self._params.get("lose_threshold", 30))
        self._consecutive_loss = 0
        self._abandoned = False

    def _create_tracker(self):
        if self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        return cv2.TrackerCSRT_create()

    def track(self, max_frames=None):
        results = []
        for frame_index in range(self._init_frame, self._reader.frame_count):
            if max_frames is not None and len(results) >= max_frames:
                break
            frame = self._reader.read_frame(frame_index)
            if frame is None:
                break
            if self._abandoned:
                results.append(None)
                continue
            if frame_index == self._init_frame:
                ok, box = self._start_tracking(frame, self._init_bbox)
            else:
                ok, box = self._continue_tracking(frame, frame_index)
            if not ok:
                self._consecutive_loss += 1
                if self._consecutive_loss >= self._lose_threshold:
                    self._abandoned = True
            else:
                self._consecutive_loss = 0
            results.append(box if ok else None)
        return results

    def _start_tracking(self, frame, bbox):
        self._tracker = self._create_tracker()
        ok = self._tracker.init(frame, tuple(bbox))
        self._last_box = bbox
        if not ok:
            return False, None
        # 初始化帧直接返回初始框
        return True, tuple(int(v) for v in bbox)

    def _continue_tracking(self, frame, frame_index):
        ok, box = self._tracker.update(frame)
        if not ok:
            return False, None
        x, y, w, h = (int(v) for v in box)
        # 定期重定位
        if frame_index % self._params["redetect_interval"] == 0:
            new_box = self._redetect(frame)
            if new_box is not None:
                x, y, w, h = new_box
                self._tracker.init(frame, (x, y, w, h))
        self._last_box = (x, y, w, h)
        return True, (x, y, w, h)

    def _redetect(self, frame):
        """YOLO 检测所有人，返回与当前框 IoU 最高的 person 框。"""
        from ultralytics import YOLO

        if self._model is None:
            self._model = YOLO("yolov8n.pt")
        dets = self._model(frame, verbose=False)[0]
        best_box, best_iou = None, 0.0
        cx, cy = self._center(self._last_box)
        for det in dets.boxes:
            if int(det.cls) != 0:  # 只取 person 类
                continue
            x1, y1, x2, y2 = [float(v) for v in det.xyxy[0]]
            box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            iou = self._iou(self._last_box, box)
            if iou > best_iou:
                best_iou, best_box = iou, box
        # IoU 过低说明当前框可能已漂移，用离当前中心最近的检测替代
        if best_iou < 0.1:
            nearest = None
            min_dist = float("inf")
            for det in dets.boxes:
                if int(det.cls) != 0:
                    continue
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[0]]
                b = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
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
```


- [ ] **Step 4: 跑测试验证通过**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest tests/test_tracker.py -v
```

Expected: 3 个测试全部 PASS（首次运行会下载 yolov8n.pt 约 6MB）。

**注意**：`redetect_interval` 默认 30，而合成视频只有 20 帧，重定位不会触发，测试只验证基础跟踪。若要验证重定位，可临时把 `redetect_interval` 调成 5（在单独的集成测试中做）。

- [ ] **Step 5: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add engine/engine/tracker.py engine/tests/test_tracker.py
git commit -m "feat: add person tracker with csrt/kcf + yolo redetect"
```

---

### Task 4: 引擎核心 — 输出框与裁剪渲染

**Files:**
- Create: `engine/engine/render.py`
- Test: `engine/tests/test_render.py`

**Interfaces:**
- Consumes: Task 2 的 `VideoReader`，Task 3 的 `PersonTracker`。
- Produces:
  - `engine.engine.render.Renderer`：
    - `__init__(self, reader: VideoReader, tracker: PersonTracker, init_frame: int, viewport: tuple[int, int, int, int], output_size: tuple[int, int], params: dict | None = None) -> None`
      - `viewport` = 第一帧取景框 `(x, y, w, h)`（9:16，视频原始分辨率坐标）
      - `output_size` = `(width, height)` 输出尺寸
      - `params`：`{"smooth_window": int=10}`
    - `render(self, output_path: str, on_progress: Callable[[int], None] | None = None) -> None`
    - 输出：H.264 MP4，固定输出分辨率，进度回调 0-100 整数百分比。

- [ ] **Step 1: 写失败测试**

写 `engine/tests/test_render.py`：

```python
import os

import cv2
import numpy as np

from engine.render import Renderer
from engine.tracker import PersonTracker
from engine.video import VideoReader
from .fixtures import make_synthetic_video


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
```

- [ ] **Step 2: 跑测试验证失败**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest tests/test_render.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'engine.render'`。

- [ ] **Step 3: 实现 Renderer**

写 `engine/engine/render.py`：

```python
"""裁剪渲染器：把跟踪到的目标裁剪到固定输出尺寸并编码。"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


class Renderer:
    """按固定输出框把视频裁剪为输出分辨率，用 FFmpeg 编码为 H.264 MP4。"""

    def __init__(self, reader, tracker, init_frame, viewport, output_size, params=None):
        self._reader = reader
        self._tracker = tracker
        self._init_frame = init_frame
        self._viewport = tuple(int(v) for v in viewport)
        self._output_size = (int(output_size[0]), int(output_size[1]))
        self._smooth_window = (params or {}).get("smooth_window", 10)

    def _smooth(self, boxes):
        """对 bbox 序列做指数移动平均平滑。None 帧保持上一值。"""
        smoothed = []
        last = None
        alpha = 2.0 / (self._smooth_window + 1)
        for box in boxes:
            if box is None:
                smoothed.append(last)
                continue
            if last is None:
                last = [float(v) for v in box]
            else:
                for i in range(4):
                    last[i] = alpha * box[i] + (1 - alpha) * last[i]
            smoothed.append([int(round(v)) for v in last])
        return smoothed

    def _crop_centered(self, frame, box):
        """取 viewport 尺寸、中心对齐 box 的裁剪区域，clamp 到视频边界。"""
        vw, vh = self._viewport[2], self._viewport[3]
        fh, fw = frame.shape[:2]
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        x1 = int(round(cx - vw / 2))
        y1 = int(round(cy - vh / 2))
        x1 = max(0, min(x1, fw - vw))
        y1 = max(0, min(y1, fh - vh))
        x2, y2 = x1 + vw, y1 + vh
        # 若取景框比画面还大，退回画满
        if vw >= fw or vh >= fh:
            return frame
        return frame[y1:y2, x1:x2]

    def render(self, output_path, on_progress=None):
        boxes = self._tracker.track()
        smoothed = self._smooth(boxes)
        out_w, out_h = self._output_size
        total = len(smoothed)
        if total == 0:
            raise RuntimeError("没有可渲染的帧")

        # 收集所有裁剪帧到临时目录，再交给 FFmpeg 编码
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for i, box in enumerate(smoothed):
                frame = self._reader.read_frame(self._init_frame + i)
                if frame is None or box is None:
                    continue
                crop = self._crop_centered(frame, box)
                resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(tmpdir_path / f"{i:06d}.png"), resized)
                if on_progress is not None and (i + 1) % max(1, total // 20) == 0:
                    on_progress(int((i + 1) / total * 100))

            # FFmpeg 编码（PNG 序列 → H.264 MP4）
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self._reader.fps),
                "-i", str(tmpdir_path / "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(output_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg 编码失败: {proc.stderr[-500:]}")
        if on_progress is not None:
            on_progress(100)
```

**注意**：临时帧用 PNG 序列会占大量磁盘（每帧 ≈ 输出分辨率×4 字节）。对 MVP 可接受，但实现时若遇磁盘压力，可改为逐帧管道喂给 FFmpeg stdin（`-f rawvideo` + 子进程 stdin 写入）。MVP 用 PNG 序列优先保证正确性。

- [ ] **Step 4: 跑测试验证通过**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest tests/test_render.py -v
```

Expected: `test_render_produces_output_video` PASS（输出 180×320、20 帧、进度到 100）。

- [ ] **Step 5: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add engine/engine/render.py engine/tests/test_render.py
git commit -m "feat: add renderer with smooth crop and ffmpeg encode"
```

---

### Task 5: 引擎 HTTP 服务（Task API）

**Files:**
- Create: `engine/engine/server.py`
- Create: `engine/engine/task.py`
- Create: `engine/engine/__main__.py`
- Test: `engine/tests/test_server.py`

**Interfaces:**
- Consumes: Task 2/3/4 的 `VideoReader` / `PersonTracker` / `Renderer`。
- Produces:
  - `engine.engine.task.TaskManager`：
    - `submit(self, video_path: str, init_frame: int, bbox: tuple[int,int,int,int], output_size: tuple[int,int], output_path: str, params: dict | None = None) -> str`（返回 task_id）
    - `get(self, task_id: str) -> dict`（`{"task_id", "status", "progress", "message"}`）
    - `cancel(self, task_id: str) -> bool`
  - `engine.engine.server.start(port: int = 8787) -> None`（阻塞，起 HTTP 服务）
  - HTTP 接口：`POST /task`、`GET /task/{id}`、`POST /task/{id}/cancel`。用 Python 标准库 `http.server`，**不引入 Flask/FastAPI**（避免额外依赖）。

- [ ] **Step 1: 写失败测试**

写 `engine/tests/test_server.py`：

```python
import json
import threading
import time
import urllib.request

from engine.server import start, PORT
from .fixtures import make_synthetic_video


def _wait_server_ready(base, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError("server not ready")


def _post(base, path, payload):
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=15) as resp:
        return json.loads(resp.read().decode())


def test_submit_and_query_task(tmp_path):
    video = str(tmp_path / "v.mp4")
    make_synthetic_video(video, frames=15)
    out = str(tmp_path / "out.mp4")
    port = 8891
    base = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=start, kwargs={"port": port}, daemon=True)
    thread.start()
    try:
        _wait_server_ready(base)
        resp = _post(base, "/task", {
            "video_path": video,
            "init_frame": 0,
            "bbox": [100, 60, 80, 140],
            "output_size": [180, 320],
            "output_path": out,
            "params": {},
        })
        task_id = resp["task_id"]
        assert resp["status"] == "QUEUED"

        # 轮询直到 DONE
        for _ in range(30):
            st = _get(base, f"/task/{task_id}")
            if st["status"] in ("DONE", "FAILED"):
                break
            time.sleep(0.2)
        assert st["status"] == "DONE", st
        assert st["progress"] == 100
    finally:
        # 关闭服务
        import engine.server as srv
        srv.stop()
```

- [ ] **Step 2: 跑测试验证失败**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest tests/test_server.py -v
```

Expected: FAIL，`ModuleNotFoundError: No module named 'engine.server'`。

- [ ] **Step 3: 实现 task.py（任务管理）**

写 `engine/engine/task.py`：

```python
"""任务管理：提交 / 查询 / 取消跟踪渲染任务。"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from .render import Renderer
from .tracker import PersonTracker
from .video import VideoReader


@dataclass
class Task:
    task_id: str
    video_path: str
    init_frame: int
    bbox: tuple
    output_size: tuple
    output_path: str
    params: dict
    status: str = "QUEUED"
    progress: int = 0
    message: str = ""
    _thread: threading.Thread = field(default=None, repr=False)
    _cancel: bool = field(default=False, repr=False)


class TaskManager:
    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def submit(self, video_path, init_frame, bbox, output_size, output_path, params=None):
        task_id = uuid.uuid4().hex[:12]
        task = Task(
            task_id=task_id,
            video_path=video_path,
            init_frame=init_frame,
            bbox=tuple(int(v) for v in bbox),
            output_size=(int(output_size[0]), int(output_size[1])),
            output_path=output_path,
            params=params or {},
        )
        with self._lock:
            self._tasks[task_id] = task
        task._thread = threading.Thread(target=self._run, args=(task,), daemon=True)
        task._thread.start()
        return task_id

    def get(self, task_id):
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return None
            return {
                "task_id": t.task_id,
                "status": t.status,
                "progress": t.progress,
                "message": t.message,
            }

    def cancel(self, task_id):
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return False
            if t.status in ("DONE", "FAILED", "CANCELLED"):
                return False
            t._cancel = True
            t.status = "CANCELLED"
            return True

    def _run(self, task):
        try:
            task.status = "RUNNING"
            reader = VideoReader(task.video_path)
            try:
                tracker = PersonTracker(reader, task.init_frame, task.bbox, task.params)
                renderer = Renderer(
                    reader, tracker, task.init_frame,
                    task.bbox, task.output_size, task.params,
                )
                # 取消检查：渲染过程中定期检查 task._cancel
                def on_progress(p):
                    if task._cancel:
                        raise _Cancelled()
                    task.progress = p

                renderer.render(task.output_path, on_progress=on_progress)
            finally:
                reader.release()
            if task._cancel:
                task.status = "CANCELLED"
            else:
                task.status = "DONE"
                task.progress = 100
        except _Cancelled:
            task.status = "CANCELLED"
        except Exception as exc:  # noqa: BLE001
            task.status = "FAILED"
            task.message = str(exc)


class _Cancelled(Exception):
    pass
```

- [ ] **Step 4: 实现 server.py（HTTP 服务）**

写 `engine/engine/server.py`：

```python
"""轻量 HTTP 服务：Task API。用标准库实现，避免额外依赖。"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .task import TaskManager

PORT = 8787

_manager = TaskManager()
_httpd: ThreadingHTTPServer | None = None


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode())

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path.startswith("/task/"):
            task_id = self.path.split("/")[-1]
            info = _manager.get(task_id)
            if info is None:
                self._send_json(404, {"error": "task not found"})
                return
            self._send_json(200, info)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/task":
            payload = self._read_json()
            try:
                task_id = _manager.submit(
                    payload["video_path"],
                    int(payload["init_frame"]),
                    payload["bbox"],
                    payload["output_size"],
                    payload["output_path"],
                    payload.get("params"),
                )
            except KeyError as exc:
                self._send_json(400, {"error": f"missing field: {exc}"})
                return
            self._send_json(200, {"task_id": task_id, "status": "QUEUED"})
            return
        if self.path.endswith("/cancel"):
            task_id = self.path.split("/")[-2]
            ok = _manager.cancel(task_id)
            self._send_json(200, {"cancelled": ok})
            return
        self._send_json(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # 静默访问日志
        pass


def start(port=PORT):
    global _httpd
    _httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    _httpd.serve_forever()


def stop():
    global _httpd
    if _httpd is not None:
        _httpd.shutdown()
        _httpd = None
```

写 `engine/engine/__main__.py`（供 `python -m engine` 启动服务）：

```python
"""启动引擎 HTTP 服务。用法: python -m engine [port]"""
import sys

from .server import start


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"Engine server listening on 127.0.0.1:{port}")
    start(port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 跑测试验证通过**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
python -m pytest tests/test_server.py -v
```

Expected: `test_submit_and_query_task` PASS。

**注意**：`start()` 是阻塞的，测试里用线程跑；`stop()` 需要从测试线程外部调用。若测试偶发超时，可在 `finally` 中确保 `srv.stop()` 执行，并适当增大轮询等待。

- [ ] **Step 6: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add engine/engine/task.py engine/engine/server.py engine/engine/__main__.py engine/tests/test_server.py
git commit -m "feat: add http task api server"
```

---

### Task 6: UI — Electron + React 脚手架 + IPC 打通

**Files:**
- Create: `app/package.json`
- Create: `app/vite.config.ts`
- Create: `app/tsconfig.json`
- Create: `app/index.html`
- Create: `app/electron/main.ts`
- Create: `app/electron/preload.ts`
- Create: `app/src/main.tsx`
- Create: `app/src/App.tsx`
- Create: `app/src/App.test.tsx`
- Create: `app/vitest.config.ts`

**Interfaces:**
- Consumes: Task 5 的引擎 HTTP 服务（Task API）。
- Produces:
  - `app/electron/main.ts`：创建 BrowserWindow；`ipcMain.handle("engine:start", ...)` 启动 Python 引擎子进程（或连接已有服务）；`ipcMain.handle("engine:submit-task", ...)` 转发 HTTP 请求到引擎。
  - `app/electron/preload.ts`：暴露 `window.api`（见下）。
  - `window.api`：
    - `submitTask(payload: object): Promise<{ task_id: string; status: string }>`
    - `getTask(taskId: string): Promise<{ task_id: string; status: string; progress: number; message: string }>`
    - `openVideo(): Promise<{ path: string } | null>`

- [ ] **Step 1: 写失败测试（React 组件最小渲染测试）**

写 `app/package.json`：

```json
{
  "name": "vibe-dance-app",
  "private": true,
  "version": "0.1.0",
  "main": "dist-electron/electron/main.js",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest run",
    "electron:dev": "npm run build && electron ."
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "electron": "^33.2.0",
    "typescript": "^5.6.3",
    "vite": "^5.4.11",
    "vitest": "^2.1.8",
    "@testing-library/react": "^16.0.1",
    "@testing-library/jest-dom": "^6.6.3",
    "jsdom": "^25.0.1"
  }
}
```

> 说明：不设 `"type": "module"`（Electron 33 主进程以 ESM 加载会崩，`require` 不可用）；`main` 指向 tsc 产物 `dist-electron/electron/main.js`（outDir 保留目录结构）。

写 `app/vite.config.ts`：

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: 'dist' },
});
```

写 `app/vitest.config.ts`：

```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: [],
  },
});
```

写 `app/tsconfig.json`：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM"],
    "module": "CommonJS",
    "moduleResolution": "node",
    "jsx": "react-jsx",
    "strict": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "types": ["vitest/globals"],
    "outDir": "dist-electron"
  },
  "include": ["src", "electron", "vite.config.ts", "vitest.config.ts"],
  "exclude": ["src/**/*.test.ts", "src/**/*.test.tsx"]
}
```

> 说明：`module: CommonJS` 使 Electron 33 主进程能加载（ESM 加载会崩）；`outDir` 使 tsc 发射到 `dist-electron/`（保留目录结构，产物为 `dist-electron/electron/main.js`）；`exclude` 避免测试文件被编译进产物并被 vitest 重复扫描。

写 `app/index.html`：

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vibe Dance Editor</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

写 `app/src/main.tsx`：

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

写 `app/src/App.tsx`（MVP 占位 UI）：

```tsx
import { useState } from 'react';

export default function App() {
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [status, setStatus] = useState('idle');

  const openVideo = async () => {
    const res = await window.api.openVideo();
    if (res) {
      setVideoPath(res.path);
      setStatus('ready');
    }
  };

  return (
    <div style={{ padding: 24, fontFamily: 'sans-serif' }}>
      <h1>Vibe Dance Editor</h1>
      <button onClick={openVideo}>打开视频</button>
      {videoPath && <p>视频: {videoPath}</p>}
      <p>状态: {status}</p>
    </div>
  );
}
```

写 `app/src/App.test.tsx`：

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

describe('App', () => {
  it('renders title and open button', () => {
    render(<App />);
    expect(screen.getByText('Vibe Dance Editor')).toBeTruthy();
    expect(screen.getByRole('button', { name: '打开视频' })).toBeTruthy();
  });
});
```

写 `app/electron/main.ts`：

```typescript
import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'node:path';
import { spawn } from 'node:child_process';

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 960,
    height: 720,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }
}

// 启动 Python 引擎子进程
let engineProc: ReturnType<typeof spawn> | null = null;

function startEngine() {
  // 引擎路径：开发时指向 engine/.venv；打包后由 Task 10 注入
  const engineDir = path.join(__dirname, '../../engine');
  engineProc = spawn(
    path.join(engineDir, '.venv/bin/python'),
    ['-m', 'engine', '8787'],
    { cwd: engineDir },
  );
  engineProc.on('exit', (code: number | null) => {
    console.log(`engine exited with code ${code}`);
    engineProc = null;
  });
}

const ENGINE_BASE = 'http://127.0.0.1:8787';

async function engineFetch(pathname: string, init?: RequestInit) {
  const res = await fetch(`${ENGINE_BASE}${pathname}`, init);
  if (!res.ok) {
    throw new Error(`engine error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

ipcMain.handle('engine:submit-task', async (_e, payload) => {
  return engineFetch('/task', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
});

ipcMain.handle('engine:get-task', async (_e, taskId: string) => {
  return engineFetch(`/task/${taskId}`);
});

ipcMain.handle('open-video', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    filters: [{ name: '视频', extensions: ['mp4', 'mov'] }],
    properties: ['openFile'],
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return { path: result.filePaths[0] };
});

app.whenReady().then(() => {
  createWindow();
  ipcMain.handle('engine:start', async () => {
    startEngine();
    return { ok: true };
  });
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
```

写 `app/electron/preload.ts`：

```typescript
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('api', {
  openVideo: () => ipcRenderer.invoke('open-video'),
  submitTask: (payload: object) => ipcRenderer.invoke('engine:submit-task', payload),
  getTask: (taskId: string) => ipcRenderer.invoke('engine:get-task', taskId),
  startEngine: () => ipcRenderer.invoke('engine:start'),
});
```

写 `app/src/vite-env.d.ts`：

```typescript
/// <reference types="vite/client" />

interface Window {
  api: {
    openVideo: () => Promise<{ path: string } | null>;
    submitTask: (payload: object) => Promise<{ task_id: string; status: string }>;
    getTask: (taskId: string) => Promise<{ task_id: string; status: string; progress: number; message: string }>;
    startEngine: () => Promise<{ ok: boolean }>;
  };
}
```

- [ ] **Step 2: 安装依赖并跑测试**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm install
npm test
```

Expected: `App` 渲染测试 PASS。

- [ ] **Step 3: 跑构建验证**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm run build
```

Expected: `dist/` 与 `dist-electron/` 生成，无类型错误。

- [ ] **Step 4: 手动冒烟测试（可选，需 GUI）**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm run electron:dev
```

Expected: Electron 窗口打开，显示 "Vibe Dance Editor" 与「打开视频」按钮。点「打开视频」能弹出文件选择框并显示所选路径。

- [ ] **Step 5: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add app/
git commit -m "feat: scaffold electron + react app with engine ipc bridge"
```

---

### Task 7: UI — 9:16 取景框交互

**Files:**
- Modify: `app/src/App.tsx`
- Create: `app/src/VideoPreview.tsx`
- Create: `app/src/VideoPreview.test.tsx`

**Interfaces:**
- Consumes: Task 6 的 `window.api.openVideo`。
- Produces:
  - `app/src/VideoPreview.tsx` 组件：
    - Props: `{ videoPath: string }`
    - 渲染 `<video>` + canvas overlay 的 9:16 取景框。
    - `onBoxChange(box: { x: number; y: number; w: number; h: number }, videoSize: { width: number; height: number })` 回调（视频原始分辨率坐标）。
    - 交互：拖动取景框移动；拖四角等比缩放（锁 9:16）。

- [ ] **Step 1: 写失败测试**

写 `app/src/VideoPreview.test.tsx`：

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import VideoPreview from './VideoPreview';

// jsdom 不实现 HTMLVideoElement 尺寸，mock 之
Object.defineProperty(HTMLMediaElement.prototype, 'videoWidth', {
  configurable: true,
  get: () => 1080,
});
Object.defineProperty(HTMLMediaElement.prototype, 'videoHeight', {
  configurable: true,
  get: () => 1920,
});
Object.defineProperty(HTMLMediaElement.prototype, 'load', {
  configurable: true,
  value: vi.fn(),
});

describe('VideoPreview 9:16 取景框', () => {
  it('默认显示 9:16 比例取景框', () => {
    const onBoxChange = vi.fn();
    render(<VideoPreview videoPath="test.mp4" onBoxChange={onBoxChange} />);
    // 初始框回调应触发一次，且宽:高 = 9:16
    expect(onBoxChange).toHaveBeenCalled();
    const box = onBoxChange.mock.calls[0][0];
    expect(box.w / box.h).toBeCloseTo(9 / 16, 5);
  });

  it('拖动取景框会更新位置', () => {
    const onBoxChange = vi.fn();
    render(<VideoPreview videoPath="test.mp4" onBoxChange={onBoxChange} />);
    // 触发 mousedown 于框中心附近 + mousemove + mouseup
    // 通过直接调用内部状态较复杂，此处仅验证渲染出取景框元素
    expect(document.querySelector('[data-testid="viewport"]')).toBeTruthy();
  });
});
```

- [ ] **Step 2: 跑测试验证失败**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm test
```

Expected: FAIL，`Cannot find module './VideoPreview'`。

- [ ] **Step 3: 实现 VideoPreview 组件**

写 `app/src/VideoPreview.tsx`：

```tsx
import { useEffect, useRef, useState } from 'react';

const ASPECT = 9 / 16;

export interface ViewportBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Props {
  videoPath: string;
  onBoxChange: (box: ViewportBox, videoSize: { width: number; height: number }) => void;
}

export default function VideoPreview({ videoPath, onBoxChange }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number }>({ width: 1080, height: 1920 });
  const [box, setBox] = useState<ViewportBox | null>(null);
  const dragState = useRef<{ mode: 'move' | 'resize'; startX: number; startY: number; orig: ViewportBox } | null>(null);

  // 视频元数据加载后，初始化居中的 9:16 取景框（占画面 80% 高度）
  const onLoadedMetadata = () => {
    const v = videoRef.current;
    if (!v) return;
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    setVideoSize({ width: vw, height: vh });
    const h = Math.round(vh * 0.8);
    const w = Math.round(h * ASPECT);
    const x = Math.round((vw - w) / 2);
    const y = Math.round((vh - h) / 2);
    const initial = { x, y, w, h };
    setBox(initial);
    onBoxChange(initial, { width: vw, height: vh });
  };

  // 把取景框画到 canvas overlay 上
  useEffect(() => {
    const canvas = canvasRef.current;
    const v = videoRef.current;
    if (!canvas || !v || !box) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = v.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);
    // 换算到显示尺寸
    const scaleX = rect.width / videoSize.width;
    const scaleY = rect.height / videoSize.height;
    const bx = box.x * scaleX;
    const by = box.y * scaleY;
    const bw = box.w * scaleX;
    const bh = box.h * scaleY;
    // 框外暗化
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(0, 0, rect.width, rect.height);
    ctx.clearRect(bx, by, bw, bh);
    // 框线
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.strokeRect(bx, by, bw, bh);
    // 四角把手
    ctx.fillStyle = '#4fc3f7';
    const handle = 8;
    ctx.fillRect(bx - handle / 2, by - handle / 2, handle, handle);
    ctx.fillRect(bx + bw - handle / 2, by - handle / 2, handle, handle);
    ctx.fillRect(bx - handle / 2, by + bh - handle / 2, handle, handle);
    ctx.fillRect(bx + bw - handle / 2, by + bh - handle / 2, handle, handle);
  }, [box, videoSize]);

  // 事件换算：屏幕坐标 → 视频原始分辨率坐标
  const toVideoCoords = (clientX: number, clientY: number) => {
    const v = videoRef.current!;
    const rect = v.getBoundingClientRect();
    const px = (clientX - rect.left) / rect.width * videoSize.width;
    const py = (clientY - rect.top) / rect.height * videoSize.height;
    return { px, py };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!box) return;
    const { px, py } = toVideoCoords(e.clientX, e.clientY);
    // 是否命中四角（屏幕距离 < 12px）
    const scaleX = videoSize.width / videoRef.current!.getBoundingClientRect().width;
    const scaleY = videoSize.height / videoRef.current!.getBoundingClientRect().height;
    const near = (ax: number, ay: number) => Math.hypot((ax - px) * scaleX, (ay - py) * scaleY) < 12;
    const corners = [
      [box.x, box.y],
      [box.x + box.w, box.y],
      [box.x, box.y + box.h],
      [box.x + box.w, box.y + box.h],
    ];
    const mode = corners.some(([cx, cy]) => near(cx, cy)) ? 'resize' : 'move';
    dragState.current = { mode, startX: px, startY: py, orig: { ...box } };
    e.preventDefault();
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const drag = dragState.current;
    if (!drag || !box) return;
    const { px, py } = toVideoCoords(e.clientX, e.clientY);
    const dx = px - drag.startX;
    const dy = py - drag.startY;
    let next: ViewportBox;
    if (drag.mode === 'move') {
      next = {
        ...drag.orig,
        x: drag.orig.x + dx,
        y: drag.orig.y + dy,
      };
    } else {
      // 等比缩放：以中心为锚，取 dx/dy 中缩放更小者，保持 9:16
      const scale = Math.max(0.1, (drag.orig.w + dx) / drag.orig.w);
      next = {
        x: drag.orig.x - (drag.orig.w * (scale - 1)) / 2,
        y: drag.orig.y - (drag.orig.h * (scale - 1)) / 2,
        w: Math.round(drag.orig.w * scale),
        h: Math.round(drag.orig.h * scale),
      };
    }
    // clamp 到画面内
    next.x = Math.max(0, Math.min(next.x, videoSize.width - next.w));
    next.y = Math.max(0, Math.min(next.y, videoSize.height - next.h));
    setBox(next);
    onBoxChange(next, videoSize);
  };

  const handleMouseUp = () => {
    dragState.current = null;
  };

  return (
    <div
      data-testid="viewport-container"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      style={{ position: 'relative', width: '100%', maxWidth: 480 }}
    >
      <video
        ref={videoRef}
        src={videoPath}
        onLoadedMetadata={onLoadedMetadata}
        data-testid="viewport-video"
        style={{ width: '100%', display: 'block' }}
      />
      <canvas
        ref={canvasRef}
        data-testid="viewport"
        onMouseDown={handleMouseDown}
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', cursor: 'move' }}
      />
    </div>
  );
}
```

- [ ] **Step 4: 更新 App.tsx 接入 VideoPreview**

修改 `app/src/App.tsx`：

```tsx
import { useState } from 'react';
import VideoPreview, { ViewportBox } from './VideoPreview';

export default function App() {
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [status, setStatus] = useState('idle');
  const [box, setBox] = useState<ViewportBox | null>(null);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number } | null>(null);

  const openVideo = async () => {
    const res = await window.api.openVideo();
    if (res) {
      setVideoPath(res.path);
      setStatus('ready');
    }
  };

  const handleBoxChange = (b: ViewportBox, size: { width: number; height: number }) => {
    setBox(b);
    setVideoSize(size);
  };

  const startTracking = async () => {
    if (!videoPath || !box || !videoSize) return;
    await window.api.submitTask({
      video_path: videoPath,
      init_frame: 0,
      bbox: [box.x, box.y, box.w, box.h],
      output_size: [1080, 1920],
      output_path: `/tmp/vibe_dance_out_${Date.now()}.mp4`,
    });
    setStatus('tracking');
  };

  return (
    <div style={{ padding: 24, fontFamily: 'sans-serif' }}>
      <h1>Vibe Dance Editor</h1>
      <button onClick={openVideo}>打开视频</button>
      {videoPath && (
        <>
          <p>视频: {videoPath}</p>
          <VideoPreview videoPath={videoPath} onBoxChange={handleBoxChange} />
          <button onClick={startTracking}>开始跟踪</button>
        </>
      )}
      <p>状态: {status}</p>
    </div>
  );
}
```

- [ ] **Step 5: 跑测试验证通过**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm test
```

Expected: `VideoPreview` 测试 PASS，`App` 测试 PASS。

- [ ] **Step 6: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add app/src/VideoPreview.tsx app/src/VideoPreview.test.tsx app/src/App.tsx
git commit -m "feat: add 9:16 viewport selection interaction"
```

---

### Task 8: UI — 任务进度轮询与导出

**Files:**
- Modify: `app/src/App.tsx`
- Create: `app/src/App.test.tsx`（扩展）

**Interfaces:**
- Consumes: Task 6 的 `window.api.submitTask` / `getTask`，Task 7 的 `ViewportBox`。
- Produces: `App` 具备完整导出流程：选视频 → 框选 → 开始跟踪 → 轮询进度 → 显示完成/失败。

- [ ] **Step 1: 写失败测试**

在 `app/src/App.test.tsx` 中新增：

```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';

describe('App 导出流程', () => {
  beforeEach(() => {
    // mock window.api
    (window as any).api = {
      openVideo: vi.fn().mockResolvedValue({ path: '/tmp/a.mp4' }),
      submitTask: vi.fn().mockResolvedValue({ task_id: 'abc', status: 'QUEUED' }),
      getTask: vi.fn().mockResolvedValue({ task_id: 'abc', status: 'DONE', progress: 100, message: '' }),
      startEngine: vi.fn().mockResolvedValue({ ok: true }),
    };
  });

  it('打开视频后显示开始跟踪按钮', async () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: '打开视频' }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始跟踪' })).toBeTruthy();
    });
  });
});
```

- [ ] **Step 2: 跑测试验证失败**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm test
```

Expected: FAIL，`Cannot find window.api`（jsdom 下 `window.api` 未定义）。

- [ ] **Step 3: 实现进度轮询**

修改 `app/src/App.tsx` 的 `startTracking`，加入进度轮询：

```tsx
  const [progress, setProgress] = useState(0);
  const [taskId, setTaskId] = useState<string | null>(null);

  const startTracking = async () => {
    if (!videoPath || !box || !videoSize) return;
    const res = await window.api.submitTask({
      video_path: videoPath,
      init_frame: 0,
      bbox: [box.x, box.y, box.w, box.h],
      output_size: [1080, 1920],
      output_path: `/tmp/vibe_dance_out_${Date.now()}.mp4`,
    });
    setTaskId(res.task_id);
    setStatus('tracking');
    pollTask(res.task_id);
  };

  const pollTask = async (id: string) => {
    const st = await window.api.getTask(id);
    setProgress(st.progress);
    if (st.status === 'DONE') {
      setStatus('done');
      return;
    }
    if (st.status === 'FAILED') {
      setStatus(`failed: ${st.message}`);
      return;
    }
    setTimeout(() => pollTask(id), 500);
  };
```

UI 中渲染进度：

```tsx
      {status === 'tracking' && <p>跟踪中… {progress}%</p>}
      {status === 'done' && <p>✅ 导出完成</p>}
      {status.startsWith('failed') && <p style={{ color: 'red' }}>{status}</p>}
```

- [ ] **Step 4: 跑测试验证通过**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm test
```

Expected: 全部测试 PASS。

- [ ] **Step 5: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add app/src/App.tsx app/src/App.test.tsx
git commit -m "feat: add task progress polling and export flow"
```

---

### Task 9: UI — 输出分辨率设置（自定义尺寸）

**Files:**
- Modify: `app/src/App.tsx`
- Create: `app/src/OutputSizeSelector.tsx`
- Create: `app/src/OutputSizeSelector.test.tsx`

**Interfaces:**
- Consumes: 无新依赖。
- Produces: `app/src/OutputSizeSelector.tsx`：
  - Props: `{ value: { width: number; height: number }; onChange: (v: { width: number; height: number }) => void }`
  - 预设：720×1280 / 1080×1920；支持自定义宽高输入（偶数校验）。

- [ ] **Step 1: 写失败测试**

写 `app/src/OutputSizeSelector.test.tsx`：

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import OutputSizeSelector from './OutputSizeSelector';

describe('OutputSizeSelector', () => {
  it('默认显示 1080x1920', () => {
    const onChange = vi.fn();
    render(<OutputSizeSelector value={{ width: 1080, height: 1920 }} onChange={onChange} />);
    expect(document.querySelector('[data-testid="out-size"]')).toBeTruthy();
  });

  it('选择 720x1280 触发 onChange', () => {
    const onChange = vi.fn();
    render(<OutputSizeSelector value={{ width: 1080, height: 1920 }} onChange={onChange} />);
    fireEvent.click(document.querySelector('[data-testid="preset-720"]')!);
    expect(onChange).toHaveBeenCalledWith({ width: 720, height: 1280 });
  });
});
```

- [ ] **Step 2: 跑测试验证失败**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm test
```

Expected: FAIL，`Cannot find module './OutputSizeSelector'`。

- [ ] **Step 3: 实现 OutputSizeSelector**

写 `app/src/OutputSizeSelector.tsx`：

```tsx
import { useState } from 'react';

const PRESETS = [
  { label: '720×1280', width: 720, height: 1280 },
  { label: '1080×1920', width: 1080, height: 1920 },
];

interface Props {
  value: { width: number; height: number };
  onChange: (v: { width: number; height: number }) => void;
}

export default function OutputSizeSelector({ value, onChange }: Props) {
  const [custom, setCustom] = useState(false);

  const isPreset = PRESETS.some((p) => p.width === value.width && p.height === value.height);

  const applyCustom = (wStr: string, hStr: string) => {
    const w = parseInt(wStr, 10);
    const h = parseInt(hStr, 10);
    if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
      onChange({ width: w % 2 ? w + 1 : w, height: h % 2 ? h + 1 : h });
    }
  };

  return (
    <div data-testid="out-size">
      {PRESETS.map((p) => (
        <button
          key={p.label}
          data-testid={`preset-${p.width}`}
          onClick={() => {
            setCustom(false);
            onChange({ width: p.width, height: p.height });
          }}
          style={{ marginRight: 8 }}
        >
          {p.label}
        </button>
      ))}
      <button
        data-testid="preset-custom"
        onClick={() => setCustom(true)}
      >
        自定义
      </button>
      {custom && (
        <div>
          <input data-testid="custom-w" type="number" placeholder="宽" defaultValue={value.width}
            onChange={(e) => applyCustom(e.target.value, value.height.toString())} />
          <span>×</span>
          <input data-testid="custom-h" type="number" placeholder="高" defaultValue={value.height}
            onChange={(e) => applyCustom(value.width.toString(), e.target.value)} />
          <p>宽高需为偶数</p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: 接入 App.tsx**

修改 `app/src/App.tsx`：引入 `OutputSizeSelector`，用 `outputSize` state 替换硬编码 `[1080, 1920]`：

```tsx
  const [outputSize, setOutputSize] = useState({ width: 1080, height: 1920 });
  // ...
  const startTracking = async () => {
    if (!videoPath || !box || !videoSize) return;
    const res = await window.api.submitTask({
      video_path: videoPath,
      init_frame: 0,
      bbox: [box.x, box.y, box.w, box.h],
      output_size: [outputSize.width, outputSize.height],
      output_path: `/tmp/vibe_dance_out_${Date.now()}.mp4`,
    });
    // ...
  };
```

并在 UI 中 `<VideoPreview />` 上方渲染：

```tsx
          <OutputSizeSelector value={outputSize} onChange={setOutputSize} />
```

- [ ] **Step 5: 跑测试验证通过**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm test
```

Expected: 全部测试 PASS。

- [ ] **Step 6: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add app/src/OutputSizeSelector.tsx app/src/OutputSizeSelector.test.tsx app/src/App.tsx
git commit -m "feat: add output size selector with custom resolution"
```

---

### Task 10: 打包 — electron-builder + PyInstaller

**Files:**
- Modify: `app/package.json`（加 build 配置与脚本）
- Create: `app/electron-builder.yml`
- Create: `engine/engine.spec`（PyInstaller 配置）
- Create: `app/build/after-pack.ts`（把引擎二进制拷入 app）
- Modify: `app/electron/main.ts`（引擎路径按打包环境解析）
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1-9 全部产物。
- Produces: macOS `.dmg` / Windows `.exe` 安装包。

- [ ] **Step 1: 配置 electron-builder**

写 `app/electron-builder.yml`：

```yaml
appId: com.vibedance.editor
productName: VibeDance
directories:
  output: release
files:
  - dist/**
  - dist-electron/**
  - build/**
mac:
  target:
    - dmg
win:
  target:
    - nsis
```

- [ ] **Step 2: 配置 PyInstaller 打包引擎**

写 `engine/engine.spec`（在 engine 目录生成引擎独立可执行文件）：

```python
# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec：把引擎打包成单目录可执行文件
a = Analysis(
    ["engine/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vibe_engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="engine_bundle",
)
```

打包命令（macOS）：

```bash
cd /Users/RuzZ/personalspace/vibe_dance/engine
source .venv/bin/activate
pip install pyinstaller
pyinstaller engine.spec --distpath dist --workpath build --noconfirm
```

Expected: `engine/dist/engine_bundle/vibe_engine` 生成。

**注意**：PyInstaller 打包 ultralytics + torch 体积很大（数百 MB）。**MVP 备选方案**：引擎不打包成二进制，而是在安装包内附带 Python venv + 模型文件，Electron 直接 spawn venv 里的 python。此方案在 Task 11 的文档中说明，若体积不可接受则采用。本任务先完成 PyInstaller 可行性验证。

- [ ] **Step 3: 配置打包后复制引擎**

写 `app/build/after-pack.ts`：

```typescript
import { copyFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';

// 把 PyInstaller 产物复制到 app 的 release 资源目录
export default async function afterPack(context: any) {
  const appDir = context.appOutDir;
  const engineSrc = path.join(context.outDir, '..', '..', 'engine', 'dist', 'engine_bundle');
  const engineDest = path.join(appDir, 'engine_bundle');
  mkdirSync(engineDest, { recursive: true });
  copyFileSync(path.join(engineSrc, 'vibe_engine'), path.join(engineDest, 'vibe_engine'));
}
```

- [ ] **Step 4: 更新 main.ts 引擎路径解析**

修改 `app/electron/main.ts` 的 `startEngine()`：

```typescript
function startEngine() {
  let engineCmd: string;
  let engineArgs: string[];
  let engineCwd: string;
  if (app.isPackaged) {
    // 打包后：使用附带的 engine_bundle
    engineCmd = path.join(process.resourcesPath, 'engine_bundle', 'vibe_engine');
    engineArgs = ['8787'];
    engineCwd = path.join(process.resourcesPath, 'engine_bundle');
  } else {
    // 开发时：使用 engine/.venv
    const engineDir = path.join(__dirname, '../../engine');
    engineCmd = path.join(engineDir, '.venv/bin/python');
    engineArgs = ['-m', 'engine', '8787'];
    engineCwd = engineDir;
  }
  engineProc = spawn(engineCmd, engineArgs, { cwd: engineCwd });
  engineProc.on('exit', (code) => {
    console.log(`engine exited with code ${code}`);
    engineProc = null;
  });
}
```

- [ ] **Step 5: 打包并验证**

```bash
cd /Users/RuzZ/personalspace/vibe_dance/app
npm run build
npx electron-builder --mac --dir
```

Expected: `app/release/mac-arm64/` 下生成可运行 `.app`，双击能打开，引擎随包启动。

- [ ] **Step 6: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add app/electron-builder.yml app/build/after-pack.ts app/electron/main.ts app/package.json engine/engine.spec README.md
git commit -m "feat: add packaging config for mac dmg / windows exe"
```

---

### Task 11: 集成测试与验收

**Files:**
- Create: `scripts/e2e_smoke.py`（引擎端到端冒烟）
- Create: `docs/superpowers/plans/README.md`（构建与验收说明）

**Interfaces:**
- Consumes: 全部已完成组件。
- Produces: 一份可重复运行的验收脚本与文档。

- [ ] **Step 1: 写引擎端到端冒烟脚本**

写 `scripts/e2e_smoke.py`：

```python
"""端到端冒烟：合成视频 → 框选 → 跟踪 → 导出，验证全链路。"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

from engine.tests.fixtures import make_synthetic_video  # noqa: E402
from engine.tracker import PersonTracker  # noqa: E402
from engine.render import Renderer  # noqa: E402
from engine.video import VideoReader  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "scene.mp4")
        out = os.path.join(tmp, "out.mp4")
        make_synthetic_video(video, width=640, height=480, fps=15, frames=60)
        reader = VideoReader(video)
        try:
            # 目标：画面右侧移动方块
            tracker = PersonTracker(reader, 0, (300, 100, 80, 140))
            viewport = (300, 100, 80, 142)  # 9:16
            renderer = Renderer(reader, tracker, 0, viewport, (360, 640))
            renderer.render(out, on_progress=lambda p: print(f"progress: {p}%", flush=True))
        finally:
            reader.release()
        assert os.path.exists(out) and os.path.getsize(out) > 0, "输出文件为空"
        print(f"SMOKE OK: {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
```

运行：

```bash
cd /Users/RuzZ/personalspace/vibe_dance
source engine/.venv/bin/activate
python scripts/e2e_smoke.py
```

Expected: 输出进度 0-100%，最后 `SMOKE OK`。

- [ ] **Step 2: 写验收说明文档**

写 `docs/superpowers/plans/README.md`：

```markdown
# 验收清单

## 引擎（无 UI）

```bash
cd engine && source .venv/bin/activate
python -m pytest -v            # 全部通过
python ../scripts/e2e_smoke.py # 输出 SMOKE OK
```

## 桌面应用

```bash
cd app
npm test          # 全部通过
npm run electron:dev  # 手动验证：打开视频 → 框选 → 开始跟踪 → 进度 → 完成
```

## 真实视频验收

1. 准备一段单机位固定拍摄、含目标人物的多人视频（mp4/mov）。
2. 打开应用，导入视频。
3. 用 9:16 取景框框住目标人物。
4. 点「开始跟踪」，等待导出完成。
5. 打开输出视频，确认：目标人物全程在画面内、周围人被裁掉、画面不抖动。

## 已知限制（MVP）

- 固定取景框：人物走近可能出框、走远会被放大。
- 遮挡过久可能丢失，需增强重定位策略。
- 单机位固定拍摄（不支持运镜运动补偿）。
```

- [ ] **Step 3: 提交**

```bash
cd /Users/RuzZ/personalspace/vibe_dance
git add scripts/e2e_smoke.py docs/superpowers/plans/README.md
git commit -m "docs: add e2e smoke script and acceptance checklist"
```
