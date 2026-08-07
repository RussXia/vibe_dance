"""端到端冒烟：合成视频 → 框选 → 跟踪 → 导出，验证全链路。"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

from tests.fixtures import make_synthetic_video  # noqa: E402
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
