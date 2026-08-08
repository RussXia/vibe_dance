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


def test_audio_task_submit_query_render(tmp_path):
    """audio-task 全链路：提交对齐 → 查询 → render 混流。"""
    from engine.audiotask import AudioTaskManager
    # 复用 Task 3 的合成函数（若已在 conftest/fixtures，则 import）
    import subprocess as sp
    video_a = str(tmp_path / "a.mp4")
    audio_b = str(tmp_path / "b.m4a")
    out = str(tmp_path / "out.mp4")
    sp.run(["ffmpeg","-y","-f","lavfi","-i","color=c=black:s=160x120:d=2",
            "-f","lavfi","-i","sine=frequency=440:duration=2",
            "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",video_a],
           capture_output=True)
    sp.run(["ffmpeg","-y","-f","lavfi","-i","sine=frequency=300:duration=2",
            "-c:a","aac",audio_b], capture_output=True)

    port = 8893
    base = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=start, kwargs={"port": port}, daemon=True)
    thread.start()
    try:
        _wait_server_ready(base)
        resp = _post(base, "/audio-task", {
            "video_a_path": video_a,
            "audio_b_path": audio_b,
            "output_path": out,
        })
        task_id = resp["task_id"]
        assert resp["status"] == "QUEUED"

        for _ in range(50):
            st = _get(base, f"/audio-task/{task_id}")
            if st["status"] in ("DONE", "FAILED"):
                break
            time.sleep(0.1)
        assert st["status"] == "DONE", st
        assert "align_result" in st
        offset = st["align_result"]["offset_seconds"]

        # render
        r = _post(base, f"/audio-task/{task_id}/render", {"offset_seconds": offset})
        assert r.get("ok") is True, r
        for _ in range(100):
            st2 = _get(base, f"/audio-task/{task_id}")
            if st2["status"] in ("DONE", "FAILED"):
                break
            time.sleep(0.1)
        assert st2["status"] == "DONE", st2
    finally:
        import engine.server as srv
        srv.stop()
