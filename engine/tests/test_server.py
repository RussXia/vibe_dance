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
