"""轻量 HTTP 服务：Task API。用标准库实现，避免额外依赖。"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .task import TaskManager
from .audiotask import AudioTaskManager

PORT = 8787

_manager = TaskManager()
_audio_manager = AudioTaskManager()
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
        if self.path.startswith("/audio-task/"):
            task_id = self.path.split("/")[-1]
            info = _audio_manager.get(task_id)
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
        if self.path == "/audio-task":
            payload = self._read_json()
            try:
                task_id = _audio_manager.submit(
                    payload["video_a_path"],
                    payload["audio_b_path"],
                    payload["output_path"],
                    payload.get("params"),
                )
            except KeyError as exc:
                self._send_json(400, {"error": f"missing field: {exc}"})
                return
            self._send_json(200, {"task_id": task_id, "status": "QUEUED"})
            return
        if self.path.startswith("/audio-task/") and self.path.endswith("/render"):
            task_id = self.path.split("/")[-2]
            payload = self._read_json()
            try:
                _audio_manager.render(
                    task_id,
                    payload["offset_seconds"],
                    payload.get("tempo_ratio"),
                    payload.get("output_path"),
                )
            except (KeyError, RuntimeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, {"ok": True})
            return
        if self.path.startswith("/audio-task/") and self.path.endswith("/cancel"):
            task_id = self.path.split("/")[-2]
            ok = _audio_manager.cancel(task_id)
            self._send_json(200, {"cancelled": ok})
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
