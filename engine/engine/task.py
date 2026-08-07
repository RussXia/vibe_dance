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
