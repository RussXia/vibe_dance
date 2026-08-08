"""音频替换任务：对齐（DTW/节拍/从头）→ 变速 → 混流导出。

与渲染任务（render.py）共用任务状态机模式：
QUEUED → RUNNING → DONE | FAILED | CANCELLED

两阶段：
1. submit() 只做对齐，返回 align_result + preview（波形/试听音轨）。
2. render() 用用户调整后的 offset（+可选 tempo_ratio）触发混流导出。
"""
from __future__ import annotations

import os
import subprocess
import threading
import uuid
from dataclasses import dataclass, field

from .align import _extract_audio, align_tracks
from .ffmpeg import find_ffmpeg
from .waveform import extract_preview_audio, extract_waveform

_SAMPLE_RATE = 16000


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, **kwargs)


def _work_dir(task_id: str) -> str:
    """任务临时目录（对齐特征/波形/试听音频），随任务生命周期存在。"""
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "vibe_audio", task_id)
    os.makedirs(d, exist_ok=True)
    return d


@dataclass
class AudioTask:
    task_id: str
    video_a_path: str
    audio_b_path: str
    output_path: str
    params: dict
    status: str = "QUEUED"
    progress: int = 0
    message: str = ""
    align_result: dict | None = None
    preview: dict | None = None
    _thread: threading.Thread = field(default=None, repr=False)
    _cancel: bool = field(default=False, repr=False)


class AudioTaskManager:
    def __init__(self):
        self._tasks: dict[str, AudioTask] = {}
        self._lock = threading.Lock()

    def submit(self, video_a_path, audio_b_path, output_path, params=None):
        task_id = uuid.uuid4().hex[:12]
        task = AudioTask(
            task_id=task_id,
            video_a_path=video_a_path,
            audio_b_path=audio_b_path,
            output_path=output_path,
            params=params or {},
        )
        with self._lock:
            self._tasks[task_id] = task
        task._thread = threading.Thread(target=self._run_align, args=(task,), daemon=True)
        task._thread.start()
        return task_id

    def get(self, task_id):
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return None
            info = {
                "task_id": t.task_id,
                "status": t.status,
                "progress": t.progress,
                "message": t.message,
            }
            if t.align_result is not None:
                info["align_result"] = t.align_result
            if t.preview is not None:
                info["preview"] = t.preview
            return info

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

    def render(self, task_id, offset_seconds, tempo_ratio=None):
        """用调整后的 offset 触发混流导出（复用任务线程，不新建任务）。"""
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                raise KeyError(f"task not found: {task_id}")
            if t.status == "CANCELLED":
                raise RuntimeError("task cancelled")
            t._cancel = False
            t.status = "RUNNING"
            t.progress = 0
            t.message = ""
            thread = threading.Thread(
                target=self._run_render,
                args=(t, float(offset_seconds), tempo_ratio),
                daemon=True,
            )
            thread.start()

    def _run_align(self, task):
        work = _work_dir(task.task_id)
        try:
            task.status = "RUNNING"
            task.progress = 5
            # 提取 A/B 的 16kHz wav（对齐特征）
            a_wav = os.path.join(work, "a.wav")
            b_wav = os.path.join(work, "b.wav")
            _extract_audio(task.video_a_path, a_wav)
            task.progress = 25
            _extract_audio(task.audio_b_path, b_wav)
            task.progress = 35
            # 对齐
            result = align_tracks(a_wav, b_wav, task.params)
            task.align_result = result
            task.progress = 60
            # 波形（供前端绘图）
            wf_a = extract_waveform(a_wav)
            wf_b = extract_waveform(b_wav)
            # 试听音轨（供前端 Web Audio）
            audio_a = os.path.join(work, "a_preview.m4a")
            audio_b = os.path.join(work, "b_preview.m4a")
            extract_preview_audio(task.video_a_path, audio_a)
            extract_preview_audio(task.audio_b_path, audio_b)
            task.preview = {
                "video_a_path": task.video_a_path,
                "audio_a_path": audio_a,
                "audio_b_path": audio_b,
                "waveform_a": wf_a,
                "waveform_b": wf_b,
            }
            task.progress = 100
            if task._cancel:
                task.status = "CANCELLED"
            else:
                task.status = "DONE"
                task.message = "对齐完成，可拖动波形微调或直接下载"
        except _Cancelled:
            task.status = "CANCELLED"
        except Exception as exc:  # noqa: BLE001
            task.status = "FAILED"
            task.message = str(exc)

    def _run_render(self, task, offset_seconds, tempo_ratio):
        work = _work_dir(task.task_id)
        try:
            task.status = "RUNNING"
            task.progress = 0
            ffmpeg = find_ffmpeg()
            # 生成变速后的 B 音轨（若有 tempo_ratio）
            b_audio = os.path.join(work, "b_preview.m4a")
            b_varied = os.path.join(work, "b_varied.m4a")
            ratio = float(tempo_ratio) if tempo_ratio else float(
                (task.align_result or {}).get("tempo_ratio", 1.0)
            )
            if abs(ratio - 1.0) < 0.01:
                b_varied = b_audio  # 基本原速：直接用原音轨
            else:
                # atempo 只支持 0.5-2.0，超范围分多段
                seq = _atempo_chain(ratio)
                cmd = [ffmpeg, "-y", "-i", b_audio]
                for a in seq:
                    cmd += ["-af", f"atempo={a}"]
                cmd += ["-c:a", "aac", b_varied]
                _run(cmd, check=True)
            task.progress = 40

            # 混流：A 视频流 + 变速后 B 音轨，从 offset 处开始铺，时长以 A 为准
            cmd = [
                ffmpeg, "-y",
                "-i", task.video_a_path,
                "-i", b_varied,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                # B 音轨整体前移 offset 秒：用 adelay 把 B 起点推迟到 offset
                "-af", f"adelay={int(offset_seconds * 1000)}:all=1",
                task.output_path,
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read() if proc.stderr else b""
                raise RuntimeError(
                    f"混流导出失败: {err[-500:].decode(errors='replace')}"
                )
            task.progress = 100
            if task._cancel:
                task.status = "CANCELLED"
            else:
                task.status = "DONE"
                task.message = f"导出完成: {task.output_path}"
        except _Cancelled:
            task.status = "CANCELLED"
        except Exception as exc:  # noqa: BLE001
            task.status = "FAILED"
            task.message = str(exc)


def _atempo_chain(ratio: float) -> list[str]:
    """把变速比拆成 atempo 可用的链（每段 0.5-2.0）。"""
    seq = []
    r = ratio
    while r > 2.0:
        seq.append("2.0")
        r /= 2.0
    while r < 0.5:
        seq.append("0.5")
        r /= 0.5
    seq.append(f"{r:.4f}")
    return seq


class _Cancelled(Exception):
    pass
