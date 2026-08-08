"""音频任务（对齐 + 混流导出）测试。"""
import os
import subprocess

from engine.audiotask import AudioTaskManager


def _make_video_with_audio(path, freq=440, duration=2):
    """生成 2 秒带音的视频。"""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
        capture_output=True, check=True,
    )


def _make_audio(path, freq=440, duration=2):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"sine=frequency={freq}:duration={duration}",
         "-c:a", "aac", path],
        capture_output=True, check=True,
    )


def test_submit_align_then_render(tmp_path):
    """提交 → 对齐 DONE（带 align_result + preview）→ render 混流输出。"""
    video_a = str(tmp_path / "a.mp4")
    audio_b = str(tmp_path / "b.m4a")
    out = str(tmp_path / "out.mp4")
    _make_video_with_audio(video_a)
    _make_audio(audio_b)

    mgr = AudioTaskManager()
    task_id = mgr.submit(video_a, audio_b, out)
    # 轮询直到 DONE（对齐阶段）
    for _ in range(50):
        st = mgr.get(task_id)
        if st["status"] in ("DONE", "FAILED"):
            break
        import time
        time.sleep(0.1)
    assert st["status"] == "DONE", st
    assert "align_result" in st, st
    assert st["align_result"]["method"] in ("dtw", "beat", "zero")
    assert "preview" in st, st
    assert st["preview"]["audio_a_path"] and st["preview"]["audio_b_path"]
    assert st["preview"]["waveform_a"] and st["preview"]["waveform_b"]
    assert st["progress"] == 100

    # render 阶段：用自动对齐的 offset 混流
    offset = st["align_result"]["offset_seconds"]
    mgr.render(task_id, offset)
    for _ in range(100):
        st2 = mgr.get(task_id)
        if st2["status"] in ("DONE", "FAILED"):
            break
        import time
        time.sleep(0.1)
    assert st2["status"] == "DONE", st2
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_submit_missing_audio_raises(tmp_path):
    """B 不是有效音频 → 任务 FAILED。"""
    video_a = str(tmp_path / "a.mp4")
    bad_b = str(tmp_path / "b.txt")
    out = str(tmp_path / "out.mp4")
    _make_video_with_audio(video_a)
    with open(bad_b, "w") as f:
        f.write("not audio")

    mgr = AudioTaskManager()
    task_id = mgr.submit(video_a, bad_b, out)
    for _ in range(50):
        st = mgr.get(task_id)
        if st["status"] in ("DONE", "FAILED"):
            break
        import time
        time.sleep(0.1)
    assert st["status"] == "FAILED", st
    assert st["message"]
