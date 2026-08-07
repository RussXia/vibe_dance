import subprocess

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


def add_audio_track(video_path, audio_path, duration):
    """用 ffmpeg 给视频加上一段正弦波音轨，返回新视频路径。"""
    # 先生成正弦波音频
    cmd_audio = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=%d" % duration,
        "-c:a", "aac", "-b:a", "64k",
        audio_path,
    ]
    subprocess.run(cmd_audio, capture_output=True, check=True)
    # 合并音频到视频
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        video_path + ".withaudio.mp4",
    ]
    subprocess.run(cmd_mux, capture_output=True, check=True)
    return video_path + ".withaudio.mp4"
