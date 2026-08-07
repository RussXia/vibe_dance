"""裁剪渲染器：把跟踪到的目标裁剪到固定输出尺寸并编码。"""
from __future__ import annotations

import subprocess

import cv2

from .ffmpeg import find_ffmpeg


class Renderer:
    """按固定输出框把视频裁剪为输出分辨率，用 FFmpeg 编码为 H.264 MP4。

    性能设计：
    - 用 VideoReader.iter_frames() 顺序解码（一次 read() 流式），避免随机
      访问 set+read 的跳帧解码开销（实测 H.264 上比顺序读慢 60 倍+）。
    - 裁剪帧通过 rawvideo 管道逐帧喂给 FFmpeg stdin，不落 PNG 序列，
      避免磁盘写满，且天然无断号丢帧问题。
    """

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
        # 跟踪阶段进度映射到 0-70%（跟踪是耗时大头，UI 需要看到进度而不是卡 0%）。
        # 用浮点保留小数，避免取整让早期进度显示 0。
        def track_progress(p):
            on_progress(round(float(p) * 0.7, 1))
        track_cb = track_progress if on_progress is not None else None
        boxes = self._tracker.track(on_progress=track_cb)
        smoothed = self._smooth(boxes)
        out_w, out_h = self._output_size
        total = len(smoothed)
        if total == 0:
            raise RuntimeError("没有可渲染的帧")

        # 回退到最后可信位置（spec §8）：跟踪丢失/None 帧时用最近可信框
        # 代替，保证输出帧数 = 输入帧数。
        last_good_box = list(self._viewport)  # 初始取景框即可信框

        # rawvideo 管道：逐帧写 BGR 原始帧给 FFmpeg，编码为 H.264 MP4。
        # 同时把原视频作为第二输入，-map 复制其音轨（时长与视频一致，直接 copy 无需重编码）。
        # ffmpeg 用绝对路径：Finder 双击启动的 App 继承最小 PATH，不含 Homebrew 目录。
        ffmpeg = find_ffmpeg()
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-framerate", str(self._reader.fps),
            "-i", "-",
            "-i", self._reader.path,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-frames:v", str(total),
            "-shortest",
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

        written = 0
        try:
            for i, (frame, box) in enumerate(zip(self._reader.iter_frames(self._init_frame), smoothed)):
                if frame is None:
                    break
                if box is not None:
                    last_good_box = [int(v) for v in box]
                crop = self._crop_centered(frame, last_good_box)
                resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                # BGR uint8 → 管道；resized 是 C 连续数组，直接 tobytes
                assert proc.stdin is not None
                proc.stdin.write(resized.tobytes())
                written += 1
                if on_progress is not None and (i + 1) % max(1, total // 20) == 0:
                    # 渲染阶段映射到 70-100%
                    on_progress(round(70 + (i + 1) / total * 30, 1))
        except BrokenPipeError:
            # FFmpeg 提前退出（如输出路径不可写）：写入中断，跳到错误处理
            pass
        finally:
            if proc.stdin is not None:
                # close() 在 BrokenPipeError 后再抛会把主异常堆栈覆盖成
                # "Broken pipe"，用 try/except 保留真实错误来源。
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            # 失败/取消路径也要回收 FFmpeg 子进程，避免短暂孤儿
            proc.wait()

        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.returncode
        if returncode != 0:
            raise RuntimeError(f"FFmpeg 编码失败: {stderr[-500:].decode(errors='replace')}")
        if written != total:
            raise RuntimeError(f"实际写入 {written} 帧，预期 {total} 帧")
        if on_progress is not None:
            on_progress(100)
