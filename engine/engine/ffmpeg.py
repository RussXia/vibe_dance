"""FFmpeg 可执行文件定位。

打包后的 App 经 Finder 双击启动时，继承的是 launchd 的最小 PATH
（仅 /usr/bin:/bin:/usr/sbin:/sbin），不含 Homebrew 的 /opt/homebrew/bin
或 /usr/local/bin，导致 spawn("ffmpeg") 报 No such file or directory。
本模块在 PATH 之外再探测常见安装位置，返回 ffmpeg 绝对路径。
"""
from __future__ import annotations

import shutil

# Finder/launchd 环境不包含的常见 Homebrew 安装前缀（按优先级）
_HOMEBREW_PREFIXES = [
    "/opt/homebrew/bin",  # Apple Silicon
    "/usr/local/bin",     # Intel
]

# 允许自定义（测试/特殊环境用），优先于 PATH
_OVERRIDE_ENV = "VIBE_DANCE_FFMPEG"


def find_ffmpeg() -> str:
    """返回 ffmpeg 可执行文件绝对路径；找不到抛 FileNotFoundError。"""
    # 1. 显式环境变量覆盖（排错/CI 指定用）
    override = __import__("os").environ.get(_OVERRIDE_ENV)
    if override:
        if shutil.which(override):
            return override
        raise FileNotFoundError(
            f"环境变量 {_OVERRIDE_ENV} 指定的 ffmpeg 不存在: {override}"
        )

    # 2. PATH 中查找
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 3. 常见 Homebrew 前缀（Finder 启动的 App 不含这些目录）
    for prefix in _HOMEBREW_PREFIXES:
        candidate = f"{prefix}/ffmpeg"
        if shutil.which(candidate):
            return candidate

    raise FileNotFoundError(
        "找不到 ffmpeg。请安装 FFmpeg（如 `brew install ffmpeg`），"
        "或通过环境变量 VIBE_DANCE_FFMPEG 指定其绝对路径。"
    )
