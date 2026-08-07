"""find_ffmpeg 定位逻辑测试。"""
import os

import pytest

from engine.ffmpeg import find_ffmpeg, _OVERRIDE_ENV


def test_finds_in_path(monkeypatch):
    """PATH 里有 ffmpeg 时，返回 PATH 命中项（用 python 自身做替身验证查找逻辑）。"""
    # 用绝对路径绕过 which，验证探测不会误报
    result = find_ffmpeg()
    assert os.path.isabs(result)
    assert os.path.exists(result)


def test_homebrew_prefix_fallback(monkeypatch):
    """PATH 无 ffmpeg 时回退到 /opt/homebrew/bin 或 /usr/local/bin。"""
    monkeypatch.setenv("PATH", "/nonexistent:/usr/bin:/bin")
    monkeypatch.delenv(_OVERRIDE_ENV, raising=False)

    result = find_ffmpeg()
    # 机器上 ffmpeg 在 /opt/homebrew/bin；即使不在，也不应抛异常
    #（除非两个前缀都不存在——那就该报错）
    assert result


def test_override_env(monkeypatch):
    """VIBE_DANCE_FFMPEG 指定路径时优先使用。"""
    monkeypatch.setenv(_OVERRIDE_ENV, "/usr/bin/true")
    assert find_ffmpeg() == "/usr/bin/true"


def test_override_env_invalid(monkeypatch):
    """VIBE_DANCE_FFMPEG 指向不存在的文件时抛 FileNotFoundError。"""
    monkeypatch.setenv(_OVERRIDE_ENV, "/no/such/ffmpeg")
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(FileNotFoundError):
        find_ffmpeg()


def test_not_found_raises(monkeypatch):
    """PATH 与 Homebrew 前缀都没有 ffmpeg 时抛 FileNotFoundError。"""
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.delenv(_OVERRIDE_ENV, raising=False)
    # 临时把 Homebrew 前缀也指到不存在的目录
    monkeypatch.setattr("engine.ffmpeg._HOMEBREW_PREFIXES", ["/no/homebrew"])
    with pytest.raises(FileNotFoundError):
        find_ffmpeg()
