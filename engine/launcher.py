"""PyInstaller 启动入口：把 engine 包（内层）作为独立程序启动。

PyInstaller 以脚本方式运行本文件，无法用 `python -m engine` 的相对导入
（`from .server import`），故这里用绝对导入。运行时把 engine/ 加入
sys.path，使 `import engine` 指向内层包。
"""
import os
import sys

# 把 engine/ 外层目录加入 path，使 `import engine` 解析到 engine/engine/ 包
_ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))  # 即 .../engine/
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)


def main():
    from engine.server import start

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"Engine server listening on 127.0.0.1:{port}", flush=True)
    start(port)


if __name__ == "__main__":
    main()
