"""启动引擎 HTTP 服务。用法: python -m engine [port]"""
import sys

from .server import start


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"Engine server listening on 127.0.0.1:{port}")
    start(port)


if __name__ == "__main__":
    main()
