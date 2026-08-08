# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec：把引擎打包成单目录可执行文件（独立分发，不依赖目标机 Python）。
# 用法（在 engine/ 目录、激活 venv 后）：
#   pyinstaller engine.spec --distpath dist --workpath build --noconfirm
# 产物：engine/dist/engine_bundle/vibe_engine（macOS）/ vibe_engine.exe（Windows）
#
# 入口用 launcher.py（脚本模式 + 绝对导入），避免 __main__.py 的相对导入在冻结时失败。
a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[
        # YOLO 权重随引擎分发，避免运行时联网下载
        ("yolov8n.pt", "."),
    ],
    hiddenimports=[
        # librosa + 其依赖的动态导入模块
        # numba/sklearn/soundfile 在打包时需显式指定以避免 ImportError
        "numba.core.registry",
        "numba.core.typedarray",
        "numba.core.types",
        "sklearn.utils._typedefs",
        "sklearn.utils._heap",
        "sklearn.neighbors._partition_nodes",
        "soundfile",
        "audioread.ffdec",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vibe_engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="engine_bundle",
)
