# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [
    ('models/yolov8n-face.onnx', 'models'),
    ('models/scrfd_2.5g.onnx', 'models'),
    ('models/w600k_mbf.onnx', 'models'),
    ('config.example.json', '.'),
    ('GPU安装教程.md', '.'),
    ('laocao/README.md', 'laocao'),
    ('laocao/0f18b067c807a1bbbcdb0e2450cfd880.jpg', 'laocao'),
    ('laocao/30f42f7cccac4f2003a1071e198de4da.jpg', 'laocao'),
    ('laocao/ref_front_seated.png', 'laocao'),
    ('laocao/ref_front_talking.png', 'laocao'),
    ('laocao/ref_looking_down.png', 'laocao'),
    ('laocao/ref_profile_right.png', 'laocao'),
    ('laocao/ref_three_quarter.png', 'laocao'),
]
binaries = []
hiddenimports = []
for package in ('onnxruntime', 'cv2'):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    ['ui_gpu.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'shiboken5', 'torch', 'torchvision', 'ultralytics',
        'polars', 'faiss', 'transformers', 'matplotlib', 'IPython', 'notebook',
    ],
    noarchive=False,
    optimize=0,
)
_bad_icu = {'icuuc.dll', 'icudt78.dll'}
a.binaries = [item for item in a.binaries if item[0].lower() not in _bad_icu]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LaoCaoMirror-GPU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LaoCaoMirror-GPU',
)
