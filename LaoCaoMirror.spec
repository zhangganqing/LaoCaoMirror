# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')   # cv2 5.0 的 loader 文件布局 PyInstaller hook 覆盖不到, 全量收集
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')   # nms 等算子在 torchvision._C 扩展里, 必须完整收集
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['ui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'shiboken5',
              # 以下为 collect_all(ultralytics) 连带拖入但本项目不用的库
              # 注意: sympy/networkx/mpmath 是 torch 运行时依赖, 不能排除!
              'polars', 'faiss', 'transformers', 'jedi', 'av',
              'matplotlib', 'IPython', 'notebook',
              'torch.utils.tensorboard', 'tensorboard'],
    noarchive=False,
    optimize=0,
)
# Qt6Core 在 Windows 上使用系统 ICU 接口。若构建环境的 PATH 中碰巧有
# Poppler 自带的 icuuc.dll，PyInstaller 会误收进根目录并抢先加载，导致
# PyQt6.QtCore 报“找不到指定的程序”。明确排除这组非 Qt 运行库。
_bad_icu = {'icuuc.dll', 'icudt78.dll'}
a.binaries = [item for item in a.binaries if item[0].lower() not in _bad_icu]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LaoCaoMirror',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
if not os.environ.get('LCSB_EXE_ONLY'):
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='LaoCaoMirror',
    )
