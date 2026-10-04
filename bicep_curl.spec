# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

sys.modules['tensorflow'] = None

block_cipher = None

# Collect MediaPipe data files (.binarypb, .tflite, .dll)
mediapipe_datas = collect_data_files('mediapipe')

added_datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('models', 'models'),
] + mediapipe_datas

# Hidden imports needed by scikit-learn and mediapipe
hidden_imports = [
    'mediapipe',
    'mediapipe.python.solutions.pose',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.preprocessing',
    'sklearn.preprocessing._data',
    'sklearn.utils._typedefs',
    'joblib',
    'cv2',
    'numpy',
    'flask',
    'werkzeug',
]

# Exclude unnecessary heavy packages to keep build ultra fast and .exe compact
excluded_modules = [
    'pandas',
    'scipy',
    'jax',
    'jaxlib',
    'mediapipe.tasks',
    'tensorflow',
    'tensorboard',
    'keras',
    'torch',
    'torchvision',
    'torchaudio',
    'transformers',
    'spacy',
    'thinc',
    'timm',
    'nltk',
    'av',
    'pydub',
    'pdfminer',
    'pypdfium2',
    'datasets',
    'matplotlib',
    'seaborn',
    'plotly',
    'skimage',
    'statsmodels',
    'altair',
    'duckdb',
    'pyarrow',
    'fastparquet',
    'h5py',
    'sqlalchemy',
    'lxml',
    'openpyxl',
    'IPython',
    'pytest',
    'jupyter',
    'notebook',
    'PIL.ImageTk',
    'tkinter',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BicepCurlAICoach',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
