# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

import types
sys.modules['tensorflow'] = None
if 'mediapipe.tasks' not in sys.modules:
    sys.modules['mediapipe.tasks'] = types.ModuleType('mediapipe.tasks')
    sys.modules['mediapipe.tasks.python'] = types.ModuleType('mediapipe.tasks.python')
if 'matplotlib' not in sys.modules:
    m = types.ModuleType('matplotlib')
    m.pyplot = types.ModuleType('matplotlib.pyplot')
    sys.modules['matplotlib'] = m
    sys.modules['matplotlib.pyplot'] = m.pyplot

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
    'onnxruntime',
    'mlflow',
    'pyspark',
    'skops',
    'bokeh',
    'selenium',
    'gradio',
    'gradio_client',
    'uvicorn',
    'dask',
    'polars',
    'sympy',
    'mpmath',
    'networkx',
    'lightgbm',
    'catboost',
    'xgboost',
    'fastapi',
    'starlette',
    'docker',
    'redis',
    'pymongo',
    'alembic',
    'mako',
    'huggingface_hub',
    'cupy',
    'sklearn.externals.array_api_compat.torch',
    'sklearn.externals.array_api_compat.cupy',
    'sklearn.externals.array_api_compat.dask',
    'optuna',
    'ray',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['scripts/rthook_stubs.py'],
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
