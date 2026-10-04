"""
rthook_stubs.py
---------------
PyInstaller runtime hook executed before any module is loaded.
Provides required stubs to allow MediaPipe solutions (Pose) to run
without pulling in heavy/colliding optional packages (tensorflow, tasks, matplotlib).
"""

import os
import sys
import tempfile
import types

# 1. Safe stdout/stderr redirection for GUI mode (console=False)
log_file = os.path.join(tempfile.gettempdir(), 'bicep_curl_app.log')
if sys.stdout is None:
    try:
        sys.stdout = open(log_file, 'a', encoding='utf-8')
    except Exception:
        sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    try:
        sys.stderr = open(log_file, 'a', encoding='utf-8')
    except Exception:
        sys.stderr = open(os.devnull, 'w')

# 2. Block tensorflow to avoid protobuf version collisions with mediapipe
sys.modules['tensorflow'] = None

# 3. Stub mediapipe.tasks so mediapipe/__init__.py line 17 doesn't fail
if 'mediapipe.tasks' not in sys.modules:
    tasks_mod = types.ModuleType('mediapipe.tasks')
    tasks_py_mod = types.ModuleType('mediapipe.tasks.python')
    sys.modules['mediapipe.tasks'] = tasks_mod
    sys.modules['mediapipe.tasks.python'] = tasks_py_mod

# 4. Stub matplotlib so mediapipe drawing utilities don't fail
if 'matplotlib' not in sys.modules:
    mat_mod = types.ModuleType('matplotlib')
    mat_plt = types.ModuleType('matplotlib.pyplot')
    sys.modules['matplotlib'] = mat_mod
    sys.modules['matplotlib.pyplot'] = mat_plt
