"""
main.py
-------
Desktop launcher for Bicep Curl AI Coach application.
Launches Flask server and opens a dedicated, borderless standalone desktop window
(App Mode with Edge or Chrome) without command prompt or browser URL/tab bars.
"""

import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.request
import webbrowser

# Safe stdout/stderr redirection if packaged with console=False
LOG_FILE = os.path.join(tempfile.gettempdir(), 'bicep_curl_app.log')
if sys.stdout is None:
    try:
        sys.stdout = open(LOG_FILE, 'a', encoding='utf-8')
    except Exception:
        sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    try:
        sys.stderr = open(LOG_FILE, 'a', encoding='utf-8')
    except Exception:
        sys.stderr = open(os.devnull, 'w')

# Prevent protobuf conflict between tensorflow and mediapipe
sys.modules['tensorflow'] = None

# Stub matplotlib so mediapipe doesn't fail when matplotlib is excluded
if 'matplotlib' not in sys.modules:
    m = types.ModuleType('matplotlib')
    m.pyplot = types.ModuleType('matplotlib.pyplot')
    sys.modules['matplotlib'] = m
    sys.modules['matplotlib.pyplot'] = m.pyplot

from app import app

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("Launcher")

HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def find_standalone_browser():
    """Locate Microsoft Edge or Google Chrome to launch in App Mode."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def run_server():
    """Run Flask server via Werkzeug."""
    from werkzeug.serving import run_simple
    run_simple(HOST, PORT, app, use_reloader=False, threaded=True)


def wait_for_server(timeout=20):
    """Wait until Flask responds with 200 OK."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(f"{URL}/status", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    logger.info("Starting Bicep Curl AI Coach desktop application...")

    # Start Flask server on a background daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to be responsive
    ready = wait_for_server(timeout=25)
    if not ready:
        logger.warning("Server took too long to respond; proceeding to open window.")

    # Try launching in Standalone App Mode (Edge or Chrome)
    browser_exe = find_standalone_browser()

    if browser_exe:
        logger.info(f"Launching standalone app window via {browser_exe}")
        profile_dir = os.path.join(tempfile.gettempdir(), 'bicep_curl_browser_profile')
        cmd = [
            browser_exe,
            f"--app={URL}",
            f"--user-data-dir={profile_dir}",
            "--window-size=1300,880",
            "--window-position=80,40",
            "--no-first-run",
            "--no-default-browser-check"
        ]
        try:
            subprocess.Popen(cmd)
            logger.info("Standalone app window spawned successfully.")
        except Exception as e:
            logger.error(f"Failed to launch standalone window: {e}")
            webbrowser.open(URL)
    else:
        logger.info("Opening URL in standard browser as fallback...")
        webbrowser.open(URL)

    # Keep application alive until /shutdown is triggered or user closes window
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application exiting.")


if __name__ == '__main__':
    main()
