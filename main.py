"""
main.py
-------
Desktop launcher for Bicep Curl AI Coach application.
Starts the Flask server, waits for readiness, and automatically opens
the web dashboard in the default browser.
"""

import logging
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request
import webbrowser

# Prevent potential protobuf conflict between tensorflow and mediapipe
sys.modules['tensorflow'] = None

from app import app

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Launcher")

HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def run_server():
    """Run Flask server."""
    # Run with Werkzeug server without debug reloading
    from werkzeug.serving import run_simple
    run_simple(HOST, PORT, app, use_reloader=False, threaded=True)


def wait_and_open_browser():
    """Wait for server to respond to health check and open browser."""
    logger.info(f"Waiting for server at {URL}...")
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{URL}/status", timeout=1) as resp:
                if resp.status == 200:
                    logger.info("Server is up and healthy! Opening browser...")
                    webbrowser.open(URL)
                    return
        except Exception:
            time.sleep(0.5)
    logger.warning("Server startup timeout; opening browser directly.")
    webbrowser.open(URL)


def print_banner():
    banner = f"""
======================================================================
               BICEP CURL AI COACH - DESKTOP RUNNER
======================================================================
  Application URL: {URL}
  Kinematic Model: Random Forest (Upper-Body Machine Curl Mode)
  Pose Tracker:    MediaPipe Pose
  Features:        23 Biomechanical Angles & Normalized Coordinates

  The AI Coach dashboard will automatically launch in your browser.
  To exit, press Ctrl+C in this terminal window.
======================================================================
"""
    print(banner)


def main():
    print_banner()

    # Start browser opener thread
    threading.Thread(target=wait_and_open_browser, daemon=True).start()

    # Run server in main thread
    try:
        run_server()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down Bicep Curl AI Coach. Goodbye!")


if __name__ == '__main__':
    main()
