"""
app.py
------
Flask Web Application for Real-Time Biceps Curl Form Analysis and Rep Counting.
"""

import base64
import json
import logging
from pathlib import Path
import sys
import threading
import time

import os
import tempfile
import types

# Safe stdout/stderr redirection if running as a windowed/noconsole GUI app
if sys.stdout is None:
    try:
        sys.stdout = open(os.path.join(tempfile.gettempdir(), 'bicep_curl_app.log'), 'a', encoding='utf-8')
    except Exception:
        sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    try:
        sys.stderr = open(os.path.join(tempfile.gettempdir(), 'bicep_curl_app.log'), 'a', encoding='utf-8')
    except Exception:
        sys.stderr = open(os.devnull, 'w')

# Prevent potential protobuf conflict between tensorflow and mediapipe
sys.modules['tensorflow'] = None

# Stub matplotlib so mediapipe doesn't fail when matplotlib is excluded
if 'matplotlib' not in sys.modules:
    m = types.ModuleType('matplotlib')
    m.pyplot = types.ModuleType('matplotlib.pyplot')
    sys.modules['matplotlib'] = m
    sys.modules['matplotlib.pyplot'] = m.pyplot

import cv2
from flask import Flask, Response, jsonify, render_template, request
import mediapipe.python.solutions.pose as mp_pose
import numpy as np

from src.form_analysis import BicepCurlAnalyzer
from src.pose_features import extract_features_from_mediapipe
from src.predict import get_predictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'templates'),
    static_folder=str(BASE_DIR / 'static')
)

# Initialize singletons at startup
logger.info("Initializing ML inference engine...")
predictor = get_predictor()
analyzer = BicepCurlAnalyzer()

# Initialize MediaPipe Pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
logger.info("MediaPipe Pose initialized successfully.")


@app.route('/')
def index():
    """Render main interactive AI Coach dashboard."""
    metadata_path = BASE_DIR / 'models' / 'model_metadata.json'
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    return render_template('index.html', metadata=metadata)


@app.route('/status', methods=['GET'])
def status():
    """Health check and model status."""
    return jsonify({
        'status': 'healthy',
        'classes': predictor.classes,
        'model_type': type(predictor.model).__name__,
        'smoothing_window': predictor.smoothing_window
    })


@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    """Reset rep counter and temporal buffers."""
    analyzer.reset()
    predictor.reset_history()
    return jsonify({
        'status': 'success',
        'message': 'Workout counter and prediction buffer reset.',
        'rep_count': 0,
        'partial_rep_count': 0
    })


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Real-time inference endpoint for browser webcam frames.
    Expects JSON: { "image": "data:image/jpeg;base64,..." }
    """
    start_time = time.time()
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided in request body.'}), 400

        # Decode base64 image
        encoded_data = data['image']
        if ',' in encoded_data:
            encoded_data = encoded_data.split(',')[1]

        image_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid or corrupted image frame.'}), 400

        h, w, _ = frame.shape

        # MediaPipe Pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)

        if not results.pose_landmarks:
            return jsonify({
                'pose_detected': False,
                'message': 'No person detected. Adjust your position in front of the camera.',
                'rep_count': analyzer.rep_count,
                'partial_rep_count': analyzer.partial_rep_count,
                'state': analyzer.state.value,
                'latency_ms': round((time.time() - start_time) * 1000, 1)
            })

        # Extract upper-body biomechanical features
        feat_vec, feat_dict = extract_features_from_mediapipe(
            results.pose_landmarks,
            image_width=float(w),
            image_height=float(h)
        )

        # ML Model Form Classification
        ml_result = predictor.predict_from_features(feat_dict)

        # Biomechanical State & Rep Counting Analysis
        analysis_result = analyzer.process_frame(feat_dict, ml_prediction=ml_result)

        # Prepare landmark coordinates for frontend canvas rendering
        # Select key upper body landmarks
        key_landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        landmarks_data = []
        for idx in key_landmark_indices:
            lm = results.pose_landmarks.landmark[idx]
            landmarks_data.append({
                'id': idx,
                'x': float(lm.x),
                'y': float(lm.y),
                'z': float(lm.z),
                'visibility': float(lm.visibility)
            })

        latency_ms = round((time.time() - start_time) * 1000, 1)

        return jsonify({
            'pose_detected': True,
            'landmarks': landmarks_data,
            'rep_count': analysis_result['rep_count'],
            'partial_rep_count': analysis_result['partial_rep_count'],
            'state': analysis_result['state'],
            'active_elbow_angle': analysis_result['active_elbow_angle'],
            'left_elbow_angle': analysis_result['left_elbow_angle'],
            'right_elbow_angle': analysis_result['right_elbow_angle'],
            'rep_completed': analysis_result['rep_completed'],
            'partial_completed': analysis_result['partial_completed'],
            'feedback_cues': analysis_result['feedback_cues'],
            'ml_prediction': analysis_result['ml_prediction'],
            'probabilities': ml_result['probabilities'],
            'torso_angle': round(feat_dict.get('torso_angle', 0.0), 1),
            'shoulder_symmetry': round(feat_dict.get('shoulder_symmetry', 0.0), 3),
            'latency_ms': latency_ms
        })

    except Exception as e:
        logger.exception("Error processing frame")
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['GET', 'POST'])
def shutdown():
    """Cleanly exit application process when standalone window is closed."""
    logger.info("Shutdown endpoint called. Exiting application...")
    def exit_soon():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=exit_soon, daemon=True).start()
    return jsonify({"status": "shutting down"})


if __name__ == '__main__':
    logger.info("Starting Bicep Curl AI Coach server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
