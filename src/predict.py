"""
predict.py
----------
Inference module for Bicep Curl Form Detection.
Loads trained Random Forest model, scaler, and label encoder once.
Provides single-frame prediction and smoothed temporal predictions.
"""

from collections import deque
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np

from src.pose_features import FEATURE_NAMES, extract_features_from_mediapipe


if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent


class BicepCurlPredictor:
    """
    Inference engine that evaluates upper-body bicep curl mechanics.
    """

    LABEL_DISPLAY_NAMES = {
        'Correct': 'CORRECT FORM',
        'Partial': 'PARTIAL REP (INCOMPLETE ROM)',
        'Leg_Drive': 'EXCESSIVE MOMENTUM / SWING'
    }

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        encoder_path: Optional[Union[str, Path]] = None,
        smoothing_window: int = 7,
        confidence_threshold: float = 0.55
    ):
        self.model_path = Path(model_path) if model_path else (BASE_DIR / 'models' / 'bicep_curl_model.pkl')
        self.scaler_path = Path(scaler_path) if scaler_path else (BASE_DIR / 'models' / 'scaler.pkl')
        self.encoder_path = Path(encoder_path) if encoder_path else (BASE_DIR / 'models' / 'label_encoder.pkl')
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

        # Load artifacts once
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.encoder_path)
        self.classes = list(self.label_encoder.classes_)

        # Temporal smoothing buffers
        self.prediction_history: deque = deque(maxlen=self.smoothing_window)
        self.prob_history: deque = deque(maxlen=self.smoothing_window)

    def reset_history(self) -> None:
        """Reset temporal smoothing buffer."""
        self.prediction_history.clear()
        self.prob_history.clear()

    def predict_from_features(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict curl form from an extracted feature dictionary.
        
        Parameters
        ----------
        features_dict : dict containing the 23 standard upper-body features.
        
        Returns
        -------
        dict with raw_label, display_label, confidence, probabilities, smoothed_label
        """
        # Ensure exact feature order as 2D numpy array
        feature_vector = np.array(
            [[features_dict[name] for name in FEATURE_NAMES]],
            dtype=np.float32
        )

        # Apply same scaling used in training
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scaled_vector = self.scaler.transform(feature_vector)

        # Probabilities & raw prediction
        probs = self.model.predict_proba(scaled_vector)[0]
        raw_pred_idx = int(np.argmax(probs))
        raw_label = str(self.classes[raw_pred_idx])
        confidence = float(probs[raw_pred_idx])

        # Temporal smoothing via moving average of probabilities
        self.prob_history.append(probs)
        self.prediction_history.append(raw_label)

        avg_probs = np.mean(self.prob_history, axis=0)
        smoothed_pred_idx = int(np.argmax(avg_probs))
        smoothed_raw_label = str(self.classes[smoothed_pred_idx])
        smoothed_conf = float(avg_probs[smoothed_pred_idx])

        prob_dict = {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
        }

        display_label = self.LABEL_DISPLAY_NAMES.get(smoothed_raw_label, smoothed_raw_label)

        return {
            'raw_label': raw_label,
            'smoothed_raw_label': smoothed_raw_label,
            'display_label': display_label,
            'confidence': smoothed_conf,
            'instant_confidence': confidence,
            'probabilities': prob_dict,
            'features': features_dict
        }

    def predict_from_mediapipe(
        self,
        pose_landmarks,
        image_width: float = 1.0,
        image_height: float = 1.0
    ) -> Dict[str, Any]:
        """
        Full inference from MediaPipe Pose landmarks object.
        """
        _, feat_dict = extract_features_from_mediapipe(
            pose_landmarks,
            image_width=image_width,
            image_height=image_height
        )
        return self.predict_from_features(feat_dict)


# Global singleton instance for Flask
_global_predictor: Optional[BicepCurlPredictor] = None


def get_predictor() -> BicepCurlPredictor:
    """Retrieve or initialize the global inference model instance."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = BicepCurlPredictor()
    return _global_predictor
