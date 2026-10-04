"""
form_analysis.py
----------------
Biomechanical Rule-Based Analysis and Repetition Counter for Bicep Curls.

Designed specifically for Bicep Curl Machine / Strict Arm Isolation:
1. Finite State Machine for Repetition Counting (avoiding double counts).
2. Range of motion (ROM) verification (peak contraction & complete extension).
3. Elbow stability & pad contact monitoring.
4. Shoulder shrugging & symmetry analysis.
5. Clear separation between ML model classification and rule-based kinematic feedback.
"""

from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CurlState(str, Enum):
    EXTENDED = "EXTENDED"          # Bottom position (Elbow angle >= 140°)
    CURLING_UP = "CURLING_UP"      # Concentric phase (Angle decreasing)
    PEAK = "PEAK"                  # Top contraction (Elbow angle <= 55°)
    LOWERING = "LOWERING"          # Eccentric phase (Angle increasing)


class BicepCurlAnalyzer:
    """
    Analyzes biomechanics, tracks repetition states, and produces coaching feedback.
    """

    def __init__(
        self,
        full_extension_threshold: float = 140.0,
        full_contraction_threshold: float = 55.0,
        partial_rep_threshold: float = 75.0,
        angle_smoothing_window: int = 5
    ):
        self.full_extension_threshold = full_extension_threshold
        self.full_contraction_threshold = full_contraction_threshold
        self.partial_rep_threshold = partial_rep_threshold

        # Smoothing buffers for joint angles
        self.left_angle_history: deque = deque(maxlen=angle_smoothing_window)
        self.right_angle_history: deque = deque(maxlen=angle_smoothing_window)

        # Rep counter states
        self.state: CurlState = CurlState.EXTENDED
        self.rep_count: int = 0
        self.partial_rep_count: int = 0
        self.min_angle_in_current_rep: float = 180.0
        self.rep_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset rep count and state history."""
        self.state = CurlState.EXTENDED
        self.rep_count = 0
        self.partial_rep_count = 0
        self.min_angle_in_current_rep = 180.0
        self.left_angle_history.clear()
        self.right_angle_history.clear()
        self.rep_history.clear()

    def process_frame(
        self,
        features: Dict[str, float],
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate single frame features, update rep state, and return feedback cues.
        """
        # Smooth elbow angles
        raw_l_elbow = features.get('left_elbow_angle', 160.0)
        raw_r_elbow = features.get('right_elbow_angle', 160.0)

        self.left_angle_history.append(raw_l_elbow)
        self.right_angle_history.append(raw_r_elbow)

        smooth_l_elbow = float(np.median(self.left_angle_history))
        smooth_r_elbow = float(np.median(self.right_angle_history))

        # Use primary active arm or average of both
        active_elbow_angle = (smooth_l_elbow + smooth_r_elbow) / 2.0

        # State machine for rep counting
        rep_completed_this_frame = False
        partial_completed_this_frame = False

        if self.state == CurlState.EXTENDED:
            self.min_angle_in_current_rep = active_elbow_angle
            if active_elbow_angle < (self.full_extension_threshold - 15.0):
                self.state = CurlState.CURLING_UP

        elif self.state == CurlState.CURLING_UP:
            if active_elbow_angle < self.min_angle_in_current_rep:
                self.min_angle_in_current_rep = active_elbow_angle

            if active_elbow_angle <= self.full_contraction_threshold:
                self.state = CurlState.PEAK
            elif active_elbow_angle > self.min_angle_in_current_rep + 20.0:
                # Started lowering before reaching peak
                self.state = CurlState.LOWERING

        elif self.state == CurlState.PEAK:
            if active_elbow_angle < self.min_angle_in_current_rep:
                self.min_angle_in_current_rep = active_elbow_angle

            if active_elbow_angle > (self.full_contraction_threshold + 15.0):
                self.state = CurlState.LOWERING

        elif self.state == CurlState.LOWERING:
            if active_elbow_angle >= self.full_extension_threshold:
                # Reached bottom extension
                if self.min_angle_in_current_rep <= self.full_contraction_threshold:
                    self.rep_count += 1
                    rep_completed_this_frame = True
                    self.rep_history.append({
                        'rep': self.rep_count,
                        'type': 'FULL',
                        'min_angle': self.min_angle_in_current_rep
                    })
                elif self.min_angle_in_current_rep <= self.partial_rep_threshold:
                    self.partial_rep_count += 1
                    partial_completed_this_frame = True
                    self.rep_history.append({
                        'rep': self.rep_count + self.partial_rep_count,
                        'type': 'PARTIAL',
                        'min_angle': self.min_angle_in_current_rep
                    })

                self.state = CurlState.EXTENDED
                self.min_angle_in_current_rep = 180.0

        # Rule-based biomechanical form cues
        feedback_cues: List[Dict[str, str]] = []

        # 1. Range of Motion Feedback
        if self.state == CurlState.CURLING_UP and active_elbow_angle > self.partial_rep_threshold:
            feedback_cues.append({
                'type': 'info',
                'message': 'Curling up... Drive wrists toward shoulders'
            })
        elif self.state == CurlState.PEAK:
            feedback_cues.append({
                'type': 'success',
                'message': '✓ Excellent peak contraction! Squeeze biceps'
            })
        elif self.state == CurlState.LOWERING:
            if active_elbow_angle < (self.full_extension_threshold - 20.0):
                feedback_cues.append({
                    'type': 'info',
                    'message': 'Lower smoothly to full extension'
                })
            else:
                feedback_cues.append({
                    'type': 'success',
                    'message': '✓ Full stretch reached at bottom'
                })

        # 2. Incomplete ROM Warning
        if self.state == CurlState.LOWERING and self.min_angle_in_current_rep > self.partial_rep_threshold:
            feedback_cues.append({
                'type': 'warning',
                'message': f'⚠ Partial rep: Peak was {self.min_angle_in_current_rep:.0f}° (target < {self.full_contraction_threshold:.0f}°)'
            })

        # 3. Upper-arm stability & Elbow drift (Machine Curl Isolation)
        l_arm_vert = features.get('left_arm_vertical_angle', 0.0)
        r_arm_vert = features.get('right_arm_vertical_angle', 0.0)
        avg_arm_vert = (l_arm_vert + r_arm_vert) / 2.0

        if avg_arm_vert > 65.0:
            feedback_cues.append({
                'type': 'warning',
                'message': '⚠ Elbows drifting forward - keep elbows anchored on machine pad'
            })
        else:
            feedback_cues.append({
                'type': 'success',
                'message': '✓ Stable elbow positioning'
            })

        # 4. Shoulder & Torso Stability
        torso_angle = features.get('torso_angle', 0.0)
        if torso_angle > 28.0:
            feedback_cues.append({
                'type': 'warning',
                'message': '⚠ Excessive torso lean - avoid swinging body'
            })

        shoulder_sym = features.get('shoulder_symmetry', 0.0)
        if shoulder_sym > 0.18:
            feedback_cues.append({
                'type': 'warning',
                'message': '⚠ Shoulder asymmetry detected - keep shoulders level'
            })

        # Combine ML prediction status
        ml_display = ml_prediction.get('display_label', 'EVALUATING') if ml_prediction else 'EVALUATING'
        ml_conf = ml_prediction.get('confidence', 0.0) if ml_prediction else 0.0
        ml_raw = ml_prediction.get('smoothed_raw_label', 'Correct') if ml_prediction else 'Correct'

        return {
            'state': self.state.value,
            'rep_count': self.rep_count,
            'partial_rep_count': self.partial_rep_count,
            'active_elbow_angle': round(active_elbow_angle, 1),
            'left_elbow_angle': round(smooth_l_elbow, 1),
            'right_elbow_angle': round(smooth_r_elbow, 1),
            'rep_completed': rep_completed_this_frame,
            'partial_completed': partial_completed_this_frame,
            'feedback_cues': feedback_cues,
            'ml_prediction': {
                'display': ml_display,
                'raw': ml_raw,
                'confidence_pct': round(ml_conf * 100.0, 1),
            }
        }
