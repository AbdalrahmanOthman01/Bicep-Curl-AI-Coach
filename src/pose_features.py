"""
pose_features.py
----------------
Biomechanical Upper-Body Feature Extraction Module for Bicep Curl Analysis.

Extracts normalized upper-body landmarks and biomechanical joint angles,
omitting all leg dependencies to support bicep curl machines (e.g., seated / preacher curls).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


FEATURE_NAMES: List[str] = [
    'left_elbow_angle',
    'right_elbow_angle',
    'left_shoulder_angle',
    'right_shoulder_angle',
    'left_arm_vertical_angle',
    'right_arm_vertical_angle',
    'left_forearm_vertical_angle',
    'right_forearm_vertical_angle',
    'left_wrist_norm_x',
    'left_wrist_norm_y',
    'right_wrist_norm_x',
    'right_wrist_norm_y',
    'left_elbow_norm_x',
    'left_elbow_norm_y',
    'right_elbow_norm_x',
    'right_elbow_norm_y',
    'shoulder_symmetry',
    'elbow_symmetry',
    'torso_angle',
    'elbow_distance',
    'wrist_distance',
    'left_wrist_to_shoulder_dist',
    'right_wrist_to_shoulder_dist',
]


def calculate_angle(
    a: Union[np.ndarray, List[float]],
    b: Union[np.ndarray, List[float]],
    c: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculate the planar angle (in degrees) at joint vertex b between vectors ba and bc.
    
    Parameters
    ----------
    a : array-like (x, y) - proximal point (e.g. shoulder)
    b : array-like (x, y) - vertex joint point (e.g. elbow)
    c : array-like (x, y) - distal point (e.g. wrist)
    
    Returns
    -------
    angle : float in [0.0, 180.0] degrees
    """
    va = np.array(a[:2], dtype=np.float64) - np.array(b[:2], dtype=np.float64)
    vc = np.array(c[:2], dtype=np.float64) - np.array(b[:2], dtype=np.float64)

    norm_va = np.linalg.norm(va)
    norm_vc = np.linalg.norm(vc)

    if norm_va < 1e-6 or norm_vc < 1e-6:
        return 0.0

    cosine_angle = np.dot(va, vc) / (norm_va * norm_vc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_segment_angle_to_vertical(
    p1: Union[np.ndarray, List[float]],
    p2: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculate inclination angle (in degrees) of directed segment p1->p2 relative to vertical downwards vector (0, 1).
    Used to detect forward/backward arm sway or torso lean.
    """
    vec = np.array(p2[:2], dtype=np.float64) - np.array(p1[:2], dtype=np.float64)
    norm_v = np.linalg.norm(vec)
    if norm_v < 1e-6:
        return 0.0

    vertical = np.array([0.0, 1.0], dtype=np.float64)
    cosine = np.dot(vec, vertical) / norm_v
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute_biomechanical_features(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    left_elbow: np.ndarray,
    right_elbow: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    neck_or_head: Optional[np.ndarray] = None,
    mid_hip: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Pure geometric feature computer used across training and inference.
    All inputs are 2D points (x, y).
    """
    ls = np.array(left_shoulder[:2], dtype=np.float64)
    rs = np.array(right_shoulder[:2], dtype=np.float64)
    le = np.array(left_elbow[:2], dtype=np.float64)
    re = np.array(right_elbow[:2], dtype=np.float64)
    lw = np.array(left_wrist[:2], dtype=np.float64)
    rw = np.array(right_wrist[:2], dtype=np.float64)

    shoulder_center = (ls + rs) / 2.0
    shoulder_width = float(np.linalg.norm(rs - ls))
    if shoulder_width < 1e-5:
        shoulder_width = 1.0  # Safeguard against zero division

    # Joint angles
    l_elbow_ang = calculate_angle(ls, le, lw)
    r_elbow_ang = calculate_angle(rs, re, rw)

    # Shoulder angles (relative to body centerline / shoulder anchor)
    l_shoulder_ang = calculate_angle(shoulder_center, ls, le)
    r_shoulder_ang = calculate_angle(shoulder_center, rs, re)

    # Segment angles relative to vertical
    l_arm_vert = calculate_segment_angle_to_vertical(ls, le)
    r_arm_vert = calculate_segment_angle_to_vertical(rs, re)
    l_forearm_vert = calculate_segment_angle_to_vertical(le, lw)
    r_forearm_vert = calculate_segment_angle_to_vertical(re, rw)

    # Torso orientation: if neck/head or hip provided
    if neck_or_head is not None:
        ref_pt = np.array(neck_or_head[:2], dtype=np.float64)
        torso_angle = calculate_segment_angle_to_vertical(ref_pt, shoulder_center)
    elif mid_hip is not None:
        ref_pt = np.array(mid_hip[:2], dtype=np.float64)
        torso_angle = calculate_segment_angle_to_vertical(shoulder_center, ref_pt)
    else:
        torso_angle = 0.0

    # Normalized positions centered on shoulder midpoint, scaled by shoulder width
    # Invariant to user distance from camera
    l_wrist_norm_x = float((lw[0] - shoulder_center[0]) / shoulder_width)
    l_wrist_norm_y = float((lw[1] - shoulder_center[1]) / shoulder_width)
    r_wrist_norm_x = float((rw[0] - shoulder_center[0]) / shoulder_width)
    r_wrist_norm_y = float((rw[1] - shoulder_center[1]) / shoulder_width)

    l_elbow_norm_x = float((le[0] - shoulder_center[0]) / shoulder_width)
    l_elbow_norm_y = float((le[1] - shoulder_center[1]) / shoulder_width)
    r_elbow_norm_x = float((re[0] - shoulder_center[0]) / shoulder_width)
    r_elbow_norm_y = float((re[1] - shoulder_center[1]) / shoulder_width)

    # Symmetries & distances
    shoulder_symmetry = float(abs(ls[1] - rs[1]) / shoulder_width)
    elbow_symmetry = float(abs(le[1] - re[1]) / shoulder_width)
    elbow_dist = float(np.linalg.norm(re - le) / shoulder_width)
    wrist_dist = float(np.linalg.norm(rw - lw) / shoulder_width)
    l_wrist_to_shoulder = float(np.linalg.norm(lw - ls) / shoulder_width)
    r_wrist_to_shoulder = float(np.linalg.norm(rw - rs) / shoulder_width)

    return {
        'left_elbow_angle': l_elbow_ang,
        'right_elbow_angle': r_elbow_ang,
        'left_shoulder_angle': l_shoulder_ang,
        'right_shoulder_angle': r_shoulder_ang,
        'left_arm_vertical_angle': l_arm_vert,
        'right_arm_vertical_angle': r_arm_vert,
        'left_forearm_vertical_angle': l_forearm_vert,
        'right_forearm_vertical_angle': r_forearm_vert,
        'left_wrist_norm_x': l_wrist_norm_x,
        'left_wrist_norm_y': l_wrist_norm_y,
        'right_wrist_norm_x': r_wrist_norm_x,
        'right_wrist_norm_y': r_wrist_norm_y,
        'left_elbow_norm_x': l_elbow_norm_x,
        'left_elbow_norm_y': l_elbow_norm_y,
        'right_elbow_norm_x': r_elbow_norm_x,
        'right_elbow_norm_y': r_elbow_norm_y,
        'shoulder_symmetry': shoulder_symmetry,
        'elbow_symmetry': elbow_symmetry,
        'torso_angle': torso_angle,
        'elbow_distance': elbow_dist,
        'wrist_distance': wrist_dist,
        'left_wrist_to_shoulder_dist': l_wrist_to_shoulder,
        'right_wrist_to_shoulder_dist': r_wrist_to_shoulder,
    }


def extract_features_from_mediapipe(
    pose_landmarks,
    image_width: float = 1.0,
    image_height: float = 1.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract standardized upper-body feature vector from MediaPipe Pose landmarks.
    
    MediaPipe Pose Landmark indices used:
      0: NOSE
      11: LEFT_SHOULDER
      12: RIGHT_SHOULDER
      13: LEFT_ELBOW
      14: RIGHT_ELBOW
      15: LEFT_WRIST
      16: RIGHT_WRIST
      (No legs required)
      
    Returns
    -------
    (features_array, features_dict)
    """
    landmarks = pose_landmarks.landmark

    # Scale normalized [0, 1] coordinates by aspect ratio/pixel scale
    def pt(idx: int) -> np.ndarray:
        return np.array([landmarks[idx].x * image_width, landmarks[idx].y * image_height], dtype=np.float64)

    nose = pt(0)
    l_shoulder = pt(11)
    r_shoulder = pt(12)
    l_elbow = pt(13)
    r_elbow = pt(14)
    l_wrist = pt(15)
    r_wrist = pt(16)

    # Approximate neck as midpoint between nose and shoulder center
    shoulder_center = (l_shoulder + r_shoulder) / 2.0
    neck = (nose + shoulder_center) / 2.0

    feat_dict = compute_biomechanical_features(
        left_shoulder=l_shoulder,
        right_shoulder=r_shoulder,
        left_elbow=l_elbow,
        right_elbow=r_elbow,
        left_wrist=l_wrist,
        right_wrist=r_wrist,
        neck_or_head=neck
    )

    feat_vec = np.array([feat_dict[col] for col in FEATURE_NAMES], dtype=np.float32)
    return feat_vec, feat_dict


def extract_features_from_dataset_row(row: pd.Series) -> Dict[str, float]:
    """
    Extract upper-body features from a row in training_data_bicep_curls.csv.
    Coordinates match Sports2D landmark output:
      RShoulder: X17, Y17
      RElbow:    X18, Y18
      RWrist:    X19, Y19
      LShoulder: X20, Y20
      LElbow:    X21, Y21
      LWrist:    X22, Y22
      Neck:      X14, Y14
    """
    rs = np.array([row['RShoulder_X17'], row['RShoulder_Y17']])
    re = np.array([row['RElbow_X18'], row['RElbow_Y18']])
    rw = np.array([row['RWrist_X19'], row['RWrist_Y19']])
    ls = np.array([row['LShoulder_X20'], row['LShoulder_Y20']])
    le = np.array([row['LElbow_X21'], row['LElbow_Y21']])
    lw = np.array([row['LWrist_X22'], row['LWrist_Y22']])
    neck = np.array([row['Neck_X14'], row['Neck_Y14']])

    return compute_biomechanical_features(
        left_shoulder=ls,
        right_shoulder=rs,
        left_elbow=le,
        right_elbow=re,
        left_wrist=lw,
        right_wrist=rw,
        neck_or_head=neck
    )
