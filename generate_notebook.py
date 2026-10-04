"""
generate_notebook.py
Generates the complete, fully executable Jupyter Notebook:
notebook/bicep_curl_training.ipynb
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

nb = {
    "cells": [],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Section 1: Overview
add_md("""# Bicep Curl Form Analysis and Kinematic Classification System
### End-to-End Machine Learning & Biomechanical Analysis Pipeline
**Dataset:** Kaggle `pr4nut/bicep-curl-train`  
**Target Application:** Bicep Curl Machine & Strict Arm Isolation (Zero Leg Dependencies)

---
### System Overview
This notebook presents the complete machine learning and computer vision pipeline for detecting and analyzing biceps curls:
1. **Dataset Inspection:** Parsing kinematic 3D landmarks and segment angles.
2. **Machine Curl Adaptation:** Explicit elimination of lower-body features to match bicep curl machine mechanics.
3. **Feature Engineering:** Calculation of joint flexion angles, segment orientations to vertical, and normalized coordinates.
4. **Data Quality & Split:** Verification of completeness, stratified 70% / 15% / 15% train/validation/test split.
5. **Model Benchmarking:** Comparing Logistic Regression, SVM (RBF), Random Forest, XGBoost, and MLP Neural Networks.
6. **Hyperparameter Tuning:** GridSearchCV optimization on the top candidate.
7. **Model Serialization:** Exporting model, scaler, and label encoder for deployment in the Flask web app.
""")

# Section 1: Imports
add_md("""## 1. Imports
Importing necessary libraries for numerical analysis, machine learning, computer vision, and visualization.
""")
add_code("""import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Prevent protobuf conflict between tensorflow and mediapipe
sys.modules['tensorflow'] = None

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
print('All libraries imported successfully.')
""")

# Section 2: Configuration
add_md("""## 2. Configuration
Setting up file paths and reproducibility constants.
""")
add_code("""PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebook' else Path('.').resolve()
DATA_RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'training_data_bicep_curls.csv'
DATA_PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'bicep_curl_landmarks.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print('Project Root:', PROJECT_ROOT)
print('Raw Data Path:', DATA_RAW_PATH)
print('Models Dir:', MODELS_DIR)
""")

# Section 3: Dataset Inspection
add_md("""## 3. Dataset Inspection
We load the raw Kaggle dataset `training_data_bicep_curls.csv` to inspect its structure, dimensions, class distributions, and data types.

> **Dataset Structure & Labels:**
> - Total samples: **1,425 frames** across **24 distinct video sequences**.
> - Ground truth labels:
>   * `Correct` (483 frames, 7 videos): Full range of motion, proper form.
>   * `Partial` (377 frames, 7 videos): Incomplete range of motion (failure to reach peak contraction or full extension).
>   * `Leg_Drive` (565 frames, 10 videos): Momentum / cheat curls in standing context.
>
> **Machine Curl Requirement:**
> When the user exercises on a **Bicep Curl Machine** (e.g. preacher bench or seated machine), legs are stationary and do not drive the movement. Therefore, **all leg landmarks and angles (hips, knees, ankles, toes, shanks, thighs) are excluded from feature engineering**.
""")
add_code("""df_raw = pd.read_csv(DATA_RAW_PATH)
print(f'Raw Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns')
print('\\nClass Distribution:')
print(df_raw['label'].value_counts())
print('\\nUnique Video Sequences:', df_raw['filename'].nunique())
""")

# Section 4: Dataset Visualization
add_md("""## 4. Dataset Visualization
Visualizing the class distribution and video sequence distribution.
""")
add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette = {'Correct': '#10b981', 'Partial': '#f59e0b', 'Leg_Drive': '#ef4444'}

sns.countplot(data=df_raw, x='label', ax=axes[0], palette=palette)
axes[0].set_title('Class Distribution (Total Frames = 1,425)')
axes[0].set_xlabel('Exercise Form Class')
axes[0].set_ylabel('Number of Frames')
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')

video_counts = df_raw.groupby('label')['filename'].nunique().reset_index()
sns.barplot(data=video_counts, x='label', y='filename', ax=axes[1], palette=palette)
axes[1].set_title('Unique Video Sequences per Class (Total = 24 Videos)')
axes[1].set_xlabel('Exercise Form Class')
axes[1].set_ylabel('Number of Unique Videos')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
""")

# Section 5: MediaPipe Pose Initialization
add_md("""## 5. MediaPipe Pose Initialization
Verifying MediaPipe Pose pipeline for real-time landmark estimation.
""")
add_code("""mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print('MediaPipe Pose initialized successfully.')
""")

# Section 6 & 7: Feature Engineering
add_md("""## 6 & 7. Landmark Extraction & Biomechanical Feature Engineering
We implement standardized geometric feature calculation functions:
- `calculate_angle(a, b, c)`: Planar joint angle at vertex `b` between vectors `ba` and `bc`.
- `calculate_segment_angle_to_vertical(p1, p2)`: Segment inclination relative to the vertical axis.
- Normalization: Coordinates scaled relative to shoulder width to remain invariant to distance from the camera.
""")
add_code("""def calculate_angle(a, b, c):
    va = np.array(a[:2], dtype=np.float64) - np.array(b[:2], dtype=np.float64)
    vc = np.array(c[:2], dtype=np.float64) - np.array(b[:2], dtype=np.float64)
    norm_va = np.linalg.norm(va)
    norm_vc = np.linalg.norm(vc)
    if norm_va < 1e-6 or norm_vc < 1e-6:
        return 0.0
    cosine = np.clip(np.dot(va, vc) / (norm_va * norm_vc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def calculate_segment_angle_to_vertical(p1, p2):
    vec = np.array(p2[:2], dtype=np.float64) - np.array(p1[:2], dtype=np.float64)
    norm_v = np.linalg.norm(vec)
    if norm_v < 1e-6:
        return 0.0
    vertical = np.array([0.0, 1.0], dtype=np.float64)
    cosine = np.clip(np.dot(vec, vertical) / norm_v, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

FEATURE_NAMES = [
    'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle',
    'left_arm_vertical_angle', 'right_arm_vertical_angle', 'left_forearm_vertical_angle', 'right_forearm_vertical_angle',
    'left_wrist_norm_x', 'left_wrist_norm_y', 'right_wrist_norm_x', 'right_wrist_norm_y',
    'left_elbow_norm_x', 'left_elbow_norm_y', 'right_elbow_norm_x', 'right_elbow_norm_y',
    'shoulder_symmetry', 'elbow_symmetry', 'torso_angle',
    'elbow_distance', 'wrist_distance', 'left_wrist_to_shoulder_dist', 'right_wrist_to_shoulder_dist'
]
print(f'Total Upper-Body Features Engineered: {len(FEATURE_NAMES)}')
""")

# Section 8: Dataset Creation
add_md("""## 8. Processed Dataset Creation
Extracting the 23 upper-body biomechanical features across all 1,425 samples and saving to `data/processed/bicep_curl_landmarks.csv`.
""")
add_code("""processed_rows = []

for idx, row in df_raw.iterrows():
    rs = np.array([row['RShoulder_X17'], row['RShoulder_Y17']])
    re = np.array([row['RElbow_X18'], row['RElbow_Y18']])
    rw = np.array([row['RWrist_X19'], row['RWrist_Y19']])
    ls = np.array([row['LShoulder_X20'], row['LShoulder_Y20']])
    le = np.array([row['LElbow_X21'], row['LElbow_Y21']])
    lw = np.array([row['LWrist_X22'], row['LWrist_Y22']])
    neck = np.array([row['Neck_X14'], row['Neck_Y14']])
    
    sc = (ls + rs) / 2.0
    sw = float(np.linalg.norm(rs - ls))
    if sw < 1e-5:
        sw = 1.0
        
    feat = {
        'left_elbow_angle': calculate_angle(ls, le, lw),
        'right_elbow_angle': calculate_angle(rs, re, rw),
        'left_shoulder_angle': calculate_angle(sc, ls, le),
        'right_shoulder_angle': calculate_angle(sc, rs, re),
        'left_arm_vertical_angle': calculate_segment_angle_to_vertical(ls, le),
        'right_arm_vertical_angle': calculate_segment_angle_to_vertical(rs, re),
        'left_forearm_vertical_angle': calculate_segment_angle_to_vertical(le, lw),
        'right_forearm_vertical_angle': calculate_segment_angle_to_vertical(re, rw),
        'left_wrist_norm_x': (lw[0] - sc[0]) / sw,
        'left_wrist_norm_y': (lw[1] - sc[1]) / sw,
        'right_wrist_norm_x': (rw[0] - sc[0]) / sw,
        'right_wrist_norm_y': (rw[1] - sc[1]) / sw,
        'left_elbow_norm_x': (le[0] - sc[0]) / sw,
        'left_elbow_norm_y': (le[1] - sc[1]) / sw,
        'right_elbow_norm_x': (re[0] - sc[0]) / sw,
        'right_elbow_norm_y': (re[1] - sc[1]) / sw,
        'shoulder_symmetry': abs(ls[1] - rs[1]) / sw,
        'elbow_symmetry': abs(le[1] - re[1]) / sw,
        'torso_angle': calculate_segment_angle_to_vertical(neck, sc),
        'elbow_distance': np.linalg.norm(re - le) / sw,
        'wrist_distance': np.linalg.norm(rw - lw) / sw,
        'left_wrist_to_shoulder_dist': np.linalg.norm(lw - ls) / sw,
        'right_wrist_to_shoulder_dist': np.linalg.norm(rw - rs) / sw,
        'filename': row['filename'],
        'label': row['label']
    }
    processed_rows.append(feat)

df_processed = pd.DataFrame(processed_rows)
df_processed.to_csv(DATA_PROCESSED_PATH, index=False)
print(f'Processed dataset shape: {df_processed.shape}')
df_processed.head(3)
""")

# Section 9: Data Quality & Cleaning
add_md("""## 9. Data Quality & Cleaning
Reporting missing values, duplicates, and feature distributions.
""")
add_code("""null_counts = df_processed.isnull().sum()
print('Null values found:')
print(null_counts[null_counts > 0] if (null_counts > 0).any() else 'None (0 missing values across all features).')

print('\\nFeature statistics (First 6 features):')
display(df_processed[FEATURE_NAMES[:6]].describe())
""")

# Section 10: Train / Validation / Test Split
add_md("""## 10. Train / Validation / Test Split
Splitting into:
- 70% Training
- 15% Validation
- 15% Testing
With stratified class proportions.
""")
add_code("""X = df_processed[FEATURE_NAMES]
y = df_processed['label']

label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)
class_names = list(label_enc.classes_)
print('Classes Encoded:', {name: int(i) for i, name in enumerate(class_names)})

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=RANDOM_STATE
)

print(f'Train: {len(X_train)} ({len(X_train)/len(X):.1%})')
print(f'Val:   {len(X_val)} ({len(X_val)/len(X):.1%})')
print(f'Test:  {len(X_test)} ({len(X_test)/len(X):.1%})')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
""")

# Section 11 & 12: Model Training & Comparison
add_md("""## 11 & 12. Model Training and Comparison
Benchmarking five machine learning classifiers on the validation set.
""")
add_code("""candidate_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=RANDOM_STATE),
    'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)
}

comparison_list = []
for name, clf in candidate_models.items():
    clf.fit(X_train_scaled, y_train)
    val_pred = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average='macro')
    comparison_list.append({
        'Model': name,
        'Validation Accuracy': round(acc, 4),
        'Validation Macro-F1': round(f1, 4)
    })

df_comp = pd.DataFrame(comparison_list).sort_values(by='Validation Macro-F1', ascending=False)
display(df_comp)

# Plot Model Comparison
fig, ax = plt.subplots(figsize=(9, 4.5))
df_comp.plot(kind='bar', x='Model', y=['Validation Accuracy', 'Validation Macro-F1'], ax=ax, colormap='viridis')
ax.set_title('Model Performance Comparison (Validation Set)')
ax.set_ylabel('Score')
ax.set_ylim(0.70, 1.02)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
""")

# Section 13: Hyperparameter Tuning
add_md("""## 13. Hyperparameter Tuning
Optimizing Random Forest Classifier using `GridSearchCV`.
""")
add_code("""param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 12, 18],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print('Best Parameters:', grid_search.best_params_)
print(f'Best Cross-Validation Macro-F1: {grid_search.best_score_:.4f}')
""")

# Section 14: Final Evaluation
add_md("""## 14. Final Evaluation on Held-Out Test Set
Evaluating the finalized model on the untouched test set (214 samples).
""")
add_code("""y_test_pred = best_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)')
print(f'Test Macro-F1: {test_f1:.4f}')
print('\\nClassification Report:')
print(classification_report(y_test, y_test_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Test Confusion Matrix (Accuracy: {test_acc*100:.1f}%)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.tight_layout()
plt.show()
""")

# Section 15: Model Saving
add_md("""## 15. Model Serialization
Exporting the artifacts required for Flask inference.
""")
add_code("""joblib.dump(best_model, MODELS_DIR / 'bicep_curl_model.pkl')
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
joblib.dump(label_enc, MODELS_DIR / 'label_encoder.pkl')

metadata = {
    'model_type': 'RandomForestClassifier',
    'best_params': grid_search.best_params_,
    'feature_names': FEATURE_NAMES,
    'classes': class_names,
    'metrics': {
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1)
    },
    'confusion_matrix': cm.tolist()
}

with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('All artifacts successfully saved to', MODELS_DIR)
""")

# Section 16 & 17: Inference Demonstration
add_md("""## 16 & 17. Reusable Inference Demonstration
Testing `src/predict.py` with feature predictions and confidence scoring.
""")
add_code("""from src.predict import BicepCurlPredictor

predictor = BicepCurlPredictor(
    model_path=MODELS_DIR / 'bicep_curl_model.pkl',
    scaler_path=MODELS_DIR / 'scaler.pkl',
    encoder_path=MODELS_DIR / 'label_encoder.pkl'
)

print('--- Inference Demonstration on 5 Test Frames ---')
for i in [0, 20, 50, 100, 150]:
    sample_feat = X_test.iloc[i].to_dict()
    true_label = class_names[y_test[i]]
    pred_res = predictor.predict_from_features(sample_feat)
    print(f'Sample {i:3d} | True: {true_label:10s} | Pred: {pred_res[\"display_label\"]:28s} | Conf: {pred_res[\"confidence\"]*100:.1f}%')
""")

output_file = PROJECT_ROOT / 'notebook' / 'bicep_curl_training.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Successfully generated notebook:', output_file)
