# Bicep Curl AI Coach: Real-Time Form Analysis & Kinematic Classification

An end-to-end Computer Vision and Machine Learning system for real-time biceps curl form detection, repetition counting, and biomechanical feedback using **MediaPipe Pose** and an optimized **Random Forest Classifier**.

Designed specifically for **Bicep Curl Machines (Preacher / Seated Arm Curl)**: eliminates all lower-body/leg artifacts, delivering strict upper-body isolation, range-of-motion monitoring, and live coaching cues.

[![GitHub Release](https://img.shields.io/github/v/release/AbdalrahmanOthman01/Bicep-Curl-AI-Coach?color=00f2fe&style=flat-square)](https://github.com/AbdalrahmanOthman01/Bicep-Curl-AI-Coach/releases/tag/v1.0.0)
[![Windows Executable](https://img.shields.io/badge/Windows-Standalone_.exe-00c853?style=flat-square&logo=windows)](https://github.com/AbdalrahmanOthman01/Bicep-Curl-AI-Coach/releases/download/v1.0.0/BicepCurlAICoach.exe)

> 📦 **Download Standalone App:** Grab [`BicepCurlAICoach.exe`](https://github.com/AbdalrahmanOthman01/Bicep-Curl-AI-Coach/releases/download/v1.0.0/BicepCurlAICoach.exe) directly from the [GitHub Releases](https://github.com/AbdalrahmanOthman01/Bicep-Curl-AI-Coach/releases/tag/v1.0.0). No Python installation or dependencies required—double-click and start training!

---

## 1. System Architecture

The pipeline processes user video in real-time through a 5-stage inference architecture:

```
Webcam Frame (Browser Canvas / 640x480)
   ↓
MediaPipe Pose (Upper-Body Landmarks)
   ↓
Biomechanical Feature Engineering (23 invariant angles & normalized coordinates)
   ↓
StandardScaler Transform
   ↓
Random Forest Classifier (150 Trees, 94.9% Test Accuracy)
   ↓
Biomechanical Finite State Machine (ROM & Rep Counting)
   ↓
Real-Time HUD Overlay & Live Coaching Feedback
```

---

## 2. Dataset & Kinematic Ground Truth

- **Source:** Kaggle [`pr4nut/bicep-curl-train`](http://kaggle.com/datasets/pr4nut/bicep-curl-train)
- **Total Samples:** 1,425 frames across 24 distinct recorded video sequences.
- **Classes:**
  - `Correct` (483 frames, 7 videos): Full range of motion, controlled tempo.
  - `Partial` (377 frames, 7 videos): Incomplete range of motion (failure to reach peak contraction or full extension).
  - `Leg_Drive` (565 frames, 10 videos): Momentum / body swing from standing context.

### Bicep Curl Machine Adaptation
When performing biceps curls on an exercise machine (preacher bench or seated arm curl machine), the user is seated with feet fixed on the ground or machine footrests. Therefore:
- **All leg landmarks and angles (hips, knees, ankles, toes, shanks, thighs) are excluded from the model feature space.**
- The model focuses entirely on:
  1. Elbow flexion/extension angles (`left_elbow_angle`, `right_elbow_angle`).
  2. Shoulder stability (`left_shoulder_angle`, `right_shoulder_angle`).
  3. Upper arm verticality (`left_arm_vertical_angle`, `right_arm_vertical_angle`) to detect elbow drift off the pad.
  4. Forearm verticality (`left_forearm_vertical_angle`, `right_forearm_vertical_angle`).
  5. Normalized wrist coordinates relative to shoulder width.
  6. Torso inclination and shoulder/elbow symmetry.

---

## 3. Project Directory Structure

```
curl_biceps_v1/
│
├── notebook/
│   └── bicep_curl_training.ipynb     # Complete 17-section training & evaluation notebook
│
├── data/
│   ├── raw/
│   │   └── training_data_bicep_curls.csv   # Kaggle dataset
│   └── processed/
│       └── bicep_curl_landmarks.csv        # Extracted 23-feature dataset
│
├── models/
│   ├── bicep_curl_model.pkl          # Trained Random Forest Classifier
│   ├── scaler.pkl                    # StandardScaler for 23 features
│   ├── label_encoder.pkl             # Scikit-learn LabelEncoder
│   └── model_metadata.json           # Evaluation metrics & hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── pose_features.py              # Biomechanical angle & normalization pipeline
│   ├── predict.py                    # Inference engine with temporal smoothing
│   └── form_analysis.py              # Biomechanical state machine & rep counter
│
├── static/
│   ├── css/
│   │   └── style.css                 # Dark glassmorphism HUD styling
│   └── js/
│       └── app.js                    # Web Audio, Canvas rendering & webcam loop
│
├── templates/
│   └── index.html                    # Dashboard UI
│
├── app.py                            # Flask server and prediction API
├── requirements.txt                  # Pinned dependencies
└── README.md                         # Documentation
```

---

## 4. Model Benchmarking & Evaluation

The dataset was divided using a stratified **70% Training / 15% Validation / 15% Test** split:
- **Training Set:** 997 samples
- **Validation Set:** 214 samples
- **Held-Out Test Set:** 214 samples

### Model Comparison Table (Validation Set)

| Model Architecture | Validation Accuracy | Validation Macro-F1 |
| :--- | :---: | :---: |
| **Random Forest (Tuned)** | **97.66%** | **0.9754** |
| Multi-Layer Perceptron (MLP) | 97.20% | 0.9718 |
| XGBoost Classifier | 95.79% | 0.9570 |
| Support Vector Machine (RBF) | 85.98% | 0.8591 |
| Logistic Regression (Baseline) | 76.17% | 0.7565 |

### Final Evaluation on Untouched Test Set (214 Frames)

- **Test Accuracy:** **94.86% (~95%)**
- **Test Macro-F1:** **0.9469**

#### Classification Report:
```
              precision    recall  f1-score   support

     Correct       0.92      0.97      0.95        72
   Leg_Drive       0.95      0.96      0.96        85
     Partial       0.98      0.89      0.94        57

    accuracy                           0.95       214
   macro avg       0.95      0.94      0.95       214
weighted avg       0.95      0.95      0.95       214
```

#### Confusion Matrix:
```
               Pred Correct   Pred Swing   Pred Partial
True Correct        70            2             0
True Swing           2           82             1
True Partial         4            2            51
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10 – 3.12
- Webcam (built-in or USB)
- Modern browser (Chrome, Edge, Firefox, or Safari)

### Windows Command Line
```powershell
# 1. Clone or navigate to the project directory
cd d:\Work\Coding\curl_biceps_v1

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
```

> **Note on Protobuf & MediaPipe:**  
> MediaPipe 0.10.x requires `protobuf<5.0.0,>=4.25.3`. The project handles this automatically and includes compatibility isolation.

---

## 6. How to Run the System

### 1. Launch the Flask Web Application
```powershell
python app.py
```
Output:
```
* Running on http://127.0.0.1:5000
* Running on http://192.168.1.x:5000
```

### 2. Access the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Using the Web Interface
1. Click **"Start Camera"** and grant browser webcam permissions.
2. Position yourself so your head, shoulders, elbows, and wrists are clearly visible.
3. Start performing bicep curls:
   - **Full Reps:** Curl up until your elbow angle is $\le 55^\circ$, then lower all the way down until $\ge 140^\circ$.
   - **Partial Reps:** Incomplete ROM will be flagged in real time and recorded in the partial counter.
   - **Form Cues:** Live feedback cues provide coaching on elbow drift, torso swing, and range of motion.
4. Click **"Reset Workout"** to reset repetition counters and temporal buffers.

---

## 7. Re-Training & Jupyter Notebook

To inspect the analysis and re-train models from scratch:

```powershell
jupyter notebook notebook/bicep_curl_training.ipynb
```
The notebook executes top-to-bottom without placeholders, re-generating all plots and re-exporting the model artifacts into `models/`.

---

## 8. Technical Distinction: ML vs Biomechanical Rules

| Component | Responsibility | Methodology |
| :--- | :--- | :--- |
| **Pose Estimation** | Detect upper-body body joint coordinates | MediaPipe Pose (33 3D landmarks) |
| **ML Classification** | Global form pattern classification (`Correct`, `Partial`, `Momentum`) | Random Forest (150 trees) on 23 normalized features |
| **Rep Counting** | Track exercise phases and repetition counts | Finite State Machine (`EXTENDED` $\to$ `CURLING_UP` $\to$ `PEAK` $\to$ `LOWERING`) |
| **Form Coaching** | Specific real-time coaching cues (e.g. elbow drift, torso lean) | Geometric kinematic thresholds tailored for machine curls |

---

## 9. Limitations & Future Roadmap

1. **Camera Angle:** Optimal results occur with a 45-degree or side/three-quarter profile view where the elbow joint and forearm arc are unobstructed.
2. **Lighting & Occlusion:** Extreme loose clothing or low lighting may degrade MediaPipe confidence.
3. **Future Extension:** Multi-exercise support (Tricep pushdowns, Lateral raises, Shoulder press) using the same modular architecture.
