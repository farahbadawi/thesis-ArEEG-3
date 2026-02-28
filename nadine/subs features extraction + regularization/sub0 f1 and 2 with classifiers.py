# -- coding: utf-8 --
"""
EEG P300 Feature Extraction + Random Forest Classification (5 Classes)
Pipeline: Load → Filter → ICA → Extract P300 → Scale → Train RF → Report Accuracy
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import sys

# ✅ Always add project root (thesis-ArEEG-3) to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- Project utilities (your existing modules) ---
from Utilities.Extractor import process_eeg, filter_eeg

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE   = 42
SAMPLING_FREQ  = 250
TIME_STEPS     = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]

# ✅ Correct Windows path using forward slashes
SUBJECT_FOLDER = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"

TEST_SIZE = 0.20

print("="*70)
print("EEG P300 FEATURE EXTRACTION + RANDOM FOREST CLASSIFICATION (5 CLASSES)")
print("="*70)

# Quick path sanity check
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub0' directory."
    )

# =========================================================
# STEP 1: LOAD & FILTER
# =========================================================
print("\n[1/4] Loading raw EEG data...")
X, Y = process_eeg(
    TIME_STEPS=TIME_STEPS,
    included_states=INCLUDED_STATES,
    subject_folder=SUBJECT_FOLDER
)
print(f"✓ Loaded EEG data: X={X.shape}, Y={Y.shape}")

print("\n[2/4] Filtering data...")
X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
print(f"✓ Filtered shape: {X_filt.shape}")

# =========================================================
# STEP 2: ICA (artifact removal)
# =========================================================
print("\n[3/4] Applying ICA for artifact removal...")
n_trials, n_channels, n_samples = X_filt.shape
X_clean = np.zeros_like(X_filt)

for trial in range(n_trials):
    trial_data = X_filt[trial].T  # (samples, channels)
    ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
    trial_clean = ica.fit_transform(trial_data)  # (samples, components)
    X_clean[trial] = trial_clean.T  # back to (channels, samples)

# Safety: replace any NaNs/Infs that might arise from ICA
X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
print(f"✓ ICA complete: {X_clean.shape}")

# Optionally save preprocessed arrays
np.save("X_clean.npy", X_clean)
np.save("Y_labels.npy", Y)
print("✓ Saved: X_clean.npy, Y_labels.npy")

# =========================================================
# STEP 3: P300 FEATURE EXTRACTION (ONLY)
# =========================================================
print("\n[4/4] Extracting P300 features (only)...")

def extract_p300_features(X, sampling_freq=SAMPLING_FREQ):
    """
    Simple P300 features per channel over a 250–500 ms window:
      - p300_amp: max amplitude
      - p300_lat: latency (ms) of the max within trial
      - p300_mean: mean amplitude
      - p300_auc: area under |signal|
    """
    p300_start = int(0.25 * sampling_freq)
    p300_end   = int(0.50 * sampling_freq)

    feats = []
    for trial_idx in range(X.shape[0]):
        row = {}
        for ch in range(X.shape[1]):
            win = X[trial_idx, ch, p300_start:p300_end]
            amp = float(np.max(win))
            lat = int(np.argmax(win)) + p300_start
            row[f"p300_amp_ch{ch+1}"]  = amp
            row[f"p300_lat_ch{ch+1}"]  = (lat / sampling_freq) * 1000.0  # ms
            row[f"p300_mean_ch{ch+1}"] = float(np.mean(win))
            row[f"p300_auc_ch{ch+1}"]  = float(np.sum(np.abs(win)))
        feats.append(row)
    return pd.DataFrame(feats)

p300_df = extract_p300_features(X_clean, sampling_freq=SAMPLING_FREQ)
p300_df["Label"] = Y
print(f"✓ P300 features shape: {p300_df.shape}")

# Save features for inspection
p300_df.to_csv("p300_features.csv", index=False)
print("✓ Saved: p300_features.csv")

# =========================================================
# STEP 4: RANDOM FOREST (5-class) + ACCURACY
# =========================================================
print("\n" + "="*70)
print("TRAINING RANDOM FOREST (5 CLASSES) + REPORTING ACCURACY")
print("="*70)

# Split
X_feat = p300_df.drop(columns=["Label"]).values
y_lab  = p300_df["Label"].values

# Scale features (RF doesn’t require it, but helps if you later compare with other models)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_lab
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# Predict & Accuracy
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Random Forest Accuracy (test set): {acc:.2%}")

# Detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation (global reliability)
cv_scores = cross_val_score(rf, X_scaled, y_lab, cv=5)
print(f"\nCross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

print("\n" + "="*70)
print("DONE ✅  (P300 → Random Forest → Accuracy Printed)")
print("="*70)



##############################################################################
##############################################################################
# -- coding: utf-8 --
"""
EEG P300 Feature Extraction + SVM Classification (5 Classes)
Pipeline: Load → Filter → ICA → Extract P300 → Scale → Train SVM → Report Accuracy
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Project utilities (your existing modules) ---
from Utilities.Extractor import process_eeg, filter_eeg

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]
TEST_SIZE       = 0.20

# ✅ Windows path using forward slashes (your sub0)
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"

print("="*70)
print("EEG P300 FEATURE EXTRACTION + SVM CLASSIFICATION (5 CLASSES)")
print("="*70)

# Quick path sanity check
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub0' directory."
    )

# =========================================================
# STEP 1: LOAD & FILTER
# =========================================================
print("\n[1/4] Loading raw EEG data...")
X, Y = process_eeg(
    TIME_STEPS=TIME_STEPS,
    included_states=INCLUDED_STATES,
    subject_folder=SUBJECT_FOLDER
)
print(f"✓ Loaded EEG data: X={X.shape}, Y={Y.shape}")

print("\n[2/4] Filtering data...")
X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
print(f"✓ Filtered shape: {X_filt.shape}")

# =========================================================
# STEP 3: ICA (artifact removal)
# =========================================================
print("\n[3/4] Applying ICA for artifact removal...")
n_trials, n_channels, n_samples = X_filt.shape
X_clean = np.zeros_like(X_filt)

for trial in range(n_trials):
    trial_data = X_filt[trial].T  # (samples, channels)
    ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
    trial_clean = ica.fit_transform(trial_data)  # (samples, components)
    X_clean[trial] = trial_clean.T  # back to (channels, samples)

# Safety: replace any NaNs/Infs that might arise from ICA
X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
print(f"✓ ICA complete: {X_clean.shape}")

# Optional: save preprocessed arrays
np.save("X_clean.npy", X_clean)
np.save("Y_labels.npy", Y)
print("✓ Saved: X_clean.npy, Y_labels.npy")

# =========================================================
# STEP 4: P300 FEATURE EXTRACTION (ONLY)
# =========================================================
print("\n[4/4] Extracting P300 features (only)...")

def extract_p300_features(X, sampling_freq=SAMPLING_FREQ):
    """
    Simple P300 features per channel over a 250–500 ms window:
      - p300_amp: max amplitude
      - p300_lat: latency (ms) of the max within trial
      - p300_mean: mean amplitude
      - p300_auc: area under |signal|
    """
    p300_start = int(0.25 * sampling_freq)
    p300_end   = int(0.50 * sampling_freq)

    feats = []
    for trial_idx in range(X.shape[0]):
        row = {}
        for ch in range(X.shape[1]):
            win = X[trial_idx, ch, p300_start:p300_end]
            amp = float(np.max(win))
            lat = int(np.argmax(win)) + p300_start
            row[f"p300_amp_ch{ch+1}"]  = amp
            row[f"p300_lat_ch{ch+1}"]  = (lat / sampling_freq) * 1000.0  # ms
            row[f"p300_mean_ch{ch+1}"] = float(np.mean(win))
            row[f"p300_auc_ch{ch+1}"]  = float(np.sum(np.abs(win)))
        feats.append(row)
    return pd.DataFrame(feats)

p300_df = extract_p300_features(X_clean, sampling_freq=SAMPLING_FREQ)
p300_df["Label"] = Y
print(f"✓ P300 features shape: {p300_df.shape}")

# Save features for inspection
p300_df.to_csv("p300_features.csv", index=False)
print("✓ Saved: p300_features.csv")

# =========================================================
# SVM (5-class) + ACCURACY
# =========================================================
print("\n" + "="*70)
print("TRAINING SVM (RBF KERNEL, 5 CLASSES) + REPORTING ACCURACY")
print("="*70)

# Features and labels
X_feat = p300_df.drop(columns=["Label"]).values
y_lab  = p300_df["Label"].values

# SVM is sensitive to feature scales → standardize
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_lab
)

# SVM classifier (multiclass handled via one-vs-one internally)
svm_clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",   # helpful if classes are imbalanced
    probability=False,         # set True if you need predict_proba (slower)
    random_state=RANDOM_STATE
)
svm_clf.fit(X_train, y_train)

# Predictions & metrics
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ SVM Accuracy (test set): {acc:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation for reliability
cv_scores = cross_val_score(svm_clf, X_scaled, y_lab, cv=5)
print(f"\nCross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

print("\n" + "="*70)
print("DONE ✅  (P300 → SVM → Accuracy Printed)")
print("="*70)




######################################################################################################
###############################################################################################
# -- coding: utf-8 --
"""
EEG P300 Feature Extraction + Neural Network (MLP) Classification — 5 Classes
Pipeline: Load → Filter → ICA → Extract P300 → Scale → Train MLP → Report Accuracy

🚩 FIXED: labels are now encoded with LabelEncoder so early_stopping works.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Project utilities (your existing modules) ---
from Utilities.Extractor import process_eeg, filter_eeg

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, LabelEncoder  # <-- LabelEncoder added
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]
TEST_SIZE       = 0.20

# ✅ Windows path using forward slashes (your sub0)
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"

print("="*70)
print("EEG P300 FEATURE EXTRACTION + MLP (NEURAL NETWORK) CLASSIFICATION (5 CLASSES)")
print("="*70)

# Quick path sanity check
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub0' directory."
    )

# =========================================================
# STEP 1: LOAD & FILTER
# =========================================================
print("\n[1/4] Loading raw EEG data...")
X, Y_str = process_eeg(   # <-- keep labels as strings here
    TIME_STEPS=TIME_STEPS,
    included_states=INCLUDED_STATES,
    subject_folder=SUBJECT_FOLDER
)
print(f"✓ Loaded EEG data: X={X.shape}, Y={Y_str.shape}")

print("\n[2/4] Filtering data...")
X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
print(f"✓ Filtered shape: {X_filt.shape}")

# =========================================================
# STEP 3: ICA (artifact removal)
# =========================================================
print("\n[3/4] Applying ICA for artifact removal...")
n_trials, n_channels, n_samples = X_filt.shape
X_clean = np.zeros_like(X_filt)

for trial in range(n_trials):
    trial_data = X_filt[trial].T  # (samples, channels)
    ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
    trial_clean = ica.fit_transform(trial_data)  # (samples, components)
    X_clean[trial] = trial_clean.T  # back to (channels, samples)

# Safety: replace any NaNs/Infs that might arise from ICA
X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
print(f"✓ ICA complete: {X_clean.shape}")

# Optional: save preprocessed arrays
np.save("X_clean.npy", X_clean)
np.save("Y_labels.npy", Y_str)
print("✓ Saved: X_clean.npy, Y_labels.npy")

# =========================================================
# STEP 4: P300 FEATURE EXTRACTION (ONLY)
# =========================================================
print("\n[4/4] Extracting P300 features (only)...")

def extract_p300_features(X, sampling_freq=SAMPLING_FREQ):
    """
    Simple P300 features per channel over a 250–500 ms window:
      - p300_amp: max amplitude
      - p300_lat: latency (ms) of the max within trial
      - p300_mean: mean amplitude
      - p300_auc: area under |signal|
    """
    p300_start = int(0.25 * sampling_freq)
    p300_end   = int(0.50 * sampling_freq)

    feats = []
    for trial_idx in range(X.shape[0]):
        row = {}
        for ch in range(X.shape[1]):
            win = X[trial_idx, ch, p300_start:p300_end]
            amp = float(np.max(win))
            lat = int(np.argmax(win)) + p300_start
            row[f"p300_amp_ch{ch+1}"]  = amp
            row[f"p300_lat_ch{ch+1}"]  = (lat / sampling_freq) * 1000.0  # ms
            row[f"p300_mean_ch{ch+1}"] = float(np.mean(win))
            row[f"p300_auc_ch{ch+1}"]  = float(np.sum(np.abs(win)))
        feats.append(row)
    return pd.DataFrame(feats)

p300_df = extract_p300_features(X_clean, sampling_freq=SAMPLING_FREQ)
p300_df["Label"] = Y_str
print(f"✓ P300 features shape: {p300_df.shape}")

# Save features for inspection
p300_df.to_csv("p300_features.csv", index=False)
print("✓ Saved: p300_features.csv")

# =========================================================
# MLP (Neural Network) — 5-class + ACCURACY
# =========================================================
print("\n" + "="*70)
print("TRAINING MLP (NEURAL NETWORK, 5 CLASSES) + REPORTING ACCURACY")
print("="*70)

# Features and labels
X_feat = p300_df.drop(columns=["Label"]).values
y_str  = p300_df["Label"].values   # string labels

# ✅ Encode labels → integers (fixes the np.isnan error)
le = LabelEncoder()
y_lab = le.fit_transform(y_str)
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# MLP is sensitive to feature scales → standardize
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_lab
)

# MLP classifier
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,                # L2 regularization
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=400,
    early_stopping=True,       # uses 10% of training as validation
    n_iter_no_change=15,
    shuffle=True,
    random_state=RANDOM_STATE,
    verbose=False
)
mlp_clf.fit(X_train, y_train)

# Predictions & metrics
y_pred = mlp_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ MLP Accuracy (test set): {acc:.2%}")

print("\nClassification Report (original class names):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix (encoded labels):")
print(confusion_matrix(y_test, y_pred))

# Cross-validation for reliability (note: slower than RF/SVM)
cv_scores = cross_val_score(mlp_clf, X_scaled, y_lab, cv=5)
print(f"\nCross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

print("\n" + "="*70)
print("DONE ✅  (P300 → MLP → Accuracy Printed)")
print("="*70)






##############################################################################################
#############################################################################################
##############################################################################################
###############################################################################################
# -- coding: utf-8 --
"""
EEG Statistical Features + Random Forest Classification (5 Classes)
Pipeline: Load → Filter → ICA → Extract Statistical Features → Train Random Forest → Accuracy
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Project utilities ---
from Utilities.Extractor import process_eeg, filter_eeg
from Utilities.Preprocessing import compute_statistical_features

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]
TEST_SIZE       = 0.20

# ✅ Windows path to your sub0 folder
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"

print("="*70)
print("EEG — STATISTICAL FEATURES + RANDOM FOREST CLASSIFICATION (5 CLASSES)")
print("="*70)

# ---------------------------------------------------------
# Path check
# ---------------------------------------------------------
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub0' directory."
    )

# ============================================
# PART 1: PREPROCESSING
# ============================================
print("\n[1/3] Loading and filtering EEG data...")

X, Y = process_eeg(
    TIME_STEPS=TIME_STEPS,
    included_states=INCLUDED_STATES,
    subject_folder=SUBJECT_FOLDER
)
print(f"✓ Data loaded: X={X.shape}, Y={Y.shape}")

X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
print(f"✓ Filtered: X_filtered={X_filt.shape}")

# ICA artifact removal
print("\n[2/3] Applying ICA...")
n_trials, n_channels, n_samples = X_filt.shape
X_clean = np.zeros_like(X_filt)

for trial in range(n_trials):
    trial_data = X_filt[trial].T  # (samples, channels)
    ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
    trial_clean = ica.fit_transform(trial_data)
    X_clean[trial] = trial_clean.T

X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
print(f"✓ ICA complete: X_clean={X_clean.shape}")

np.save("X_clean.npy", X_clean)
np.save("Y_labels.npy", Y)

# ============================================
# PART 2: FEATURE EXTRACTION (STATISTICAL ONLY)
# ============================================
print("\n[3/3] Extracting statistical features...")

stat_features = compute_statistical_features(X_clean, axis='time')
print(f"✓ Statistical features extracted: {stat_features.shape}")

# Combine with labels
features_df = stat_features.copy()
features_df["Label"] = Y
features_df.to_csv("stat_features_rf.csv", index=False)
print("✓ Saved: stat_features_rf.csv")

# ============================================
# PART 3: RANDOM FOREST CLASSIFICATION
# ============================================
print("\n" + "="*70)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("="*70)

X_feat = features_df.drop(columns=["Label"]).values
y_lab  = features_df["Label"].values

# Standardize features
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_lab
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Results
print(f"\n✅ Random Forest Accuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(rf, X_scaled, y_lab, cv=5)
print(f"\nCross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

print("\n" + "="*70)
print("PIPELINE COMPLETE ✅ (STATISTICAL FEATURES + RANDOM FOREST)")
print("="*70)

#########################################################################################
#########################################################################################
# -- coding: utf-8 --
"""
EEG Statistical Features + SVM Classification (5 Classes)
Pipeline: Load → Filter → ICA → Extract Statistical Features → Train SVM → Accuracy
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Project utilities ---
from Utilities.Extractor import process_eeg, filter_eeg
from Utilities.Preprocessing import compute_statistical_features

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]
TEST_SIZE       = 0.20

# ✅ Windows path to your sub0 folder
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"

print("="*70)
print("EEG — STATISTICAL FEATURES + SVM CLASSIFICATION (5 CLASSES)")
print("="*70)

# ---------------------------------------------------------
# Path check
# ---------------------------------------------------------
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub0' directory."
    )

# ============================================
# PART 1: PREPROCESSING
# ============================================
print("\n[1/3] Loading and filtering EEG data...")

X, Y = process_eeg(
    TIME_STEPS=TIME_STEPS,
    included_states=INCLUDED_STATES,
    subject_folder=SUBJECT_FOLDER
)
print(f"✓ Data loaded: X={X.shape}, Y={Y.shape}")

X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
print(f"✓ Filtered: X_filtered={X_filt.shape}")

# ICA artifact removal
print("\n[2/3] Applying ICA...")
n_trials, n_channels, n_samples = X_filt.shape
X_clean = np.zeros_like(X_filt)

for trial in range(n_trials):
    trial_data = X_filt[trial].T  # (samples, channels)
    ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
    trial_clean = ica.fit_transform(trial_data)
    X_clean[trial] = trial_clean.T

X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
print(f"✓ ICA complete: X_clean={X_clean.shape}")

np.save("X_clean.npy", X_clean)
np.save("Y_labels.npy", Y)

# ============================================
# PART 2: FEATURE EXTRACTION (STATISTICAL ONLY)
# ============================================
print("\n[3/3] Extracting statistical features...")

stat_features = compute_statistical_features(X_clean, axis='time')
print(f"✓ Statistical features extracted: {stat_features.shape}")

# Combine with labels
features_df = stat_features.copy()
features_df["Label"] = Y
features_df.to_csv("stat_features_svm.csv", index=False)
print("✓ Saved: stat_features_svm.csv")

# ============================================
# PART 3: SVM CLASSIFICATION
# ============================================
print("\n" + "="*70)
print("TRAINING SVM CLASSIFIER (RBF KERNEL, 5 CLASSES)")
print("="*70)

X_feat = features_df.drop(columns=["Label"]).values
y_lab  = features_df["Label"].values

# SVM is sensitive to feature scale
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_feat)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_lab
)

# SVM classifier
svm_clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",   # helps with class imbalance
    probability=False,
    random_state=RANDOM_STATE
)
svm_clf.fit(X_train, y_train)

# Predictions
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Results
print(f"\n✅ SVM Accuracy (test set): {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(svm_clf, X_scaled, y_lab, cv=5)
print(f"\nCross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

print("\n" + "="*70)
print("PIPELINE COMPLETE ✅ (STATISTICAL FEATURES + SVM)")
print("="*70)

#####################################################################################################
#########################################################################################################
# -- coding: utf-8 --
"""
EEG Statistical Features ONLY + Neural Network (MLP) — 5 Classes
Pipeline: Load → Filter → ICA → Statistical Features → Scale → Train MLP → Report Accuracy
(Your original code here was already correct with LabelEncoder.)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Project utilities
from Utilities.Extractor import process_eeg, filter_eeg
from Utilities.Preprocessing import compute_statistical_features

# ML stack
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------- CONFIG ---------------------
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]
TEST_SIZE       = 0.20

# ✅ Windows path to sub0 (use forward slashes, include the whole path)
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"

def main():
    print("="*70)
    print("EEG — STATISTICAL FEATURES ONLY + MLP (NEURAL NETWORK) — 5 CLASSES")
    print("="*70)

    # Sanity check
    if not os.path.isdir(SUBJECT_FOLDER):
        raise FileNotFoundError(f"Folder not found: {SUBJECT_FOLDER}")

    # --------------------- STEP 1: LOAD & FILTER ---------------------
    print("\n[1/3] Loading raw EEG data...")
    X, Y_str = process_eeg(
        TIME_STEPS=TIME_STEPS,
        included_states=INCLUDED_STATES,
        subject_folder=SUBJECT_FOLDER
    )
    print(f"✓ Data loaded: X={X.shape}, Y={Y_str.shape}")

    print("\n[2/3] Filtering data...")
    X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
    print(f"✓ Filtered shape: {X_filt.shape}")

    # --------------------- STEP 2: ICA ---------------------
    print("\n[3/3] Applying ICA for artifact removal...")
    n_trials, n_channels, n_samples = X_filt.shape
    X_clean = np.zeros_like(X_filt)
    for trial in range(n_trials):
        trial_data = X_filt[trial].T  # (samples, channels)
        ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
        trial_clean = ica.fit_transform(trial_data)  # (samples, components)
        X_clean[trial] = trial_clean.T  # back to (channels, samples)
    X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"✓ ICA complete: {X_clean.shape}")

    # --------------------- FEATURES: Statistical ONLY ---------------------
    print("\nExtracting statistical features (time-axis) …")
    stat_df = compute_statistical_features(X_clean, axis='time')  # (trials, features)
    print(f"✓ Statistical features: {stat_df.shape}")

    # Encode labels to integers (MLP & CV prefer numeric labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y_str)
    print("✓ Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Build dataset
    features_df = stat_df.copy()
    features_df["Label"] = Y
    features_df.to_csv("stat_features_only.csv", index=False)
    print("✓ Saved: stat_features_only.csv")

    # --------------------- TRAIN: MLP ---------------------
    print("\n" + "="*70)
    print("TRAINING MLP (NEURAL NETWORK) — 5 CLASSES")
    print("="*70)

    X_feat = features_df.drop(columns=["Label"]).values
    y_lab  = features_df["Label"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_lab, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_lab
    )

    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=15,
        shuffle=True,
        random_state=RANDOM_STATE,
        verbose=False
    )

    mlp_clf.fit(X_train, y_train)

    y_pred = mlp_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ MLP Accuracy (test set): {acc:.2%}\n")

    print("Classification Report (original class names):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion Matrix (encoded labels):")
    print(confusion_matrix(y_test, y_pred))

    # Optional CV (can be slow)
    cv_scores = cross_val_score(mlp_clf, X_scaled, y_lab, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    print("\n" + "="*70)
    print("DONE  (Statistical Features → MLP → Accuracy Above)")
    print("="*70)

if __name__ == "__main__":
    main()




####################################################################################################
####################################################################################################
####################################################################################################
#####################################################################################################
#####################################################################################################
# -- coding: utf-8 --
"""
EEG P300 + Random Forest
Train on subjects: sub0, sub1
Test  on subject : sub11

Pipeline per subject: Load → Filter → ICA → Extract P300 → Features
Then: concatenate train subjects → scale → train RF → test on sub11
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from Utilities.Extractor import process_eeg, filter_eeg

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- CONFIG ----------------
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]

BASE_FOLDER     = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions"
TRAIN_SUBJECTS  = ["sub0", "sub1"]
TEST_SUBJECT    = "sub11"

print("="*80)
print("P300 + RandomForest   (train: sub0, sub1  |  test: sub11)")
print("="*80)

# ---------- helper: P300 features ----------
def extract_p300_features(X, sampling_freq=SAMPLING_FREQ):
    """
    Simple P300 features per channel over 250–500 ms window:
    - max amplitude
    - latency (ms) of the max
    - mean amplitude
    - area under |signal|
    """
    p300_start = int(0.25 * sampling_freq)
    p300_end   = int(0.50 * sampling_freq)

    feats = []
    for trial_idx in range(X.shape[0]):
        row = {}
        for ch in range(X.shape[1]):
            win = X[trial_idx, ch, p300_start:p300_end]
            amp = float(np.max(win))
            lat = int(np.argmax(win)) + p300_start
            row[f"p300_amp_ch{ch+1}"]  = amp
            row[f"p300_lat_ch{ch+1}"]  = (lat / sampling_freq) * 1000.0  # ms
            row[f"p300_mean_ch{ch+1}"] = float(np.mean(win))
            row[f"p300_auc_ch{ch+1}"]  = float(np.sum(np.abs(win)))
        feats.append(row)
    return pd.DataFrame(feats)

# ---------- helper: load + preprocess one subject ----------
def load_and_preprocess_subject(subject_name):
    folder = os.path.join(BASE_FOLDER, subject_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    print(f"\n=== Subject {subject_name} ===")
    print("[1] Loading...")
    X, Y = process_eeg(
        TIME_STEPS=TIME_STEPS,
        included_states=INCLUDED_STATES,
        subject_folder=folder
    )
    print(f"    X={X.shape}, Y={Y.shape}")

    print("[2] Filtering...")
    X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
    print(f"    X_filt={X_filt.shape}")

    print("[3] ICA...")
    n_trials, n_channels, _ = X_filt.shape
    X_clean = np.zeros_like(X_filt)
    for t in range(n_trials):
        trial_data = X_filt[t].T  # (samples, channels)
        ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
        trial_clean = ica.fit_transform(trial_data)
        X_clean[t] = trial_clean.T
    X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    X_clean={X_clean.shape}")

    print("[4] P300 features...")
    p300_df = extract_p300_features(X_clean, sampling_freq=SAMPLING_FREQ)
    p300_df["Label"] = Y
    print(f"    P300_feat={p300_df.shape}")

    return p300_df

# ======================================================
# 1) BUILD TRAIN SET  (sub0 + sub1)
# ======================================================
train_dfs = []
for s in TRAIN_SUBJECTS:
    df_s = load_and_preprocess_subject(s)
    df_s["Subject"] = s  # optional, just to know origin
    train_dfs.append(df_s)

train_df = pd.concat(train_dfs, axis=0).reset_index(drop=True)
print("\n>>> Combined TRAIN set:", train_df.shape)

# ======================================================
# 2) BUILD TEST SET  (sub11)
# ======================================================
test_df = load_and_preprocess_subject(TEST_SUBJECT)
test_df["Subject"] = TEST_SUBJECT
print(">>> TEST set:", test_df.shape)

# ======================================================
# 3) TRAIN RF ON TRAIN SUBJECTS, TEST ON sub11
# ======================================================
X_train = train_df.drop(columns=["Label", "Subject"]).values
y_train = train_df["Label"].values

X_test  = test_df.drop(columns=["Label", "Subject"]).values
y_test  = test_df["Label"].values

# scale using only TRAIN data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest on sub0 + sub1...")
rf.fit(X_train_scaled, y_train)

print("\nEvaluating on sub11...")
y_pred = rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*80)
print(f"Accuracy on subject {TEST_SUBJECT}: {acc:.2%}")
print("="*80)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nDone ✅")


########################################################################################
######################################################################################
# -- coding: utf-8 --
"""
EEG P300 + Random Forest (Cross-subject)

Train on subjects: sub0, sub1
Test  on subject : sub11

Per subject pipeline:
    Load  →  Filter  →  ICA  →  Extract P300 features

Then:
    Concatenate train subjects  →  Scale  →  Train RF  →  Test on sub11
"""

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ---------- project utilities ----------
from Utilities.Extractor import process_eeg, filter_eeg

# ---------- ML stack ----------
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- CONFIG ----------------
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]

# path to the "RecordedSessions" folder (CHANGE ONLY IF YOUR PATH IS DIFFERENT)
BASE_FOLDER     = r"C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions"

TRAIN_SUBJECTS  = ["sub0", "sub1"]
TEST_SUBJECT    = "sub11"


# ---------- helper: P300 features ----------
def extract_p300_features(X, sampling_freq=SAMPLING_FREQ):
    """
    Simple P300 features per channel over 250–500 ms window:
      - max amplitude
      - latency (ms) of the max
      - mean amplitude
      - area under |signal|
    """
    p300_start = int(0.25 * sampling_freq)
    p300_end   = int(0.50 * sampling_freq)

    feats = []
    for trial_idx in range(X.shape[0]):
        row = {}
        for ch in range(X.shape[1]):
            win = X[trial_idx, ch, p300_start:p300_end]

            amp = float(np.max(win))
            lat = int(np.argmax(win)) + p300_start

            row[f"p300_amp_ch{ch+1}"]  = amp
            row[f"p300_lat_ch{ch+1}"]  = (lat / sampling_freq) * 1000.0  # ms
            row[f"p300_mean_ch{ch+1}"] = float(np.mean(win))
            row[f"p300_auc_ch{ch+1}"]  = float(np.sum(np.abs(win)))

        feats.append(row)

    return pd.DataFrame(feats)


# ---------- helper: load + preprocess one subject ----------
def load_and_preprocess_subject(subject_name: str) -> pd.DataFrame:
    folder = os.path.join(BASE_FOLDER, subject_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    print(f"\n=== Subject {subject_name} ===")
    print("[1] Loading...")
    X, Y = process_eeg(
        TIME_STEPS=TIME_STEPS,
        included_states=INCLUDED_STATES,
        subject_folder=folder,
    )
    print(f"    X={X.shape}, Y={Y.shape}")

    print("[2] Filtering...")
    X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
    print(f"    X_filt={X_filt.shape}")

    print("[3] ICA...")
    n_trials, n_channels, _ = X_filt.shape
    X_clean = np.zeros_like(X_filt)

    for t in range(n_trials):
        trial_data = X_filt[t].T  # (samples, channels)
        ica = FastICA(
            n_components=n_channels,
            random_state=RANDOM_STATE,
            max_iter=500,
        )
        trial_clean = ica.fit_transform(trial_data)
        X_clean[t] = trial_clean.T

    # clean NaNs/Infs
    X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    X_clean={X_clean.shape}")

    print("[4] P300 features...")
    p300_df = extract_p300_features(X_clean, sampling_freq=SAMPLING_FREQ)
    p300_df["Label"] = Y
    print(f"    P300_feat={p300_df.shape}")

    return p300_df


def main():
    print("=" * 80)
    print("P300 + RandomForest   (train: sub0, sub1  |  test: sub11)")
    print("=" * 80)

    # ======================================================
    # 1) BUILD TRAIN SET  (sub0 + sub1)
    # ======================================================
    train_dfs = []
    for s in TRAIN_SUBJECTS:
        df_s = load_and_preprocess_subject(s)
        train_dfs.append(df_s)

    train_df = pd.concat(train_dfs, axis=0).reset_index(drop=True)
    print("\n>>> Combined TRAIN set:", train_df.shape)

    # ======================================================
    # 2) BUILD TEST SET  (sub11)
    # ======================================================
    test_df = load_and_preprocess_subject(TEST_SUBJECT)
    print(">>> TEST set:", test_df.shape)

    # ======================================================
    # 3) TRAIN RF ON TRAIN SUBJECTS, TEST ON sub11
    # ======================================================
    X_train = train_df.drop(columns=["Label"]).values
    y_train = train_df["Label"].values

    X_test  = test_df.drop(columns=["Label"]).values
    y_test  = test_df["Label"].values

    # scale using only TRAIN data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("\nTraining Random Forest on sub0 + sub1...")
    rf.fit(X_train_scaled, y_train)

    print("\nEvaluating on sub11...")
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 80)
    print(f"Accuracy on subject {TEST_SUBJECT}: {acc:.2%}")
    print("=" * 80)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nDone ✅")


if __name__ == "__main__":
    main()



#########################################
######################################
###############################################
#########################################
################################
##############################
#train on sub0 and sub1 and test on sub 11
# -- coding: utf-8 --
"""
EEG Cross-Subject Classification
Train on: sub0 + sub1
Test on: sub11
Pipeline: Load Multiple Subjects → Combine → Train → Test on Separate Subject
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Project utilities ---
from Utilities.Extractor import process_eeg, filter_eeg
from Utilities.Preprocessing import compute_statistical_features

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]

# ✅ Define folder paths
BASE_FOLDER = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions"
TRAIN_SUBJECTS = ["sub0", "sub1"]  # Training subjects
TEST_SUBJECT   = "sub11"            # Test subject

print("="*70)
print("EEG CROSS-SUBJECT CLASSIFICATION")
print(f"Training on: {', '.join(TRAIN_SUBJECTS)}")
print(f"Testing on: {TEST_SUBJECT}")
print("="*70)


# =========================================================
# HELPER FUNCTION: Process Single Subject
# =========================================================
def process_single_subject(subject_folder_name):
    """
    Load, filter, and clean EEG data for one subject
    Returns: X_clean (trials, channels, samples), Y (labels)
    """
    subject_path = os.path.join(BASE_FOLDER, subject_folder_name)
    
    if not os.path.isdir(subject_path):
        raise FileNotFoundError(f"Subject folder not found: {subject_path}")
    
    print(f"\n  Processing {subject_folder_name}...")
    
    # Load data
    X, Y = process_eeg(
        TIME_STEPS=TIME_STEPS,
        included_states=INCLUDED_STATES,
        subject_folder=subject_path
    )
    print(f"  ✓ Loaded: X={X.shape}, Y={Y.shape}")
    
    # Filter
    X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
    
    # ICA artifact removal
    n_trials, n_channels, n_samples = X_filt.shape
    X_clean = np.zeros_like(X_filt)
    
    for trial in range(n_trials):
        trial_data = X_filt[trial].T
        ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
        trial_clean = ica.fit_transform(trial_data)
        X_clean[trial] = trial_clean.T
    
    X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  ✓ Cleaned: X_clean={X_clean.shape}")
    
    return X_clean, Y


# =========================================================
# STEP 1: LOAD TRAINING DATA (sub0 + sub1)
# =========================================================
print("\n[1/4] Loading TRAINING subjects (sub0 + sub1)...")

X_train_list = []
Y_train_list = []

for subject in TRAIN_SUBJECTS:
    X_clean, Y = process_single_subject(subject)
    X_train_list.append(X_clean)
    Y_train_list.append(Y)

# Combine training data from both subjects
X_train_combined = np.concatenate(X_train_list, axis=0)  # Stack along trials
Y_train_combined = np.concatenate(Y_train_list, axis=0)

print(f"\n✓ Combined training data: X={X_train_combined.shape}, Y={Y_train_combined.shape}")


# =========================================================
# STEP 2: LOAD TEST DATA (sub11)
# =========================================================
print("\n[2/4] Loading TEST subject (sub11)...")

X_test_clean, Y_test = process_single_subject(TEST_SUBJECT)

print(f"✓ Test data: X={X_test_clean.shape}, Y={Y_test.shape}")


# =========================================================
# STEP 3: EXTRACT FEATURES
# =========================================================
print("\n[3/4] Extracting features...")

# Option 1: Statistical Features (recommended)
print("  Extracting statistical features...")
train_features = compute_statistical_features(X_train_combined, axis='time')
test_features = compute_statistical_features(X_test_clean, axis='time')

# Option 2: P300 Features (uncomment if you prefer)
# def extract_p300_features(X, sampling_freq=SAMPLING_FREQ):
#     p300_start = int(0.25 * sampling_freq)
#     p300_end   = int(0.50 * sampling_freq)
#     feats = []
#     for trial_idx in range(X.shape[0]):
#         row = {}
#         for ch in range(X.shape[1]):
#             win = X[trial_idx, ch, p300_start:p300_end]
#             row[f"p300_amp_ch{ch+1}"]  = float(np.max(win))
#             row[f"p300_lat_ch{ch+1}"]  = (int(np.argmax(win)) + p300_start) / sampling_freq * 1000.0
#             row[f"p300_mean_ch{ch+1}"] = float(np.mean(win))
#             row[f"p300_auc_ch{ch+1}"]  = float(np.sum(np.abs(win)))
#         feats.append(row)
#     return pd.DataFrame(feats)
# 
# train_features = extract_p300_features(X_train_combined)
# test_features = extract_p300_features(X_test_clean)

print(f"✓ Training features: {train_features.shape}")
print(f"✓ Test features: {test_features.shape}")

# Encode labels
le = LabelEncoder()
Y_train_encoded = le.fit_transform(Y_train_combined)  # Fit on training labels
Y_test_encoded = le.transform(Y_test)                 # Transform test labels

print(f"✓ Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# =========================================================
# STEP 4: STANDARDIZE FEATURES
# =========================================================
print("\n[4/4] Standardizing features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features)      # Fit on training data
X_test_scaled = scaler.transform(test_features)            # Transform test data (don't fit!)

print(f"✓ Training features scaled: {X_train_scaled.shape}")
print(f"✓ Test features scaled: {X_test_scaled.shape}")


# =========================================================
# STEP 5: TRAIN & TEST MODELS
# =========================================================
print("\n" + "="*70)
print("TRAINING & TESTING MODELS")
print("="*70)

# ------------------------------
# Model 1: Random Forest
# ------------------------------
print("\n[1/3] Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train_scaled, Y_train_encoded)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(Y_test_encoded, rf_pred)

print(f"✅ Random Forest Accuracy: {rf_acc:.2%}")
print("\nClassification Report:")
print(classification_report(Y_test_encoded, rf_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(Y_test_encoded, rf_pred))


# ------------------------------
# Model 2: SVM
# ------------------------------
print("\n[2/3] SVM (RBF Kernel)...")
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=RANDOM_STATE
)
svm.fit(X_train_scaled, Y_train_encoded)
svm_pred = svm.predict(X_test_scaled)
svm_acc = accuracy_score(Y_test_encoded, svm_pred)

print(f"✅ SVM Accuracy: {svm_acc:.2%}")
print("\nClassification Report:")
print(classification_report(Y_test_encoded, svm_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(Y_test_encoded, svm_pred))


# ------------------------------
# Model 3: Neural Network (MLP)
# ------------------------------
print("\n[3/3] Neural Network (MLP)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=400,
    early_stopping=True,
    n_iter_no_change=15,
    shuffle=True,
    random_state=RANDOM_STATE,
    verbose=False
)
mlp.fit(X_train_scaled, Y_train_encoded)
mlp_pred = mlp.predict(X_test_scaled)
mlp_acc = accuracy_score(Y_test_encoded, mlp_pred)

print(f"✅ MLP Accuracy: {mlp_acc:.2%}")
print("\nClassification Report:")
print(classification_report(Y_test_encoded, mlp_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(Y_test_encoded, mlp_pred))


# =========================================================
# SUMMARY
# =========================================================
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"Training set size: {X_train_scaled.shape[0]} trials (from sub0 + sub1)")
print(f"Test set size: {X_test_scaled.shape[0]} trials (from sub11)")
print(f"\nRandom Forest Accuracy: {rf_acc:.2%}")
print(f"SVM Accuracy: {svm_acc:.2%}")
print(f"MLP Accuracy: {mlp_acc:.2%}")
print("="*70)