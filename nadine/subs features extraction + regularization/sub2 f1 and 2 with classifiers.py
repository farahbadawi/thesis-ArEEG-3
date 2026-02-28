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
SUBJECT_FOLDER = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub2"

TEST_SIZE = 0.20

print("="*70)
print("EEG P300 FEATURE EXTRACTION + RANDOM FOREST CLASSIFICATION (5 CLASSES)")
print("="*70)

# Quick path sanity check
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub2' directory."
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

# ✅ Windows path using forward slashes (your sub2)
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub2"

print("="*70)
print("EEG P300 FEATURE EXTRACTION + SVM CLASSIFICATION (5 CLASSES)")
print("="*70)

# Quick path sanity check
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub2' directory."
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

# ✅ Windows path using forward slashes (your sub2)
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub2"

print("="*70)
print("EEG P300 FEATURE EXTRACTION + MLP (NEURAL NETWORK) CLASSIFICATION (5 CLASSES)")
print("="*70)

# Quick path sanity check
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub2' directory."
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

# ✅ Windows path to your sub2 folder
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub2"

print("="*70)
print("EEG — STATISTICAL FEATURES + RANDOM FOREST CLASSIFICATION (5 CLASSES)")
print("="*70)

# ---------------------------------------------------------
# Path check
# ---------------------------------------------------------
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub2' directory."
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

# ✅ Windows path to your sub2 folder
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub2"

print("="*70)
print("EEG — STATISTICAL FEATURES + SVM CLASSIFICATION (5 CLASSES)")
print("="*70)

# ---------------------------------------------------------
# Path check
# ---------------------------------------------------------
if not os.path.isdir(SUBJECT_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {SUBJECT_FOLDER}\n"
        "Please verify the path exists and points to your 'sub2' directory."
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

# ✅ Windows path to sub2 (use forward slashes, include the whole path)
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub2"

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
