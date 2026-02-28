# -- coding: utf-8 --
"""
EEG Pipeline (ONE CLEAN SCRIPT)
Load → Filter → ICA → Feature Extraction (P300 OR Statistical) → Scale → Split
→ Regularization comparison (Strong vs Weak) for:
   - Logistic Regression (L2 via C)
   - SVM RBF (C)
   - Random Forest (tree complexity)
   - MLP (L2 via alpha)

"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ✅ ADDED (for team workflow / Kaggle download)
import subprocess

import os
import sys

# Add the folder that CONTAINS "Utilities" to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  # gets thesis-ArEEG-3
search_root = PROJECT_ROOT

found = False
for root, dirs, files in os.walk(search_root):
    if "Utilities" in dirs:
        sys.path.insert(0, root)  # root is the parent of Utilities
        found = True
        break

if not found:
    raise ModuleNotFoundError("Could not find 'Utilities' folder anywhere under the project root.")
# --- Project utilities ---
from Utilities.Extractor import process_eeg, filter_eeg
from Utilities.Preprocessing import compute_statistical_features  # used if FEATURE_MODE="stat"

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]
TEST_SIZE       = 0.20

# Choose feature extraction mode:
#   "p300"  -> extract P300 window features per channel
#   "stat"  -> use compute_statistical_features(X_clean, axis='time')
FEATURE_MODE = "p300"   # <-- change to "stat" if you want statistical features

# ✅ Your Windows path using forward slashes
SUBJECT_FOLDER = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"

# ✅ ADDED (team-friendly dataset setup)
KAGGLE_DATASET = "eslam101ahmed/arabic-eeg-sessions"
LOCAL_DATA_DIR = os.path.join("data", "raw")  # dataset will be downloaded here if needed


# =========================================================
# ✅ ADDED: AUTO-DOWNLOAD DATASET IF MISSING (KAGGLE)
# =========================================================
def ensure_subject_folder(subject_folder: str) -> str:
    """
    Goal: Make the project work for multiple people without pushing the dataset to GitHub.

    If SUBJECT_FOLDER exists -> use it (your current setup).
    If it doesn't exist -> try to download the dataset from Kaggle into data/raw/
    and then try common dataset paths under data/raw/ until we find a folder like sub0.
    """
    # 1) If the user already has the folder, keep it exactly as is
    if os.path.isdir(subject_folder):
        return subject_folder

    # 2) Try to download dataset from Kaggle
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    print("\n[Auto-Setup] SUBJECT_FOLDER not found locally.")
    print("[Auto-Setup] Trying to download dataset from Kaggle into:", LOCAL_DATA_DIR)

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", LOCAL_DATA_DIR, "--unzip"],
            check=True
        )
        print("[Auto-Setup] ✓ Downloaded and unzipped Kaggle dataset.")
    except Exception as e:
        print("[Auto-Setup] ✗ Kaggle download failed.")
        print("           Make sure you installed Kaggle: pip install kaggle")
        print("           And placed kaggle.json in your ~/.kaggle/ or Windows .kaggle folder.")
        raise FileNotFoundError(
            f"Folder not found: {subject_folder}\n"
            "Also failed to auto-download dataset from Kaggle.\n"
            f"Error: {e}"
        )

    # 3) After download, try to locate sub0 under data/raw/
    candidate_paths = [
        os.path.join(LOCAL_DATA_DIR, "RecordedSessions", "sub0"),
        os.path.join(LOCAL_DATA_DIR, "arabic-eeg-sessions", "RecordedSessions", "sub0"),
        os.path.join(LOCAL_DATA_DIR, "RecordedSessions", "RecordedSessions", "sub0"),
        os.path.join(LOCAL_DATA_DIR, "sub0"),
    ]

    for p in candidate_paths:
        if os.path.isdir(p):
            print("[Auto-Setup] ✓ Found subject folder at:", p)
            return p

    # 4) If still not found, raise a clear error
    raise FileNotFoundError(
        f"Folder not found: {subject_folder}\n"
        "Dataset downloaded, but could not locate sub0 automatically under data/raw/.\n"
        "Please check where the dataset extracted and update SUBJECT_FOLDER accordingly."
    )


# =========================================================
# FEATURE EXTRACTION: P300
# =========================================================
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


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    print("=" * 80)
    print("EEG PIPELINE: LOAD → FILTER → ICA → FEATURES → REGULARIZATION COMPARISON")
    print(f"FEATURE_MODE = {FEATURE_MODE}")
    print("=" * 80)

    # ---------------------------
    # Path sanity check
    # ---------------------------
    # ✅ ADDED: if folder doesn't exist, auto-download and set the correct path
    global SUBJECT_FOLDER
    SUBJECT_FOLDER = ensure_subject_folder(SUBJECT_FOLDER)

    # =========================================================
    # 1) LOAD
    # =========================================================
    print("\n[1/5] Loading raw EEG data...")
    X, Y = process_eeg(
        TIME_STEPS=TIME_STEPS,
        included_states=INCLUDED_STATES,
        subject_folder=SUBJECT_FOLDER
    )
    print(f"✓ Loaded: X={X.shape}, Y={Y.shape}")

    # =========================================================
    # 2) FILTER
    # =========================================================
    print("\n[2/5] Filtering data...")
    X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
    print(f"✓ Filtered: {X_filt.shape}")

    # =========================================================
    # 3) ICA (artifact removal)
    # =========================================================
    print("\n[3/5] Applying ICA...")
    n_trials, n_channels, _ = X_filt.shape
    X_clean = np.zeros_like(X_filt)

    for trial in range(n_trials):
        trial_data = X_filt[trial].T  # (samples, channels)
        ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
        X_clean[trial] = ica.fit_transform(trial_data).T

    X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"✓ ICA complete: X_clean={X_clean.shape}")

    # Optional saves
    np.save("X_clean.npy", X_clean)
    np.save("Y_labels.npy", Y)
    print("✓ Saved: X_clean.npy, Y_labels.npy")

    # =========================================================
    # 4) FEATURE EXTRACTION
    # =========================================================
    print("\n[4/5] Extracting features...")
    if FEATURE_MODE.lower() == "p300":
        feats_df = extract_p300_features(X_clean, sampling_freq=SAMPLING_FREQ)
        feats_df["Label"] = Y
        feats_df.to_csv("features_p300.csv", index=False)
        print(f"✓ P300 features: {feats_df.shape}  (saved: features_p300.csv)")

    elif FEATURE_MODE.lower() == "stat":
        # compute_statistical_features expected to return DataFrame-like (your code uses .copy())
        stat_df = compute_statistical_features(X_clean, axis='time')
        feats_df = stat_df.copy()
        feats_df["Label"] = Y
        feats_df.to_csv("features_stat.csv", index=False)
        print(f"✓ Statistical features: {feats_df.shape}  (saved: features_stat.csv)")

    else:
        raise ValueError("FEATURE_MODE must be either 'p300' or 'stat'")

    # =========================================================
    # 5) MODELING + REGULARIZATION COMPARISON (YOUR LOGIC)
    # =========================================================
    print("\n[5/5] Scaling + split + training models with strong/weak regularization...")

    X_feat = feats_df.drop(columns=["Label"]).values
    y_raw  = feats_df["Label"].values

    # ✅ Make labels numeric so LogisticRegression / MLP are always safe
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("✓ Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results = {}

    # ------------------------------
    # 1) Logistic Regression (L2)
    # ------------------------------
    log_strong = LogisticRegression(penalty='l2', C=0.01, max_iter=2000)
    log_weak   = LogisticRegression(penalty='l2', C=10,   max_iter=2000)

    log_strong.fit(X_train, y_train)
    log_weak.fit(X_train, y_train)

    results["LogReg Strong (C=0.01)"] = accuracy_score(y_test, log_strong.predict(X_test))
    results["LogReg Weak (C=10)"]     = accuracy_score(y_test, log_weak.predict(X_test))

    # ------------------------------
    # 2) SVM (C controls regularization)
    # ------------------------------
    svm_strong = SVC(kernel='rbf', C=0.1, gamma='scale', class_weight="balanced", random_state=RANDOM_STATE)
    svm_weak   = SVC(kernel='rbf', C=10,  gamma='scale', class_weight="balanced", random_state=RANDOM_STATE)

    svm_strong.fit(X_train, y_train)
    svm_weak.fit(X_train, y_train)

    results["SVM Strong (C=0.1)"] = accuracy_score(y_test, svm_strong.predict(X_test))
    results["SVM Weak (C=10)"]    = accuracy_score(y_test, svm_weak.predict(X_test))

    # ------------------------------
    # 3) Random Forest (complexity control)
    # ------------------------------
    rf_strong = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_weak = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    rf_strong.fit(X_train, y_train)
    rf_weak.fit(X_train, y_train)

    results["RF Strong (Shallow)"] = accuracy_score(y_test, rf_strong.predict(X_test))
    results["RF Weak (Deep)"]      = accuracy_score(y_test, rf_weak.predict(X_test))

    # ------------------------------
    # 4) MLP (L2 via alpha)
    # ------------------------------
    mlp_strong = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.1,     # strong L2
        max_iter=500,
        random_state=RANDOM_STATE
    )
    mlp_weak = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.0001,  # weak L2
        max_iter=500,
        random_state=RANDOM_STATE
    )

    mlp_strong.fit(X_train, y_train)
    mlp_weak.fit(X_train, y_train)

    results["MLP Strong (alpha=0.1)"]    = accuracy_score(y_test, mlp_strong.predict(X_test))
    results["MLP Weak (alpha=0.0001)"]   = accuracy_score(y_test, mlp_weak.predict(X_test))

    # ======================================================
    # PRINT RESULTS (same format as your regularization script)
    # ======================================================
    print("\n" + "=" * 80)
    print("REGULARIZATION RESULTS (Accuracy on test set)")
    print("=" * 80)
    for model_name, acc in results.items():
        print(f"{model_name:35s} {acc:.2%}")

    # ======================================================
    # OPTIONAL: Detailed report for best model
    # ======================================================
    best_name = max(results, key=results.get)
    print("\n" + "-" * 80)
    print(f"BEST MODEL: {best_name} | Accuracy: {results[best_name]:.2%}")
    print("-" * 80)

    # Pick the best fitted model object (for report)
    # (We re-use the already-trained ones)
    best_model_map = {
        "LogReg Strong (C=0.01)": log_strong,
        "LogReg Weak (C=10)": log_weak,
        "SVM Strong (C=0.1)": svm_strong,
        "SVM Weak (C=10)": svm_weak,
        "RF Strong (Shallow)": rf_strong,
        "RF Weak (Deep)": rf_weak,
        "MLP Strong (alpha=0.1)": mlp_strong,
        "MLP Weak (alpha=0.0001)": mlp_weak,
    }
    best_model = best_model_map[best_name]
    y_pred_best = best_model.predict(X_test)

    print("\nClassification Report (best model):")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print("Confusion Matrix (best model):")
    print(confusion_matrix(y_test, y_pred_best))

    print("\nDONE ✅")


if __name__ == "__main__":
    main()