# -- coding: utf-8 --
"""
EEG Pipeline (TIME-FREQUENCY FEATURES) + Regularization Comparison
Load → Filter → ICA → Time-Frequency Feature Extraction → Scale → Split
→ Regularization comparison (Strong vs Weak) for:
   - Logistic Regression (L2 via C)
   - SVM RBF (C)
   - Random Forest (tree complexity control)
   - MLP (L2 via alpha)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Project utilities ---
from Utilities.Extractor import process_eeg, filter_eeg

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

# ✅ Windows path using forward slashes
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"


# =========================================================
# FEATURE EXTRACTION: TIME-FREQUENCY (YOUR CODE)
# =========================================================
def extract_timefreq_features(X, sampling_freq=250):
    from scipy import signal as sig

    features = []
    for trial_idx in range(X.shape[0]):
        trial_features = {}
        for ch in range(X.shape[1]):
            eeg_signal = X[trial_idx, ch, :]

            # Power spectral density
            freqs, psd = sig.welch(eeg_signal, fs=sampling_freq, nperseg=128)

            # Bands
            delta = (freqs >= 0.5) & (freqs < 4)
            theta = (freqs >= 4) & (freqs < 8)
            alpha = (freqs >= 8) & (freqs < 13)
            beta  = (freqs >= 13) & (freqs < 30)

            # Mean band power
            trial_features[f"delta_power_ch{ch+1}"] = float(np.mean(psd[delta]))
            trial_features[f"theta_power_ch{ch+1}"] = float(np.mean(psd[theta]))
            trial_features[f"alpha_power_ch{ch+1}"] = float(np.mean(psd[alpha]))
            trial_features[f"beta_power_ch{ch+1}"]  = float(np.mean(psd[beta]))

            # Relative power
            total_power = float(np.sum(psd)) if float(np.sum(psd)) != 0.0 else 1.0
            trial_features[f"delta_rel_ch{ch+1}"] = float(np.sum(psd[delta]) / total_power)
            trial_features[f"alpha_rel_ch{ch+1}"] = float(np.sum(psd[alpha]) / total_power)

        features.append(trial_features)

    return pd.DataFrame(features)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("EEG PIPELINE (TIME-FREQUENCY FEATURES) → REGULARIZATION COMPARISON")
    print("=" * 80)

    # ---------------------------
    # Path sanity check
    # ---------------------------
    if not os.path.isdir(SUBJECT_FOLDER):
        raise FileNotFoundError(
            f"Folder not found: {SUBJECT_FOLDER}\n"
            "Please verify the path exists and points to your subject folder (e.g., sub0)."
        )

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
    # 3) ICA
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

    # =========================================================
    # 4) TIME-FREQUENCY FEATURE EXTRACTION
    # =========================================================
    print("\n[4/5] Extracting time-frequency features...")
    timefreq_df = extract_timefreq_features(X_clean, sampling_freq=SAMPLING_FREQ)
    timefreq_df["Label"] = Y

    timefreq_df.to_csv("features_timefreq.csv", index=False)
    print(f"✓ Time-frequency features: {timefreq_df.shape} (saved: features_timefreq.csv)")

    # =========================================================
    # 5) MODELING + REGULARIZATION COMPARISON (SAME AS P300)
    # =========================================================
    print("\n[5/5] Scaling + split + training models (strong vs weak regularization)...")

    X_feat = timefreq_df.drop(columns=["Label"]).values
    y_raw  = timefreq_df["Label"].values

    # Encode labels → numeric
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("✓ Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results = {}

    # 1) Logistic Regression
    log_strong = LogisticRegression(penalty="l2", C=0.01, max_iter=2000)
    log_weak   = LogisticRegression(penalty="l2", C=10,   max_iter=2000)

    log_strong.fit(X_train, y_train)
    log_weak.fit(X_train, y_train)

    results["LogReg Strong (C=0.01)"] = accuracy_score(y_test, log_strong.predict(X_test))
    results["LogReg Weak (C=10)"]     = accuracy_score(y_test, log_weak.predict(X_test))

    # 2) SVM
    svm_strong = SVC(kernel="rbf", C=0.1, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    svm_weak   = SVC(kernel="rbf", C=10,  gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)

    svm_strong.fit(X_train, y_train)
    svm_weak.fit(X_train, y_train)

    results["SVM Strong (C=0.1)"] = accuracy_score(y_test, svm_strong.predict(X_test))
    results["SVM Weak (C=10)"]    = accuracy_score(y_test, svm_weak.predict(X_test))

    # 3) Random Forest
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

    # 4) MLP
    mlp_strong = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.1,
        max_iter=500,
        random_state=RANDOM_STATE
    )
    mlp_weak = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.0001,
        max_iter=500,
        random_state=RANDOM_STATE
    )

    mlp_strong.fit(X_train, y_train)
    mlp_weak.fit(X_train, y_train)

    results["MLP Strong (alpha=0.1)"]  = accuracy_score(y_test, mlp_strong.predict(X_test))
    results["MLP Weak (alpha=0.0001)"] = accuracy_score(y_test, mlp_weak.predict(X_test))

    # Print results
    print("\n" + "=" * 80)
    print("REGULARIZATION RESULTS (Accuracy on test set) — TIME-FREQUENCY")
    print("=" * 80)
    for model_name, acc in results.items():
        print(f"{model_name:35s} {acc:.2%}")

    # Best model report
    best_name = max(results, key=results.get)
    print("\n" + "-" * 80)
    print(f"BEST MODEL: {best_name} | Accuracy: {results[best_name]:.2%}")
    print("-" * 80)

    model_map = {
        "LogReg Strong (C=0.01)": log_strong,
        "LogReg Weak (C=10)": log_weak,
        "SVM Strong (C=0.1)": svm_strong,
        "SVM Weak (C=10)": svm_weak,
        "RF Strong (Shallow)": rf_strong,
        "RF Weak (Deep)": rf_weak,
        "MLP Strong (alpha=0.1)": mlp_strong,
        "MLP Weak (alpha=0.0001)": mlp_weak,
    }
    best_model = model_map[best_name]
    y_pred_best = best_model.predict(X_test)

    print("\nClassification Report (best model):")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print("Confusion Matrix (best model):")
    print(confusion_matrix(y_test, y_pred_best))

    print("\nDONE ✅")


if __name__ == "__main__":
    main()



######################################################################################
####################################################################################
#############################################################################

# -- coding: utf-8 --
"""
EEG Pipeline (SPATIAL FEATURES) + Regularization Comparison
Load → Filter → ICA → Spatial Feature Extraction → Scale → Split
→ Regularization comparison (Strong vs Weak) for:
   - Logistic Regression (L2 via C)
   - SVM RBF (C)
   - Random Forest (tree complexity control)
   - MLP (L2 via alpha)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Project utilities ---
from Utilities.Extractor import process_eeg, filter_eeg

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

# ✅ Windows path using forward slashes
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"


# =========================================================
# FEATURE EXTRACTION: SPATIAL (YOUR CODE)
# =========================================================
def extract_spatial_features(X, sampling_freq=250):
    p300_start = int(0.25 * sampling_freq)
    p300_end   = int(0.5  * sampling_freq)

    features = []
    for trial_idx in range(X.shape[0]):
        trial_features = {}

        # P300 window across all channels
        p300_all = X[trial_idx, :, p300_start:p300_end]

        trial_features["p300_max_global"]  = float(np.max(p300_all))
        trial_features["p300_mean_global"] = float(np.mean(p300_all))

        # Peak per channel
        channel_peaks = np.max(p300_all, axis=1)
        trial_features["p300_peak_channel"] = int(np.argmax(channel_peaks))
        trial_features["p300_channel_var"]  = float(np.var(channel_peaks))

        # Avg correlation between channels (upper triangle only)
        if X.shape[1] > 1:
            corr = np.corrcoef(p300_all)
            upper_tri = corr[np.triu_indices_from(corr, k=1)]
            trial_features["p300_avg_corr"] = float(np.mean(upper_tri))

        features.append(trial_features)

    return pd.DataFrame(features)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("EEG PIPELINE (SPATIAL FEATURES) → REGULARIZATION COMPARISON")
    print("=" * 80)

    # ---------------------------
    # Path sanity check
    # ---------------------------
    if not os.path.isdir(SUBJECT_FOLDER):
        raise FileNotFoundError(
            f"Folder not found: {SUBJECT_FOLDER}\n"
            "Please verify the path exists and points to your subject folder (e.g., sub0)."
        )

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
    # 3) ICA
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
    # 4) SPATIAL FEATURE EXTRACTION
    # =========================================================
    print("\n[4/5] Extracting spatial features...")
    spatial_df = extract_spatial_features(X_clean, sampling_freq=SAMPLING_FREQ)
    spatial_df["Label"] = Y

    spatial_df.to_csv("features_spatial.csv", index=False)
    print(f"✓ Spatial features: {spatial_df.shape} (saved: features_spatial.csv)")

    # =========================================================
    # 5) MODELING + REGULARIZATION COMPARISON (SAME AS P300)
    # =========================================================
    print("\n[5/5] Scaling + split + training models (strong vs weak regularization)...")

    X_feat = spatial_df.drop(columns=["Label"]).values
    y_raw  = spatial_df["Label"].values

    # Encode labels → numeric (important for LogReg + MLP)
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

    # 1) Logistic Regression (L2 via C)
    log_strong = LogisticRegression(penalty="l2", C=0.01, max_iter=2000)
    log_weak   = LogisticRegression(penalty="l2", C=10,   max_iter=2000)
    log_strong.fit(X_train, y_train)
    log_weak.fit(X_train, y_train)
    results["LogReg Strong (C=0.01)"] = accuracy_score(y_test, log_strong.predict(X_test))
    results["LogReg Weak (C=10)"]     = accuracy_score(y_test, log_weak.predict(X_test))

    # 2) SVM (C controls regularization)
    svm_strong = SVC(kernel="rbf", C=0.1, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    svm_weak   = SVC(kernel="rbf", C=10,  gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    svm_strong.fit(X_train, y_train)
    svm_weak.fit(X_train, y_train)
    results["SVM Strong (C=0.1)"] = accuracy_score(y_test, svm_strong.predict(X_test))
    results["SVM Weak (C=10)"]    = accuracy_score(y_test, svm_weak.predict(X_test))

    # 3) Random Forest (complexity control)
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

    # 4) MLP (L2 via alpha)
    mlp_strong = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.1,    max_iter=500, random_state=RANDOM_STATE)
    mlp_weak   = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.0001, max_iter=500, random_state=RANDOM_STATE)
    mlp_strong.fit(X_train, y_train)
    mlp_weak.fit(X_train, y_train)
    results["MLP Strong (alpha=0.1)"]  = accuracy_score(y_test, mlp_strong.predict(X_test))
    results["MLP Weak (alpha=0.0001)"] = accuracy_score(y_test, mlp_weak.predict(X_test))

    # Print results
    print("\n" + "=" * 80)
    print("REGULARIZATION RESULTS (Accuracy on test set) — SPATIAL FEATURES")
    print("=" * 80)
    for model_name, acc in results.items():
        print(f"{model_name:35s} {acc:.2%}")

    # Best model report
    best_name = max(results, key=results.get)
    print("\n" + "-" * 80)
    print(f"BEST MODEL: {best_name} | Accuracy: {results[best_name]:.2%}")
    print("-" * 80)

    model_map = {
        "LogReg Strong (C=0.01)": log_strong,
        "LogReg Weak (C=10)": log_weak,
        "SVM Strong (C=0.1)": svm_strong,
        "SVM Weak (C=10)": svm_weak,
        "RF Strong (Shallow)": rf_strong,
        "RF Weak (Deep)": rf_weak,
        "MLP Strong (alpha=0.1)": mlp_strong,
        "MLP Weak (alpha=0.0001)": mlp_weak,
    }
    best_model = model_map[best_name]
    y_pred_best = best_model.predict(X_test)

    print("\nClassification Report (best model):")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print("Confusion Matrix (best model):")
    print(confusion_matrix(y_test, y_pred_best))

    print("\nDONE ✅")


if __name__ == "__main__":
    main()


######################################################################################
######################################################################################
######################################################################################
# -- coding: utf-8 --
"""
EEG Pipeline (CSP FEATURES) + Regularization Comparison
Load → Filter → ICA → CSP Feature Extraction → Scale → Split
→ Regularization comparison (Strong vs Weak) for:
   - Logistic Regression (L2 via C)
   - SVM RBF (C)
   - Random Forest (tree complexity control)
   - MLP (L2 via alpha)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- Project utilities ---
from Utilities.Extractor import process_eeg, filter_eeg

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ✅ CSP comes from mne
from mne.decoding import CSP


# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]
TEST_SIZE       = 0.20

# CSP config
CSP_N_COMPONENTS = 4

# ✅ Windows path using forward slashes
SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"


# =========================================================
# CSP FEATURE EXTRACTION (your exact logic)
# =========================================================
def extract_csp_features(X, y, n_components=4):
    """
    WARNING: This is the same approach you had:
    For each trial, you fit CSP on every pair of classes and then transform that trial.
    This is heavy + has information leakage risk because CSP is fitted using all data.
    But I'm keeping it exactly the same since you asked “same way as P300”.
    """
    csp_features_list = []
    classes = np.unique(y)

    for trial_idx in range(X.shape[0]):
        trial_features = {}

        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                mask = (y == class1) | (y == class2)
                X_binary = X[mask]
                y_binary = y[mask]

                csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
                try:
                    csp.fit(X_binary, y_binary)
                    trial_data = X[trial_idx:trial_idx+1]
                    csp_transformed = csp.transform(trial_data)

                    for comp in range(n_components):
                        trial_features[f"csp_{class1}vs{class2}_comp{comp+1}"] = float(csp_transformed[0, comp])

                except Exception:
                    for comp in range(n_components):
                        trial_features[f"csp_{class1}vs{class2}_comp{comp+1}"] = 0.0

        csp_features_list.append(trial_features)

    return pd.DataFrame(csp_features_list)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("EEG PIPELINE (CSP FEATURES) → REGULARIZATION COMPARISON")
    print("=" * 80)

    # ---------------------------
    # Path sanity check
    # ---------------------------
    if not os.path.isdir(SUBJECT_FOLDER):
        raise FileNotFoundError(
            f"Folder not found: {SUBJECT_FOLDER}\n"
            "Please verify the path exists and points to your subject folder (e.g., sub0)."
        )

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
    # 3) ICA
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
    # 4) CSP FEATURE EXTRACTION
    # =========================================================
    print("\n[4/5] Extracting CSP features...")
    csp_df = extract_csp_features(X_clean, Y, n_components=CSP_N_COMPONENTS)
    csp_df["Label"] = Y

    csp_df.to_csv("features_csp.csv", index=False)
    print(f"✓ CSP features: {csp_df.shape} (saved: features_csp.csv)")

    # =========================================================
    # 5) MODELING + REGULARIZATION COMPARISON (same as P300)
    # =========================================================
    print("\n[5/5] Scaling + split + training models (strong vs weak regularization)...")

    X_feat = csp_df.drop(columns=["Label"]).values
    y_raw  = csp_df["Label"].values

    # Encode labels → numeric
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("✓ Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Replace NaNs/Infs in CSP features (sometimes happens)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results = {}

    # 1) Logistic Regression (L2 via C)
    log_strong = LogisticRegression(penalty="l2", C=0.01, max_iter=2000)
    log_weak   = LogisticRegression(penalty="l2", C=10,   max_iter=2000)
    log_strong.fit(X_train, y_train)
    log_weak.fit(X_train, y_train)
    results["LogReg Strong (C=0.01)"] = accuracy_score(y_test, log_strong.predict(X_test))
    results["LogReg Weak (C=10)"]     = accuracy_score(y_test, log_weak.predict(X_test))

    # 2) SVM (C controls regularization)
    svm_strong = SVC(kernel="rbf", C=0.1, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    svm_weak   = SVC(kernel="rbf", C=10,  gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    svm_strong.fit(X_train, y_train)
    svm_weak.fit(X_train, y_train)
    results["SVM Strong (C=0.1)"] = accuracy_score(y_test, svm_strong.predict(X_test))
    results["SVM Weak (C=10)"]    = accuracy_score(y_test, svm_weak.predict(X_test))

    # 3) Random Forest (complexity control)
    rf_strong = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    rf_weak = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    rf_strong.fit(X_train, y_train)
    rf_weak.fit(X_train, y_train)
    results["RF Strong (Shallow)"] = accuracy_score(y_test, rf_strong.predict(X_test))
    results["RF Weak (Deep)"]      = accuracy_score(y_test, rf_weak.predict(X_test))

    # 4) MLP (L2 via alpha)
    mlp_strong = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.1,    max_iter=500, random_state=RANDOM_STATE)
    mlp_weak   = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.0001, max_iter=500, random_state=RANDOM_STATE)
    mlp_strong.fit(X_train, y_train)
    mlp_weak.fit(X_train, y_train)
    results["MLP Strong (alpha=0.1)"]  = accuracy_score(y_test, mlp_strong.predict(X_test))
    results["MLP Weak (alpha=0.0001)"] = accuracy_score(y_test, mlp_weak.predict(X_test))

    # Print results
    print("\n" + "=" * 80)
    print("REGULARIZATION RESULTS (Accuracy on test set) — CSP FEATURES")
    print("=" * 80)
    for model_name, acc in results.items():
        print(f"{model_name:35s} {acc:.2%}")

    # Best model report
    best_name = max(results, key=results.get)
    print("\n" + "-" * 80)
    print(f"BEST MODEL: {best_name} | Accuracy: {results[best_name]:.2%}")
    print("-" * 80)

    model_map = {
        "LogReg Strong (C=0.01)": log_strong,
        "LogReg Weak (C=10)": log_weak,
        "SVM Strong (C=0.1)": svm_strong,
        "SVM Weak (C=10)": svm_weak,
        "RF Strong (Shallow)": rf_strong,
        "RF Weak (Deep)": rf_weak,
        "MLP Strong (alpha=0.1)": mlp_strong,
        "MLP Weak (alpha=0.0001)": mlp_weak,
    }
    best_model = model_map[best_name]
    y_pred_best = best_model.predict(X_test)

    print("\nClassification Report (best model):")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print("Confusion Matrix (best model):")
    print(confusion_matrix(y_test, y_pred_best))

    print("\nDONE ✅")


if __name__ == "__main__":
    main()


#######################################################################################
##############################################################################
################################################################################
# -- coding: utf-8 --
"""
EEG Pipeline (CSP FEATURES) + Regularization Comparison
Load → Filter → ICA → CSP Feature Extraction → Scale → Split
→ Regularization comparison (Strong vs Weak) for:
   - Logistic Regression (L2 via C)
   - SVM RBF (C)
   - Random Forest (tree complexity control)
   - MLP (L2 via alpha)
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import mne
mne.set_log_level("ERROR")  # silence CSP spam logs

# --- Project utilities ---
from Utilities.Extractor import process_eeg, filter_eeg

# --- ML stack ---
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from mne.decoding import CSP


# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE    = 42
SAMPLING_FREQ   = 250
TIME_STEPS      = 1200
INCLUDED_STATES = ["Up", "Down", "Left", "Right", "Select"]
TEST_SIZE       = 0.20

SUBJECT_FOLDER  = "C:/Users/nadin/OneDrive/Desktop/archive (1)/RecordedSessions/sub0"
CSP_N_COMPONENTS = 4


# =========================================================
# CSP FEATURES (your function style)
# =========================================================
def extract_csp_features(X, y, n_components=4):
    """
    Builds CSP features by training CSP for each pair of classes then transforming each trial.
    NOTE: This follows your provided approach "same way as P300".
    """
    csp_features_list = []
    classes = np.unique(y)

    for trial_idx in range(X.shape[0]):
        trial_features = {}

        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                mask = (y == class1) | (y == class2)
                X_binary = X[mask]
                y_binary = y[mask]

                csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)

                try:
                    csp.fit(X_binary, y_binary)
                    trial_data = X[trial_idx:trial_idx+1]  # shape (1, ch, time)
                    csp_transformed = csp.transform(trial_data)  # shape (1, n_components)

                    for comp in range(n_components):
                        trial_features[f"csp_{class1}vs{class2}_comp{comp+1}"] = float(csp_transformed[0, comp])

                except Exception:
                    for comp in range(n_components):
                        trial_features[f"csp_{class1}vs{class2}_comp{comp+1}"] = 0.0

        csp_features_list.append(trial_features)

    return pd.DataFrame(csp_features_list)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("EEG PIPELINE: LOAD → FILTER → ICA → CSP → REGULARIZATION COMPARISON")
    print("=" * 80)

    if not os.path.isdir(SUBJECT_FOLDER):
        raise FileNotFoundError(f"Folder not found: {SUBJECT_FOLDER}")

    # 1) LOAD
    print("\n[1/5] Loading raw EEG data...")
    X, Y = process_eeg(
        TIME_STEPS=TIME_STEPS,
        included_states=INCLUDED_STATES,
        subject_folder=SUBJECT_FOLDER
    )
    print(f"✓ Loaded: X={X.shape}, Y={Y.shape}")

    # 2) FILTER
    print("\n[2/5] Filtering data...")
    X_filt = filter_eeg(X, sampling_freq=SAMPLING_FREQ)
    print(f"✓ Filtered: {X_filt.shape}")

    # 3) ICA
    print("\n[3/5] Applying ICA...")
    n_trials, n_channels, _ = X_filt.shape
    X_clean = np.zeros_like(X_filt)

    for trial in range(n_trials):
        trial_data = X_filt[trial].T
        ica = FastICA(n_components=n_channels, random_state=RANDOM_STATE, max_iter=500)
        X_clean[trial] = ica.fit_transform(trial_data).T

    X_clean = np.nan_to_num(X_clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"✓ ICA complete: X_clean={X_clean.shape}")

    np.save("X_clean.npy", X_clean)
    np.save("Y_labels.npy", Y)
    print("✓ Saved: X_clean.npy, Y_labels.npy")

    # 4) CSP FEATURES  (THIS IS YOUR STEP)
    print("\n[4/5] Extracting CSP features...")
    csp_features = extract_csp_features(X_clean, Y, n_components=CSP_N_COMPONENTS)
    csp_features["Label"] = Y
    csp_features.to_csv("features_csp.csv", index=False)
    print(f"✓ CSP features: {csp_features.shape} (saved: features_csp.csv)")

    # 5) REGULARIZATION COMPARISON (same as your P300)
    print("\n[5/5] Scaling + split + training models with strong/weak regularization...")

    X_feat = csp_features.drop(columns=["Label"]).values
    y_raw  = csp_features["Label"].values

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("✓ Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

    # safety for any NaN/Inf
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results = {}

    # Logistic Regression
    log_strong = LogisticRegression(penalty="l2", C=0.01, max_iter=2000)
    log_weak   = LogisticRegression(penalty="l2", C=10,   max_iter=2000)
    log_strong.fit(X_train, y_train)
    log_weak.fit(X_train, y_train)
    results["LogReg Strong (C=0.01)"] = accuracy_score(y_test, log_strong.predict(X_test))
    results["LogReg Weak (C=10)"]     = accuracy_score(y_test, log_weak.predict(X_test))

    # SVM
    svm_strong = SVC(kernel="rbf", C=0.1, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    svm_weak   = SVC(kernel="rbf", C=10,  gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    svm_strong.fit(X_train, y_train)
    svm_weak.fit(X_train, y_train)
    results["SVM Strong (C=0.1)"] = accuracy_score(y_test, svm_strong.predict(X_test))
    results["SVM Weak (C=10)"]    = accuracy_score(y_test, svm_weak.predict(X_test))

    # Random Forest
    rf_strong = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    rf_weak = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    rf_strong.fit(X_train, y_train)
    rf_weak.fit(X_train, y_train)
    results["RF Strong (Shallow)"] = accuracy_score(y_test, rf_strong.predict(X_test))
    results["RF Weak (Deep)"]      = accuracy_score(y_test, rf_weak.predict(X_test))

    # MLP
    mlp_strong = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.1,    max_iter=500, random_state=RANDOM_STATE)
    mlp_weak   = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.0001, max_iter=500, random_state=RANDOM_STATE)
    mlp_strong.fit(X_train, y_train)
    mlp_weak.fit(X_train, y_train)
    results["MLP Strong (alpha=0.1)"]  = accuracy_score(y_test, mlp_strong.predict(X_test))
    results["MLP Weak (alpha=0.0001)"] = accuracy_score(y_test, mlp_weak.predict(X_test))

    # print results
    print("\n" + "=" * 80)
    print("REGULARIZATION RESULTS (Accuracy on test set) — CSP FEATURES")
    print("=" * 80)
    for model_name, acc in results.items():
        print(f"{model_name:35s} {acc:.2%}")

    # best model report
    best_name = max(results, key=results.get)
    print("\n" + "-" * 80)
    print(f"BEST MODEL: {best_name} | Accuracy: {results[best_name]:.2%}")
    print("-" * 80)

    model_map = {
        "LogReg Strong (C=0.01)": log_strong,
        "LogReg Weak (C=10)": log_weak,
        "SVM Strong (C=0.1)": svm_strong,
        "SVM Weak (C=10)": svm_weak,
        "RF Strong (Shallow)": rf_strong,
        "RF Weak (Deep)": rf_weak,
        "MLP Strong (alpha=0.1)": mlp_strong,
        "MLP Weak (alpha=0.0001)": mlp_weak,
    }
    best_model = model_map[best_name]
    y_pred_best = best_model.predict(X_test)

    print("\nClassification Report (best model):")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print("Confusion Matrix (best model):")
    print(confusion_matrix(y_test, y_pred_best))

    print("\nDONE ✅")


if __name__ == "__main__":
    main()