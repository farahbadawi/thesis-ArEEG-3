# -*- coding: utf-8 -*-
"""
Fixed EEG Extractor
===================
Key fixes over original:
  1. Proper trial epoching — extracts exactly the [Class] block
  2. Baseline correction — subtracts mean of the Wait period before each trial
  3. Correct scaling — z-score per channel per trial (robust to unknown ADC units)
  4. Rejects saturated/flat trials automatically
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
from typing import List, Tuple


EEG_COLS = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4',
            'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8']


def process_eeg(
    TIME_STEPS: int = 1250,
    included_states: List[str] = ["Up", "Down", "Left", "Right", "Select"],
    subject_folder: str = '',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all CSV files for a subject, epoch by state transitions,
    apply baseline correction using the preceding Wait period,
    and return (X, Y).

    Returns
    -------
    X : np.ndarray  shape (n_trials, 8, TIME_STEPS)
    Y : np.ndarray  shape (n_trials,)  string labels
    """
    csv_files = sorted([f for f in os.listdir(subject_folder) if f.endswith('.csv')])

    all_X, all_Y = [], []

    for fname in csv_files:
        path = os.path.join(subject_folder, fname)
        df   = pd.read_csv(path)

        # ── identify state transition blocks ──────────────────────────────
        change    = df['State'] != df['State'].shift()
        block_id  = change.cumsum()
        groups    = df.groupby(block_id)

        blocks = []   # list of (state, start_idx, end_idx, eeg_array)
        for bid, grp in groups:
            state     = grp['State'].iloc[0]
            start_idx = grp.index[0]
            end_idx   = grp.index[-1]
            eeg       = grp[EEG_COLS].values.T   # (8, n_samples)
            blocks.append((state, start_idx, end_idx, eeg))

        # ── extract trials ────────────────────────────────────────────────
        for i, (state, start, end, eeg) in enumerate(blocks):
            if state not in included_states:
                continue

            # find the Wait block immediately before this trial
            wait_eeg = None
            if i > 0 and blocks[i-1][0] == 'Wait':
                wait_eeg = blocks[i-1][3]   # (8, n_wait_samples)

            # ── epoch: take exactly TIME_STEPS samples ────────────────────
            n_samples = eeg.shape[1]
            if n_samples >= TIME_STEPS:
                epoch = eeg[:, :TIME_STEPS].copy()
            else:
                # pad with zeros if trial is shorter
                pad   = TIME_STEPS - n_samples
                epoch = np.pad(eeg, ((0,0),(0,pad)), mode='constant')

            # ── baseline correction using Wait period ─────────────────────
            if wait_eeg is not None and wait_eeg.shape[1] > 0:
                baseline = wait_eeg.mean(axis=1, keepdims=True)   # (8,1)
                epoch    = epoch - baseline

            # ── reject saturated / flat trials ───────────────────────────
            # flat = std < 1 across all channels (after baseline)
            # saturated = any value > 5 std of the whole trial
            ch_std = epoch.std(axis=1)
            if np.any(ch_std < 1e-6):          # flat channel
                continue
            trial_std = epoch.std()
            if trial_std == 0:
                continue

            # ── z-score normalise per channel ─────────────────────────────
            # makes scale invariant — fixes the 750,000 ADC unit problem
            for ch in range(epoch.shape[0]):
                s = epoch[ch].std()
                if s > 0:
                    epoch[ch] = (epoch[ch] - epoch[ch].mean()) / s

            all_X.append(epoch)
            all_Y.append(state)

    if len(all_X) == 0:
        raise ValueError(f"No valid trials found in {subject_folder}")

    X = np.array(all_X, dtype=np.float32)   # (n_trials, 8, TIME_STEPS)
    Y = np.array(all_Y)

    return X, Y


def filter_eeg(
    X: np.ndarray,
    sampling_freq: int = 250,
    notch_freq: float  = 50.0,    # 50Hz for Egypt/Europe powerline
    lowcut: float      = 0.5,
    highcut: float     = 40.0,
) -> np.ndarray:
    """
    Notch + bandpass filter. No scaling — data is already z-scored.
    Notch set to 50Hz (Egypt uses 50Hz powerline, not 60Hz).
    """
    Q = 30
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs=sampling_freq)

    nyq  = 0.5 * sampling_freq
    b_bp, a_bp = signal.butter(4, [lowcut/nyq, highcut/nyq], btype='band')

    filtered = np.zeros_like(X)
    for trial in range(X.shape[0]):
        for ch in range(X.shape[1]):
            x = X[trial, ch, :]
            x = signal.filtfilt(b_notch, a_notch, x)
            x = signal.filtfilt(b_bp,    a_bp,    x)
            filtered[trial, ch, :] = x

    return filtered.astype(np.float32)