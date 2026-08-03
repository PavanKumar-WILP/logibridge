"""
--------------------------------------------------------
LogiBridge 
Component: Preprocessing Pipeline

Implements:
1. Moving Average Filter
2. Sliding Window Feature Extraction (6 Features)
3. Normalization against training_stats.npy
--------------------------------------------------------
"""

import numpy as np
from scipy.stats import kurtosis
from pathlib import Path

# Relative workspace root resolution
ROOT = Path(__file__).resolve().parents[1]
TRAINING_STATS_PATH = ROOT / "data_pipeline" / "training_stats.npy"

# Configuration
FILTER_WINDOW = 5
WINDOW_SIZE = 30          # seconds
WINDOW_STEP = 10          # seconds

# Global cached stats
TRAINING_STATS = None

def load_stats(path=None):
    """Loads mean and std vectors from training_stats.npy."""
    global TRAINING_STATS
    target_path = Path(path) if path else TRAINING_STATS_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Cannot find training stats file at '{target_path}'")
        
    data = np.load(target_path, allow_pickle=True).item()
    TRAINING_STATS = {
        'mean': np.asarray(data['mean'], dtype=np.float32),
        'std': np.asarray(data['std'], dtype=np.float32)
    }

# ---------------------------------------------------
# Moving Average Filter
# ---------------------------------------------------

def moving_average(signal, window=FILTER_WINDOW):
    """Apply moving average filter."""
    signal = np.asarray(signal)
    if len(signal) < window:
        return signal

    return np.convolve(
        signal,
        np.ones(window) / window,
        mode="valid"
    )

# ---------------------------------------------------
# Feature Functions
# ---------------------------------------------------

def temperature_mean(temp):
    return np.mean(temp)

def temperature_std(temp):
    return np.std(temp)

def temperature_rate(temp):
    """Rate of change in °C/min"""
    duration_minutes = len(temp) / 60.0
    return (temp[-1] - temp[0]) / duration_minutes if duration_minutes > 0 else 0.0

def vibration_rms(vib):
    return np.sqrt(np.mean(vib ** 2))

def vibration_peak(vib):
    return np.max(vib)

def vibration_kurtosis(vib):
    value = kurtosis(vib, fisher=False, bias=False)
    if np.isnan(value):
        value = 3.0
    return float(value)

# ---------------------------------------------------
# Feature Extraction (6 Features)
# ---------------------------------------------------

def extract_features(temp_window, vibration_window):
    """Generates the required 6-dimensional feature vector."""
    temp_filtered = moving_average(temp_window)
    vib_filtered = moving_average(vibration_window)

    features = np.array(
        [
            temperature_mean(temp_filtered),
            temperature_std(temp_filtered),
            temperature_rate(temp_filtered),
            vibration_rms(vib_filtered),
            vibration_peak(vib_filtered),
            vibration_kurtosis(vib_filtered)
        ],
        dtype=np.float32
    )
    return features

# ---------------------------------------------------
# Normalization Functions
# ---------------------------------------------------

def normalize_features(features):
    global TRAINING_STATS
    if TRAINING_STATS is None:
        load_stats()

    mean = TRAINING_STATS["mean"]
    std = TRAINING_STATS["std"]

    if len(mean) != 6 or len(std) != 6:
        raise RuntimeError(
            f"training_stats.npy must contain exactly 6 mean and 6 std values. Found {len(mean)}."
        )

    std_safe = np.where(std == 0, 1.0, std)
    normalized = (features - mean) / std_safe
    return normalized.astype(np.float32)

def normalize_shifted(features):
    global TRAINING_STATS
    if TRAINING_STATS is None:
        load_stats()

    mean = TRAINING_STATS["mean"]
    std = TRAINING_STATS["std"]

    shifted_mean = mean + (3 * std)
    std_safe = np.where(std == 0, 1.0, std)
    normalized = (features - shifted_mean) / std_safe
    return normalized.astype(np.float32)

# ---------------------------------------------------
# Pipeline Entry Point
# ---------------------------------------------------

def preprocess(window_data):
    """
    Wrapper function called by inference_service.py.
    Expects window_data dict containing 'temperature' and 'vibration' buffers.
    """
    temp_window = window_data['temperature']
    vib_window = window_data['vibration']
    
    features = extract_features(temp_window, vib_window)
    normalized = normalize_features(features)
    return normalized