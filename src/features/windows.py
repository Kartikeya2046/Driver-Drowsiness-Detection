"""Per-frame features -> normalized windows with derived temporal features
(PERCLOS, blink stats). Shared by training (build_windows.py) and the live
loop's rolling-window summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PERCLOS_CLOSED_FRAC = 0.8  # "eyes >=80% closed" convention
FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "features"
NORM_COLS = ("ear", "mar", "pitch", "yaw", "roll")


def normalize_frame_df(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Z-score df's feature columns against baseline's own mean/std (per-subject
    normalization: baseline is that subject's non-drowsy video stats)."""
    df = df.copy()
    for col in NORM_COLS:
        mu = baseline[col].mean()
        sigma = baseline[col].std()
        sigma = sigma if sigma > 1e-6 else 1.0
        df[col] = (df[col] - mu) / sigma
    return df


def iter_normalized_subject_labels(fps: float, min_window_s: float):
    """Yields (subject, label_name, normalized_df, norm_ear_threshold) for every
    subject/label pair with a usable non-drowsy baseline. Single source of truth
    for per-subject normalization + blink threshold, shared by every window
    builder (summary-stat windows, raw sequence windows, ...)."""
    subject_files = sorted(FEATURES_DIR.glob("*_non_drowsy.parquet")) + sorted(
        FEATURES_DIR.glob("*_drowsy.parquet")
    )
    subjects = sorted({f.stem.split("_")[0] for f in subject_files})

    for subject in subjects:
        nd_path = FEATURES_DIR / f"{subject}_non_drowsy.parquet"
        if not nd_path.exists():
            print(f"  skip {subject}: no non_drowsy file (needed as normalization baseline)")
            continue
        baseline_df = pd.read_parquet(nd_path)
        baseline_valid = baseline_df[baseline_df["valid"]]
        if len(baseline_valid) < fps * 30:
            print(f"  skip {subject}: baseline too short ({len(baseline_valid)} valid frames)")
            continue

        ear_thresh = blink_threshold(baseline_valid["ear"].to_numpy())
        norm_thresh = (ear_thresh - baseline_valid["ear"].mean()) / max(
            baseline_valid["ear"].std(), 1e-6
        )

        for label_name, path in (
            ("non_drowsy", nd_path),
            ("drowsy", FEATURES_DIR / f"{subject}_drowsy.parquet"),
        ):
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df = df[df["valid"]].reset_index(drop=True)
            if len(df) < min_window_s * fps:
                print(f"  skip {subject}/{label_name}: too short for one window")
                continue

            df_norm = normalize_frame_df(df, baseline_valid)
            yield subject, label_name, df_norm, norm_thresh


def blink_threshold(ear_baseline: np.ndarray) -> float:
    """EAR threshold below which the eye is considered closed, derived from a
    subject's own open-eye EAR distribution (per-subject calibration, not a
    fixed global constant)."""
    return float(np.percentile(ear_baseline, 15))


def detect_blinks(ear: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Returns list of (start_idx, end_idx) frame ranges where EAR < threshold."""
    closed = ear < threshold
    blinks = []
    i = 0
    n = len(closed)
    while i < n:
        if closed[i]:
            j = i
            while j < n and closed[j]:
                j += 1
            blinks.append((i, j))
            i = j
        else:
            i += 1
    return blinks


def window_features(df: pd.DataFrame, fps: float, ear_threshold: float) -> dict:
    """df: one window's worth of per-frame feature rows (already filtered to valid==True).
    Returns a flat dict of window-level summary stats for the baseline classifier."""
    ear = df["ear"].to_numpy()
    mar = df["mar"].to_numpy()
    pitch = df["pitch"].to_numpy()
    yaw = df["yaw"].to_numpy()

    perclos = float(np.mean(ear < ear_threshold))

    blinks = detect_blinks(ear, ear_threshold)
    duration_s = len(df) / fps if fps else 0.0
    blink_rate = len(blinks) / (duration_s / 60.0) if duration_s > 0 else 0.0
    blink_durations = [(e - s) / fps for s, e in blinks] if fps else []
    mean_blink_duration = float(np.mean(blink_durations)) if blink_durations else 0.0

    return {
        "ear_mean": float(np.mean(ear)),
        "ear_std": float(np.std(ear)),
        "ear_min": float(np.min(ear)),
        "mar_mean": float(np.mean(mar)),
        "mar_std": float(np.std(mar)),
        "mar_max": float(np.max(mar)),
        "pitch_std": float(np.std(pitch)),
        "yaw_std": float(np.std(yaw)),
        "perclos": perclos,
        "blink_rate_per_min": blink_rate,
        "blink_count": len(blinks),
        "mean_blink_duration_s": mean_blink_duration,
        "n_frames": len(df),
    }


def make_windows(
    df: pd.DataFrame, window_s: float, stride_s: float, fps: float, ear_threshold: float
) -> pd.DataFrame:
    """Slide a window_s-second window over df (must have contiguous 'valid' frames
    already filtered) with stride_s stride. Returns one row per window."""
    window_n = int(round(window_s * fps))
    stride_n = int(round(stride_s * fps))
    rows = []
    n = len(df)
    for start in range(0, n - window_n + 1, stride_n):
        chunk = df.iloc[start : start + window_n]
        feats = window_features(chunk, fps, ear_threshold)
        feats["t_start"] = chunk["t"].iloc[0]
        feats["t_end"] = chunk["t"].iloc[-1]
        rows.append(feats)
    return pd.DataFrame(rows)
