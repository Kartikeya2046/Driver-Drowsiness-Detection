"""Per-frame features -> calibrated windows (summary stats + raw sequences).
Shared by training (build_dataset.py) and the live loop: the live system calls
calibrate() on the first N s of the session and apply_calibration() after.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "features"
NORM_COLS = ("ear", "mar", "pitch", "yaw", "roll")
CALIB_RESERVE_S = 120.0  # first 120 s of every alert video: calibration only, never evaluated


def blink_threshold(ear_baseline: np.ndarray) -> float:
    """EAR below which the eye counts as closed, from the subject's own alert EAR."""
    return float(np.percentile(ear_baseline, 15))


def calibrate(calib_df: pd.DataFrame) -> dict:
    """Per-subject calibration from a known-alert window (training: first N s of the
    non-drowsy video; live: first N s of the session). Median shift only — a std
    estimated from a few seconds of footage mostly measures how many blinks happened."""
    median = {c: float(calib_df[c].median()) for c in NORM_COLS}
    ear_threshold = blink_threshold(calib_df["ear"].to_numpy()) - median["ear"]
    return {"median": median, "ear_threshold": ear_threshold}


def apply_calibration(df: pd.DataFrame, cal: dict) -> pd.DataFrame:
    df = df.copy()
    for c in NORM_COLS:
        df[c] = df[c] - cal["median"][c]
    return df


def iter_calibrated_videos(calib_s: float, fps: float = 10.0, include_unpaired: bool = False):
    """Yields (subject, label_name, calibrated_df, ear_threshold) for every subject
    with both videos (include_unpaired=True also yields subject 42's alert-only video,
    fine for the forecaster which needs no labels). Calibration = first calib_s of the
    alert video; the alert video's first CALIB_RESERVE_S are dropped for every calib_s
    so evaluation windows are identical across calibration lengths."""
    assert calib_s <= CALIB_RESERVE_S
    for nd_path in sorted(FEATURES_DIR.glob("*_non_drowsy.parquet")):
        subject = nd_path.stem.split("_")[0]
        dr_path = FEATURES_DIR / f"{subject}_drowsy.parquet"
        if not dr_path.exists() and not include_unpaired:  # subject 42: no drowsy video
            continue
        nd = pd.read_parquet(nd_path)
        nd = nd[nd["valid"]].reset_index(drop=True)

        cal = calibrate(nd.iloc[: int(calib_s * fps)])
        nd_eval = nd.iloc[int(CALIB_RESERVE_S * fps) :].reset_index(drop=True)
        videos = [("non_drowsy", nd_eval)]
        if dr_path.exists():
            dr = pd.read_parquet(dr_path)
            videos.append(("drowsy", dr[dr["valid"]].reset_index(drop=True)))
        for label_name, df in videos:
            yield subject, label_name, apply_calibration(df, cal), cal["ear_threshold"]


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
    """One window of calibrated per-frame rows -> flat dict of summary stats."""
    ear = df["ear"].to_numpy()
    mar = df["mar"].to_numpy()

    blinks = detect_blinks(ear, ear_threshold)
    duration_s = len(df) / fps
    blink_durations = [(e - s) / fps for s, e in blinks]

    return {
        "ear_mean": float(np.mean(ear)),
        "ear_std": float(np.std(ear)),
        "ear_min": float(np.min(ear)),
        "mar_mean": float(np.mean(mar)),
        "mar_std": float(np.std(mar)),
        "mar_max": float(np.max(mar)),
        "pitch_std": float(np.std(df["pitch"].to_numpy())),
        "yaw_std": float(np.std(df["yaw"].to_numpy())),
        "perclos": float(np.mean(ear < ear_threshold)),
        "blink_rate_per_min": len(blinks) / (duration_s / 60.0),
        "mean_blink_duration_s": float(np.mean(blink_durations)) if blink_durations else 0.0,
    }


def make_windows(
    df: pd.DataFrame, window_s: float, stride_s: float, fps: float, ear_threshold: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """Slide a window over df. Returns (summary_df, sequences) built from the same
    window boundaries, so row i of both describes the same window."""
    window_n = int(round(window_s * fps))
    stride_n = int(round(stride_s * fps))
    arr = df[list(NORM_COLS)].to_numpy(dtype=np.float32)
    rows, seqs = [], []
    for start in range(0, len(df) - window_n + 1, stride_n):
        chunk = df.iloc[start : start + window_n]
        feats = window_features(chunk, fps, ear_threshold)
        feats["t_start"] = chunk["t"].iloc[0]
        rows.append(feats)
        seqs.append(arr[start : start + window_n])
    if not seqs:
        return pd.DataFrame(rows), np.empty((0, window_n, len(NORM_COLS)), dtype=np.float32)
    return pd.DataFrame(rows), np.stack(seqs)
