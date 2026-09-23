"""features/*.parquet (per-frame) -> features/dataset_cal{N}.npz, one per calibration
length N. Each file holds summary-stat features (XGBoost) and raw sequences
(CNN/GRU/LSTM) for the same windows, aligned by construction.

Usage: python -m src.features.build_dataset
"""
import numpy as np
import pandas as pd

from src.features.windows import FEATURES_DIR, iter_calibrated_videos, make_windows

CALIB_LENGTHS_S = (30, 60, 120)
WINDOW_S = 45.0
STRIDE_S = 5.0
FPS = 10.0
TAB_COLS = [
    "ear_mean", "ear_std", "ear_min",
    "mar_mean", "mar_std", "mar_max",
    "pitch_std", "yaw_std",
    "perclos", "blink_rate_per_min", "mean_blink_duration_s",
]


def build(calib_s: int) -> None:
    tabs, seqs, ys, subjects, t_starts = [], [], [], [], []
    for subject, label_name, df, ear_thr in iter_calibrated_videos(calib_s, FPS):
        summary, seq = make_windows(df, WINDOW_S, STRIDE_S, FPS, ear_thr)
        tabs.append(summary[TAB_COLS].to_numpy(dtype=np.float32))
        seqs.append(seq)
        ys.append(np.full(len(seq), int(label_name == "drowsy")))
        subjects.append(np.full(len(seq), subject))
        t_starts.append(summary["t_start"].to_numpy())

    out = FEATURES_DIR / f"dataset_cal{calib_s}.npz"
    np.savez_compressed(
        out,
        X_tab=np.concatenate(tabs),
        X_seq=np.concatenate(seqs),
        y=np.concatenate(ys),
        subjects=np.concatenate(subjects),
        t_start=np.concatenate(t_starts),
        tab_cols=np.array(TAB_COLS),
    )
    y = np.concatenate(ys)
    print(f"cal={calib_s}s: {len(y)} windows ({y.sum()} drowsy), "
          f"{len(np.unique(np.concatenate(subjects)))} subjects -> {out.name}")


if __name__ == "__main__":
    for n in CALIB_LENGTHS_S:
        build(n)
