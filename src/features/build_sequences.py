"""features/*.parquet (per-frame) -> features/sequences.npz: raw per-subject-
normalized frame sequences per window (for the 1D-CNN sequence model), same
window/stride and normalization as build_windows.py so results are comparable.

Usage: python -m src.features.build_sequences
"""
from pathlib import Path

import numpy as np

from src.features.windows import NORM_COLS, iter_normalized_subject_labels

FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "features"
WINDOW_S = 45.0
STRIDE_S = 5.0
FPS = 10.0
EXCLUDE_FROM_CLASSIFIER = {"42"}


def make_sequence_windows(df_norm, window_s: float, stride_s: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, t_start): X is (n_windows, window_n, n_channels); t_start is
    each window's start timestamp — the join key back to windows.parquet, since
    row order alone shouldn't be trusted to align two independently-built files."""
    window_n = int(round(window_s * fps))
    stride_n = int(round(stride_s * fps))
    arr = df_norm[list(NORM_COLS)].to_numpy(dtype=np.float32)  # (n_frames, n_channels)
    t = df_norm["t"].to_numpy()
    n = len(arr)
    starts = list(range(0, n - window_n + 1, stride_n))
    windows = [arr[s : s + window_n] for s in starts]
    X = np.stack(windows) if windows else np.empty((0, window_n, len(NORM_COLS)), dtype=np.float32)
    t_start = t[starts] if starts else np.empty((0,), dtype=np.float64)
    return X, t_start


def main() -> None:
    all_X, all_y, all_subjects, all_usable, all_t_start = [], [], [], [], []
    for subject, label_name, df_norm, _ in iter_normalized_subject_labels(FPS, WINDOW_S):
        X, t_start = make_sequence_windows(df_norm, WINDOW_S, STRIDE_S, FPS)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(np.full(len(X), int(label_name == "drowsy"), dtype=np.int64))
        all_subjects.append(np.full(len(X), subject))
        all_usable.append(np.full(len(X), subject not in EXCLUDE_FROM_CLASSIFIER))
        all_t_start.append(t_start)
        print(f"{subject}/{label_name}: {len(X)} sequences, shape={X.shape}")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)
    usable = np.concatenate(all_usable)
    t_start = np.concatenate(all_t_start)

    out_path = FEATURES_DIR / "sequences.npz"
    np.savez_compressed(
        out_path, X=X, y=y, subjects=subjects, usable=usable, t_start=t_start, channels=list(NORM_COLS)
    )
    print(f"\nTotal sequences: {len(X)}, shape={X.shape}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
