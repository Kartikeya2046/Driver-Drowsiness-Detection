"""features/*.parquet -> features/forecast_cal{N}.npz: per-5 s aggregate features for the
Phase 3 forecaster. Same calibration as the classifier (median shift from the first N s);
tries to include subject 42 (alert-only), but its 60 s video is shorter than the 120 s
calibration reserve so it contributes nothing.

Rows are grouped by video (video_id) and ordered by step, so a forecaster sample is a
slice of consecutive rows sharing a video_id.

Usage: python -m src.features.build_forecast_dataset [calib_s]
"""
import sys

import numpy as np

from src.features.build_dataset import FPS, TAB_COLS
from src.features.windows import FEATURES_DIR, iter_calibrated_videos, make_windows

STEP_S = 5.0


def build(calib_s: int = 60) -> None:
    X, video_id, step, y, subjects, t_start, names = [], [], [], [], [], [], []
    for subject, label_name, df, ear_thr in iter_calibrated_videos(calib_s, FPS, include_unpaired=True):
        summary, _ = make_windows(df, STEP_S, STEP_S, FPS, ear_thr)
        n = len(summary)
        if n == 0:  # subject 42's alert video (60 s) is shorter than the calibration reserve
            continue
        X.append(summary[TAB_COLS].to_numpy(dtype=np.float32))
        video_id.append(np.full(n, len(names)))
        step.append(np.arange(n))
        y.append(np.full(n, int(label_name == "drowsy")))
        subjects.append(np.full(n, subject))
        t_start.append(summary["t_start"].to_numpy())
        names.append(f"{subject}_{label_name}")

    out = FEATURES_DIR / f"forecast_cal{calib_s}.npz"
    np.savez_compressed(
        out,
        X=np.concatenate(X),
        video_id=np.concatenate(video_id),
        step=np.concatenate(step),
        y=np.concatenate(y),
        subjects=np.concatenate(subjects),
        t_start=np.concatenate(t_start),
        video_names=np.array(names),
        cols=np.array(TAB_COLS),
    )
    lens = np.array([len(x) for x in X])
    print(f"cal={calib_s}s: {lens.sum()} steps of {STEP_S:.0f} s, {len(names)} videos "
          f"(steps/video min {lens.min()}, median {int(np.median(lens))}, max {lens.max()}) -> {out.name}")


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
