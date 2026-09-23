"""features/*.parquet (per-frame) -> features/windows.parquet (window-level,
normalized, ready for subject-grouped CV).

Per-subject normalization: z-score EAR/MAR/pitch/yaw using each subject's own
non-drowsy video stats (their calibration baseline), matching the live
system's 30s calibration step.

Usage: python -m src.features.build_windows
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.windows import blink_threshold, make_windows

FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "features"
WINDOW_S = 45.0
STRIDE_S = 5.0
FPS = 10.0
EXCLUDE_FROM_CLASSIFIER = {"42"}  # no drowsy video; may still be used for forecaster


def normalize_frame_df(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("ear", "mar", "pitch", "yaw", "roll"):
        mu = baseline[col].mean()
        sigma = baseline[col].std()
        sigma = sigma if sigma > 1e-6 else 1.0
        df[col] = (df[col] - mu) / sigma
    return df


def main() -> None:
    subject_files = sorted(FEATURES_DIR.glob("*_non_drowsy.parquet")) + sorted(
        FEATURES_DIR.glob("*_drowsy.parquet")
    )
    subjects = sorted({f.stem.split("_")[0] for f in subject_files})

    all_windows = []
    for subject in subjects:
        nd_path = FEATURES_DIR / f"{subject}_non_drowsy.parquet"
        if not nd_path.exists():
            print(f"  skip {subject}: no non_drowsy file (needed as normalization baseline)")
            continue
        baseline_df = pd.read_parquet(nd_path)
        baseline_valid = baseline_df[baseline_df["valid"]]
        if len(baseline_valid) < FPS * 30:
            print(f"  skip {subject}: baseline too short ({len(baseline_valid)} valid frames)")
            continue

        ear_thresh = blink_threshold(baseline_valid["ear"].to_numpy())

        for label_name, path in (
            ("non_drowsy", nd_path),
            ("drowsy", FEATURES_DIR / f"{subject}_drowsy.parquet"),
        ):
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df = df[df["valid"]].reset_index(drop=True)
            if len(df) < WINDOW_S * FPS:
                print(f"  skip {subject}/{label_name}: too short for one window")
                continue

            df_norm = normalize_frame_df(df, baseline_valid)
            # blink threshold must be applied in normalized EAR space
            norm_thresh = (ear_thresh - baseline_valid["ear"].mean()) / max(
                baseline_valid["ear"].std(), 1e-6
            )
            windows = make_windows(df_norm, WINDOW_S, STRIDE_S, FPS, norm_thresh)
            windows["subject"] = subject
            windows["label"] = label_name
            windows["label_binary"] = int(label_name == "drowsy")
            windows["usable_for_classifier"] = subject not in EXCLUDE_FROM_CLASSIFIER
            all_windows.append(windows)
            print(f"{subject}/{label_name}: {len(windows)} windows")

    result = pd.concat(all_windows, ignore_index=True)
    out_path = FEATURES_DIR.parent / "features" / "windows.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nTotal windows: {len(result)}")
    print(f"By label:\n{result['label'].value_counts()}")
    print(f"Subjects: {result['subject'].nunique()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
