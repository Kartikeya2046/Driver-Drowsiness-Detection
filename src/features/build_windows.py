"""features/*.parquet (per-frame) -> features/windows.parquet (window-level,
normalized, ready for subject-grouped CV).

Per-subject normalization: z-score EAR/MAR/pitch/yaw using each subject's own
non-drowsy video stats (their calibration baseline), matching the live
system's 30s calibration step.

Usage: python -m src.features.build_windows
"""
from pathlib import Path

import pandas as pd

from src.features.windows import iter_normalized_subject_labels, make_windows

FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "features"
WINDOW_S = 45.0
STRIDE_S = 5.0
FPS = 10.0
EXCLUDE_FROM_CLASSIFIER = {"42"}  # no drowsy video; may still be used for forecaster


def main() -> None:
    all_windows = []
    for subject, label_name, df_norm, norm_thresh in iter_normalized_subject_labels(FPS, WINDOW_S):
        windows = make_windows(df_norm, WINDOW_S, STRIDE_S, FPS, norm_thresh)
        windows["subject"] = subject
        windows["label"] = label_name
        windows["label_binary"] = int(label_name == "drowsy")
        windows["usable_for_classifier"] = subject not in EXCLUDE_FROM_CLASSIFIER
        all_windows.append(windows)
        print(f"{subject}/{label_name}: {len(windows)} windows")

    result = pd.concat(all_windows, ignore_index=True)
    out_path = FEATURES_DIR / "windows.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nTotal windows: {len(result)}")
    print(f"By label:\n{result['label'].value_counts()}")
    print(f"Subjects: {result['subject'].nunique()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
