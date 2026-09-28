"""Phase 4 domain-shift check: does the face-cropped webcam pipeline (src/features/
webcam.py) produce EAR/MAR/pose distributions comparable to the training data's?

Compares *calibrated* features (median-shifted from each session's own first 60 s),
not raw ones: that's what the classifier actually sees, and it's the only fair
comparison — pooling raw features across 59 different training subjects would mix
genuine person-to-person variation into what's supposed to be a camera/framing check,
and swamp any real domain-shift signal.

Usage (CPU env): python -m src.eval.domain_shift <path/to/recording.mp4>
"""
import sys
from pathlib import Path

import pandas as pd

from src.features.extract_features import interpolate_gaps
from src.features.webcam import process_webcam_video
from src.features.windows import NORM_COLS, calibrate, apply_calibration, iter_calibrated_videos

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT / "features"
COLS = ("ear", "mar", "pitch", "yaw", "roll")
CALIB_S = 60


def training_reference() -> pd.DataFrame:
    """All training videos' calibrated features (calib_s=60, same as deploy), each
    video's own calibration window already excluded by iter_calibrated_videos."""
    dfs = [df for _, _, df, _ in iter_calibrated_videos(CALIB_S)]
    return pd.concat(dfs, ignore_index=True)


def calibrated_webcam(df: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
    valid = df[df["valid"]].reset_index(drop=True)
    cal = calibrate(valid.iloc[: int(CALIB_S * fps)])
    evald = valid.iloc[int(CALIB_S * fps):].reset_index(drop=True)
    return apply_calibration(evald, cal)


def compare(webcam: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in COLS:
        w, t = webcam[col], training[col]
        rows.append({
            "feature": col,
            "webcam_mean": w.mean(), "train_mean": t.mean(),
            "webcam_std": w.std(), "train_std": t.std(),
            "webcam_p05": w.quantile(0.05), "train_p05": t.quantile(0.05),
            "webcam_p95": w.quantile(0.95), "train_p95": t.quantile(0.95),
        })
    return pd.DataFrame(rows)


def main(video_path: str) -> None:
    print(f"Processing {video_path} ...", flush=True)
    raw = process_webcam_video(video_path)
    df = interpolate_gaps(raw)
    out = FEATURES_DIR / "webcam_demo.parquet"
    df.to_parquet(out, index=False)
    drop_rate = 1.0 - df["valid"].mean()
    print(f"{len(df)} frames, drop_rate={drop_rate:.3f} -> {out}", flush=True)

    webcam = calibrated_webcam(df)
    training = training_reference()
    table = compare(webcam, training)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    table.to_csv(FEATURES_DIR.parent / "results" / "domain_shift_report.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
