"""Extract per-frame features for every subject/label video, save one file per
(subject, label) to features/. Stitches len60 chunks back into one continuous
timeline (they're splits of a single source video, not separate clips).

Usage: python -m src.features.extract_features
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.detector import FaceDetector
from src.features.landmarks import frame_features

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "UTA-RLDD Face Cropped Video" / "len60"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "features"
LABEL_MAP = {"0": "non_drowsy", "10": "drowsy"}
MAX_GAP_S = 0.5  # interpolate missing-face gaps shorter than this; longer -> invalid


def chunk_sort_key(path: Path) -> int:
    m = re.match(r"\d+_\d+_(\d+)\.mp4$", path.name)
    return int(m.group(1))


def extract_video_features(detector: FaceDetector, video_path: Path) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detection = detector.detect(frame)
        if detection is None:
            rows.append({"t": frame_idx / fps, "face_found": False})
        else:
            points, matrix = detection
            feats = frame_features(points, matrix)
            feats["t"] = frame_idx / fps
            feats["face_found"] = True
            rows.append(feats)
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)


def stitch_subject_label(detector: FaceDetector, subject_dir: Path) -> pd.DataFrame:
    chunks = sorted(subject_dir.glob("*.mp4"), key=chunk_sort_key)
    dfs = []
    t_offset = 0.0
    for chunk_path in chunks:
        df = extract_video_features(detector, chunk_path)
        if df.empty:
            continue
        df["t"] = df["t"] + t_offset
        dfs.append(df)
        t_offset = df["t"].iloc[-1] + (1.0 / 10.0)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def interpolate_gaps(df: pd.DataFrame, max_gap_s: float = MAX_GAP_S) -> pd.DataFrame:
    """Interpolate short missing-face gaps; mark frames in longer gaps as invalid.
    Never silently zero-fill."""
    feature_cols = ["ear_left", "ear_right", "ear", "mar", "pitch", "yaw", "roll"]
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    found = df["face_found"].to_numpy()
    n = len(df)
    valid = found.copy()

    # find runs of missing frames; mark long runs invalid, short runs interpolatable
    i = 0
    frame_period = df["t"].diff().median() if n > 1 else 0.1
    max_gap_frames = max(1, round(max_gap_s / frame_period)) if frame_period else 1
    while i < n:
        if not found[i]:
            j = i
            while j < n and not found[j]:
                j += 1
            gap_len = j - i
            if gap_len > max_gap_frames:
                valid[i:j] = False
            i = j
        else:
            i += 1

    for col in feature_cols:
        df[col] = df[col].interpolate(limit_area="inside")

    df["valid"] = valid & df[feature_cols].notna().all(axis=1)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())

    report_rows = []
    with FaceDetector(running_mode="VIDEO") as detector:
        for subject in subjects:
            for label_code, label_name in LABEL_MAP.items():
                subject_label_dir = RAW_DIR / subject / label_code
                if not subject_label_dir.exists():
                    continue

                raw_df = stitch_subject_label(detector, subject_label_dir)
                if raw_df.empty:
                    print(f"  WARNING: no frames extracted for {subject}/{label_code}")
                    continue

                df = interpolate_gaps(raw_df)
                df["subject"] = subject
                df["label"] = label_name

                out_path = OUT_DIR / f"{subject}_{label_name}.parquet"
                df.to_parquet(out_path, index=False)

                drop_rate = 1.0 - df["valid"].mean()
                report_rows.append(
                    {
                        "subject": subject,
                        "label": label_name,
                        "n_frames": len(df),
                        "drop_rate": drop_rate,
                        "duration_s": df["t"].iloc[-1] if len(df) else 0.0,
                    }
                )
                print(f"{subject}/{label_name}: {len(df)} frames, drop_rate={drop_rate:.3f}")

    report = pd.DataFrame(report_rows)
    report_path = OUT_DIR.parent / "feature_extraction_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\nWrote report to {report_path}")
    print(f"Mean drop rate: {report['drop_rate'].mean():.3f}")
    print(f"Subjects with drop_rate > 0.2: {report[report['drop_rate'] > 0.2]['subject'].tolist()}")


if __name__ == "__main__":
    main()
