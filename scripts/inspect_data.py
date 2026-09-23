"""Inspect UTA-RLDD (face-cropped) raw dataset: folder layout, label mapping,
per-video fps/resolution/duration, and per-subject chunk counts.

Usage: python scripts/inspect_data.py
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_DIR = RAW_DIR / "UTA-RLDD Face Cropped Video"
LABEL_MAP = {"0": "non_drowsy", "10": "drowsy"}


def video_props(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = n_frames / fps if fps else 0.0
    return {"fps": fps, "n_frames": n_frames, "width": w, "height": h, "duration_s": duration}


def main() -> None:
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset dir not found: {DATASET_DIR}")

    lengths = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    print(f"Length variants: {lengths}\n")

    # Use len10 as the canonical variant for label mapping / subject inventory.
    canonical = "len10" if "len10" in lengths else lengths[0]
    canonical_dir = DATASET_DIR / canonical

    subjects = sorted(p.name for p in canonical_dir.iterdir() if p.is_dir())
    print(f"Subjects in {canonical}: {len(subjects)} ({subjects[0]}..{subjects[-1]})\n")

    label_counts = defaultdict(int)
    subjects_missing_label = defaultdict(list)
    chunk_counts = {}  # subject -> {label: n_chunks}
    sample_props = {}  # length_variant -> video_props of first file found

    for subj in subjects:
        subj_dir = canonical_dir / subj
        present_labels = sorted(p.name for p in subj_dir.iterdir() if p.is_dir())
        chunk_counts[subj] = {}
        for label in ("0", "10"):
            if label not in present_labels:
                subjects_missing_label[label].append(subj)
                continue
            n_files = len(list((subj_dir / label).glob("*.mp4")))
            chunk_counts[subj][label] = n_files
            label_counts[label] += n_files

    print("=== Label mapping (folder name -> class) ===")
    for k, v in LABEL_MAP.items():
        print(f"  {k} -> {v}")
    print()

    print("=== Label coverage (canonical variant) ===")
    for label, name in LABEL_MAP.items():
        missing = subjects_missing_label[label]
        print(f"  {name}: {len(subjects) - len(missing)}/{len(subjects)} subjects have it")
        if missing:
            print(f"    missing for subjects: {missing}")
    print()

    print("=== Chunk count sanity (first 5 subjects) ===")
    for subj in subjects[:5]:
        print(f"  {subj}: {chunk_counts[subj]}")
    print()

    print("=== Video properties (one sample per length variant) ===")
    for length in lengths:
        length_dir = DATASET_DIR / length
        first_subj = sorted(p.name for p in length_dir.iterdir() if p.is_dir())[0]
        first_label_dir = sorted(p for p in (length_dir / first_subj).iterdir() if p.is_dir())[0]
        sample_file = sorted(first_label_dir.glob("*.mp4"))[0]
        props = video_props(sample_file)
        sample_props[length] = props
        print(f"  {length} sample ({sample_file.relative_to(DATASET_DIR)}): {props}")
    print()

    print("=== Cross-check: total duration per subject should match across length variants ===")
    subj = subjects[0]
    for length in lengths:
        length_dir = DATASET_DIR / length / subj / "0"
        files = sorted(length_dir.glob("*.mp4"))
        total = sum(video_props(f)["duration_s"] for f in files[:3])  # sample first 3 to keep it fast
        print(f"  {length}: first 3 chunks total duration = {total:.2f}s (n_chunks={len(files)})")

    out = {
        "lengths": lengths,
        "n_subjects": len(subjects),
        "label_counts": dict(label_counts),
        "subjects_missing_label": dict(subjects_missing_label),
        "sample_video_props": sample_props,
    }
    out_path = RAW_DIR.parent.parent / "data_inspection_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
