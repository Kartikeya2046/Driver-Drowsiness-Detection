"""Full-frame webcam video -> per-frame features, mirroring the 224x224 face-cropped
training data (Phase 4 domain-shift fix: training frames are already face crops; a raw
webcam frame is not, so crop to a square face box and resize to 224x224 before the same
landmark/feature code the training pipeline uses).

Two FaceDetector instances, not one, and this is deliberate: the bbox pass uses IMAGE
mode (stateless, one frame at a time) because consecutive frames feed it wildly
different crops (whatever box that frame's face lands in); VIDEO mode's landmarker
does temporal smoothing that assumes each call is one small step from the last, and
mixing that with a jittery bbox pass would degrade it. The second pass, on the
224x224 crops themselves, IS a smooth video (same framing as training's continuous
crops) and gets VIDEO mode, matching how extract_features.py runs on training data.
"""
import cv2
import numpy as np
import pandas as pd

from src.features.detector import FaceDetector
from src.features.landmarks import frame_features


class LiveFeatureExtractor:
    """Streaming version of the bbox-then-crop pipeline below: two FaceDetector
    instances that live for the whole session (a webcam feed or a video file played
    as one), instead of process_webcam_video's open-per-call pair. Shared by the file
    based batch function (below) and the live loop, so there's exactly one
    implementation of "how a raw frame becomes a training-shaped feature row"."""

    def __init__(self, margin: float = 0.4):
        self.bbox_det = FaceDetector(running_mode="IMAGE")
        self.crop_det = FaceDetector(running_mode="VIDEO")
        self.margin = margin

    def process(self, frame_bgr: np.ndarray) -> dict:
        crop = face_crop_224(self.bbox_det, frame_bgr, self.margin)
        detection = self.crop_det.detect(crop) if crop is not None else None
        if detection is None:
            return {"face_found": False}
        points, matrix = detection
        feats = frame_features(points, matrix)
        feats["face_found"] = True
        return feats

    def close(self):
        self.bbox_det.close()
        self.crop_det.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def face_crop_224(bbox_detector: FaceDetector, frame_bgr: np.ndarray, margin: float = 0.4):
    """Square-crop the face bounding box (landmark extent + margin) and resize to
    224x224. Returns None if no face is found in the raw frame."""
    det = bbox_detector.detect(frame_bgr)
    if det is None:
        return None
    points, _ = det
    h, w = frame_bgr.shape[:2]
    xs, ys = points[:, 0] * w, points[:, 1] * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    size = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin)
    half = size / 2
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (224, 224))


def process_webcam_video(video_path, target_fps: float = 10.0, margin: float = 0.4) -> pd.DataFrame:
    """Nearest-frame subsample to target_fps, face-crop, landmark. Returns a raw
    per-frame df in the same shape extract_features.py produces (before gap
    interpolation) - pass through interpolate_gaps() for a directly comparable df."""
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    rows = []
    frame_idx, next_tick = 0, 0
    with LiveFeatureExtractor(margin) as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx / native_fps >= next_tick / target_fps:
                feats = extractor.process(frame)
                feats["t"] = next_tick / target_fps
                rows.append(feats)
                next_tick += 1
            frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)
