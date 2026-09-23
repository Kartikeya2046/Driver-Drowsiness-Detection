"""MediaPipe FaceLandmarker wrapper: frame -> (landmarks_xyz, transform_matrix) or None.

Shared by offline extraction and the live loop. Do not create FaceLandmarker
instances elsewhere.
"""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task"


class FaceDetector:
    def __init__(self, running_mode: str = "VIDEO"):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run scripts/download_model.py first."
            )
        mode = {"VIDEO": mp_vision.RunningMode.VIDEO, "IMAGE": mp_vision.RunningMode.IMAGE}[running_mode]
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mode,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mode = running_mode
        self._next_ts_ms = 0  # VIDEO mode requires a monotonically increasing
        # stream on one landmarker instance; owned here so callers processing
        # multiple unrelated clips through the same detector can't violate it.

    def detect(self, frame_bgr: np.ndarray):
        """Returns (landmarks_xyz (478,3), transform_matrix (4,4)) or None if no face found."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._mode == "VIDEO":
            result = self._landmarker.detect_for_video(mp_image, self._next_ts_ms)
            self._next_ts_ms += 1
        else:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        points = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
        matrix = (
            np.array(result.facial_transformation_matrixes[0])
            if result.facial_transformation_matrixes
            else None
        )
        return points, matrix

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
