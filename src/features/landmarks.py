"""Per-frame landmark -> geometric feature computation.

Single source of truth for EAR/MAR/head-pose math, shared by offline
training (extract_features.py) and the live loop (src/realtime/).
Do not reimplement this logic elsewhere.
"""
import numpy as np

# MediaPipe FaceMesh (478-landmark) index sets, standard convention.
LEFT_EYE_V = [(159, 145), (160, 144)]
LEFT_EYE_H = (33, 133)
RIGHT_EYE_V = [(386, 374), (387, 373)]
RIGHT_EYE_H = (263, 362)
MOUTH_V = (13, 14)
MOUTH_H = (78, 308)


def _dist(p, a, b) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


def eye_aspect_ratio(points: np.ndarray, vertical_pairs, horizontal_pair) -> float:
    """EAR = mean(vertical distances) / horizontal distance. Higher = more open."""
    horiz = _dist(points, *horizontal_pair)
    if horiz == 0:
        return 0.0
    vert = np.mean([_dist(points, a, b) for a, b in vertical_pairs])
    return float(vert / horiz)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    horiz = _dist(points, *MOUTH_H)
    if horiz == 0:
        return 0.0
    return _dist(points, *MOUTH_V) / horiz


def head_pose_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Extract (pitch, yaw, roll) in degrees from a 4x4 facial transformation matrix."""
    r = matrix[:3, :3]
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = np.arctan2(r[1, 0], r[0, 0])
    else:
        pitch = np.arctan2(-r[1, 2], r[1, 1])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = 0.0
    return tuple(np.degrees([pitch, yaw, roll]))


def frame_features(landmarks_xyz: np.ndarray, transform_matrix: np.ndarray | None) -> dict:
    """landmarks_xyz: (478, 3) array of normalized [0,1] x,y and relative z.
    transform_matrix: 4x4 facial transformation matrix, or None if unavailable.
    Returns a flat dict of per-frame features. NaN for pose if matrix missing.
    """
    left_ear = eye_aspect_ratio(landmarks_xyz, LEFT_EYE_V, LEFT_EYE_H)
    right_ear = eye_aspect_ratio(landmarks_xyz, RIGHT_EYE_V, RIGHT_EYE_H)
    ear = (left_ear + right_ear) / 2.0
    mar = mouth_aspect_ratio(landmarks_xyz)

    if transform_matrix is not None:
        pitch, yaw, roll = head_pose_from_matrix(transform_matrix)
    else:
        pitch = yaw = roll = float("nan")

    return {
        "ear_left": left_ear,
        "ear_right": right_ear,
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }
