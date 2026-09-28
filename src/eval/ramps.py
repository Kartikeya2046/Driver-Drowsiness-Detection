"""Synthetic alert -> drowsy transitions for the Phase 3 lead-time evaluation.

Each ramp is built from one subject's own calibrated per-frame data: an alert lead-in,
then a ramp during which 5-10 s chunks are drawn from the drowsy video with a
probability that rises 0 -> 1 (linear or sigmoid), then a drowsy tail. Onset is the
nominal time the drowsy probability crosses 0.5 (the ramp's midpoint). The hard splice
(ramp duration 0) has no precursor and is the negative control: expected lead time ~ 0.

Usage (CPU env): python -m src.eval.ramps   -> features/ramps_cal60.npz
"""
import numpy as np

from src.features.windows import FEATURES_DIR, NORM_COLS, iter_calibrated_videos

CALIB_S = 60
FPS = 10
LEAD_IN_S, TAIL_S = 180, 120
DURATIONS_S = (0, 120, 300, 600)  # 0 = hard splice (negative control)
SHAPES = ("linear", "sigmoid")
CHUNK_S = (5, 10)
RAMPS_PER_CONFIG = 3  # independent random draws per subject x duration x shape


def drowsy_prob(u: np.ndarray, shape: str) -> np.ndarray:
    """u in [0,1] = progress through the ramp."""
    if shape == "linear":
        return u
    return 1 / (1 + np.exp(-12 * (u - 0.5)))  # sigmoid; ~0.002 at u=0, ~0.998 at u=1


def build_ramp(alert: np.ndarray, drowsy: np.ndarray, duration_s: int, shape: str, rng):
    """alert/drowsy: (T, 5) calibrated frames. Returns (frames, onset_s)."""
    total_s = LEAD_IN_S + duration_s + TAIL_S
    onset_s = LEAD_IN_S + duration_s / 2
    parts, t = [], 0.0
    while t < total_s:
        n = int(rng.integers(CHUNK_S[0], CHUNK_S[1] + 1)) * FPS
        mid = t + n / FPS / 2
        if mid < LEAD_IN_S:
            p = 0.0
        elif mid >= LEAD_IN_S + duration_s:
            p = 1.0
        else:
            p = float(drowsy_prob(np.array((mid - LEAD_IN_S) / duration_s), shape))
        src = drowsy if rng.random() < p else alert
        s = int(rng.integers(0, len(src) - n + 1))
        parts.append(src[s : s + n])
        t += n / FPS
    return np.concatenate(parts), onset_s


def build_all(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    videos = {}
    for subject, label, df, ear_thr in iter_calibrated_videos(CALIB_S, FPS):
        videos.setdefault(subject, {})[label] = (df[list(NORM_COLS)].to_numpy(np.float32), ear_thr)

    frames, meta = [], []
    for subject, v in videos.items():
        alert, ear_thr = v["non_drowsy"]
        drowsy, _ = v["drowsy"]
        for dur in DURATIONS_S:
            for shape in (SHAPES if dur else ("linear",)):  # splice has one shape
                for rep in range(RAMPS_PER_CONFIG):
                    x, onset = build_ramp(alert, drowsy, dur, shape, rng)
                    frames.append(x)
                    meta.append((subject, dur, shape, rep, onset, ear_thr, len(x)))

    lens = np.array([m[-1] for m in meta])
    np.savez_compressed(
        FEATURES_DIR / f"ramps_cal{CALIB_S}.npz",
        frames=np.concatenate(frames),
        offsets=np.concatenate([[0], np.cumsum(lens)]),
        subject=np.array([m[0] for m in meta]),
        duration_s=np.array([m[1] for m in meta]),
        shape=np.array([m[2] for m in meta]),
        rep=np.array([m[3] for m in meta]),
        onset_s=np.array([m[4] for m in meta], dtype=np.float32),
        ear_threshold=np.array([m[5] for m in meta], dtype=np.float32),
        channels=np.array(NORM_COLS),
    )
    print(f"{len(meta)} ramps, {len(videos)} subjects, {lens.sum() / FPS / 3600:.1f} h total "
          f"-> ramps_cal{CALIB_S}.npz", flush=True)


def load_ramps():
    """Yields (subject, duration_s, shape, rep, onset_s, ear_threshold, frames (T,5))."""
    z = np.load(FEATURES_DIR / f"ramps_cal{CALIB_S}.npz")
    d = {k: z[k] for k in z.files}  # NpzFile re-decompresses on every access, so read each array once
    o = d["offsets"]
    for i in range(len(d["subject"])):
        yield (str(d["subject"][i]), int(d["duration_s"][i]), str(d["shape"][i]), int(d["rep"][i]),
               float(d["onset_s"][i]), float(d["ear_threshold"][i]), d["frames"][o[i] : o[i + 1]])


if __name__ == "__main__":
    build_all()
