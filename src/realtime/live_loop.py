"""The live drowsiness loop: webcam (or a video file, for testing/demo playback) ->
face-crop features (src/features/webcam.py) -> calibrate on the first calib_s seconds
-> 45 s window -> deploy GRU -> risk level -> serial, at ~2 Hz (context.md protocol).

Runs fully without the Arduino attached (serial_client.SerialClient logs instead of
raising) - hardware is deferred, not ordered yet, but the software loop shouldn't be.

Usage: python -m src.realtime.live_loop [--source 0 | path/to/video.mp4] [--port COM3]
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.features.webcam import LiveFeatureExtractor
from src.features.windows import NORM_COLS, calibrate
from src.models.classifier_gru import DrowsinessGRU
from src.realtime.risk import risk_level
from src.realtime.serial_client import SerialClient

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "classifier_gru.pt"
RESULTS_DIR = ROOT / "results"
FPS = 10.0  # processing rate; source frames are subsampled to this, matching training
WINDOW_S = 45.0
DECISION_INTERVAL_S = 1.0
SERIAL_HZ = 2.0
MAX_GAP_S = 0.5  # forward-fill face-loss gaps shorter than this (matches training's
# interpolation cutoff); longer gaps clear the window and pause alerting rather than
# alert on stale data.
LEVEL_COLOR = {0: (0, 180, 0), 1: (0, 200, 200), 2: (0, 120, 255), 3: (0, 0, 255)}  # BGR: green/yellow/orange/red


def draw_overlay(frame, ear: float, perclos: float, p_now: float | None, level: int):
    """Minimal live dashboard (Phase 4 checklist): EAR/PERCLOS, current risk, alert
    level, as an on-frame overlay - no separate Streamlit app/process for what a few
    cv2.putText calls already show during an in-person demo."""
    color = LEVEL_COLOR[level]
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 8)
    lines = [f"EAR: {ear:.3f}", f"PERCLOS: {perclos:.2f}",
             f"p_now: {p_now:.2f}" if p_now is not None else "p_now: --", f"LEVEL: {level}"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (16, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


class LiveLoop:
    def __init__(self, source, port: str | None = None, calib_s: float | None = None):
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)  # our own checkpoint (state_dict + a numpy array); trusted
        self.calib_s = calib_s or ckpt["calib_s"]
        self.model = DrowsinessGRU()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.scale = ckpt["channel_scale"]

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source: {source}")
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or FPS
        self.extractor = LiveFeatureExtractor()
        self.serial = SerialClient(port)
        self.log_rows: list[dict] = []
        self._frame_idx = -1

    def _next_tick_frame(self, tick: int):
        """Reads source frames until the one at/after tick/FPS seconds; None at EOF."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return None
            self._frame_idx += 1
            if self._frame_idx / self.native_fps >= tick / FPS:
                return frame

    @staticmethod
    def _fill(feats: dict, last_valid: dict | None, gap: int, max_gap_frames: int):
        if feats.get("face_found"):
            return {c: feats[c] for c in NORM_COLS}
        if last_valid is not None and gap < max_gap_frames:
            return last_valid  # forward-fill a short face-loss gap
        return None

    def calibrate(self) -> dict:
        max_gap_frames = int(round(MAX_GAP_S * FPS))
        buffer, last_valid, gap, tick = [], None, 0, 0
        n_needed = int(self.calib_s * FPS)
        print(f"Calibrating for {self.calib_s:.0f}s...", flush=True)
        while len(buffer) < n_needed:
            frame = self._next_tick_frame(tick)
            if frame is None:
                raise RuntimeError(f"source ended during calibration ({len(buffer)}/{n_needed} frames)")
            row = self._fill(self.extractor.process(frame), last_valid, gap, max_gap_frames)
            if row is not None:
                last_valid, gap = row, 0
                buffer.append(row)
            else:
                gap += 1
            tick += 1
        self._tick = tick
        self._last_valid, self._gap = last_valid, gap
        cal = calibrate(pd.DataFrame(buffer))
        print("Calibration done.", flush=True)
        return cal

    def run(self, max_s: float | None = None, show: bool = False):
        cal = self.calibrate()
        max_gap_frames = int(round(MAX_GAP_S * FPS))
        buf = deque(maxlen=int(WINDOW_S * FPS))
        last_valid, gap, tick = self._last_valid, self._gap, self._tick
        level, last_decision_t, last_serial_t = 0, -DECISION_INTERVAL_S, 0.0
        p_now_display, ear_display = None, 0.0
        self.tick_latency_s: list[float] = []  # frame -> features -> (risk level if decided), per tick

        while True:
            t = tick / FPS
            if max_s is not None and t > max_s:
                break
            frame = self._next_tick_frame(tick)
            if frame is None:
                break
            tick_start = time.perf_counter()
            row = self._fill(self.extractor.process(frame), last_valid, gap, max_gap_frames)
            if row is not None:
                last_valid, gap = row, 0
                ear_display = row["ear"]
                buf.append(np.array([row[c] - cal["median"][c] for c in NORM_COLS], dtype=np.float32))
            else:
                gap += 1
                if gap > max_gap_frames:
                    buf.clear()  # lost the face for too long; don't alert on stale data

            ready = len(buf) == buf.maxlen
            if ready and t - last_decision_t >= DECISION_INTERVAL_S:
                x = (np.stack(buf) / self.scale).astype(np.float32)[None]
                with torch.no_grad():
                    p_now_display = torch.sigmoid(self.model(torch.from_numpy(x))).item()
                level = risk_level(p_now_display)
                last_decision_t = t
                self.log_rows.append({"t": t, "p_now": p_now_display, "level": level})

            if t - last_serial_t >= 1.0 / SERIAL_HZ:
                self.serial.send_level(level if ready else 0)
                last_serial_t = t
            if self.serial.poll_ack():
                print(f"[ack] t={t:.1f}s", flush=True)
            self.tick_latency_s.append(time.perf_counter() - tick_start)

            if show:
                perclos = float(np.mean(np.stack(buf)[:, 0] < cal["ear_threshold"])) if buf else 0.0
                cv2.imshow("ICPS live", draw_overlay(frame, ear_display, perclos, p_now_display, level))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            tick += 1

        if show:
            cv2.destroyAllWindows()
        self.serial.close()
        return pd.DataFrame(self.log_rows)

    def latency_report(self) -> dict:
        """Wall-clock cost of one processing tick (frame -> features -> decision/serial
        write, whatever ran that tick) - the frame->actuator latency up to the serial
        write; actual actuator response time is untestable without the Arduino attached.
        Also reports whether this pace can keep up with a live camera at FPS."""
        lat = np.array(self.tick_latency_s)
        budget = 1.0 / FPS
        return {
            "n_ticks": len(lat), "mean_ms": lat.mean() * 1000, "p95_ms": np.percentile(lat, 95) * 1000,
            "max_ms": lat.max() * 1000, "budget_ms": budget * 1000,
            "over_budget_frac": float((lat > budget).mean()),
        }

    def save_log(self, df: pd.DataFrame, name: str = "live_log") -> Path:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{name}_{int(time.time())}.csv"
        df.to_csv(path, index=False)
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or a video file path")
    ap.add_argument("--port", default=None, help="serial port, e.g. COM3; omit to log only")
    ap.add_argument("--calib", type=float, default=None, help="override the deploy model's calib_s")
    ap.add_argument("--max-s", type=float, default=None, help="stop after this many seconds (testing)")
    ap.add_argument("--show", action="store_true", help="live overlay window (EAR/PERCLOS/p_now/level); needs a display")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    loop = LiveLoop(source, port=args.port, calib_s=args.calib)
    log = loop.run(max_s=args.max_s, show=args.show)
    path = loop.save_log(log)
    print(f"{len(log)} decisions logged -> {path}")
    if len(log):
        print(log["level"].value_counts().sort_index().to_string())
    print("latency:", loop.latency_report())


if __name__ == "__main__":
    main()
