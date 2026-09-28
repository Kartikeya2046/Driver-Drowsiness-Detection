# Starting the system

Quick reference for turning the live drowsiness detector on. For everything else
(training, evaluation, what's been done), see `CHECKPOINT.md` and `PROGRESS.md`.

## Test it live, right now (laptop webcam, no hardware needed)

```
cd "D:\ICPS project"
.venv\Scripts\python.exe -m src.realtime.live_loop --source 0 --show
```

- Calibrates for 60s first (sit normally, face the camera) — wait for `Calibration
  done.` in the terminal before it starts making decisions.
- A window opens showing EAR / PERCLOS / p_now / alert level, with a color-coded
  border (green=0, yellow=1, orange=2, red=3).
- No `--port` given → alert levels print to the console (`[serial] R:<level>`)
  instead of being sent anywhere — safe to run with no Arduino attached.
- Press `q` (window focused) to stop. Saves a decision log to
  `results/live_log_<timestamp>.csv` and prints a latency summary.

## Once the hardware arrives (Arduino on a serial port)

```
.venv\Scripts\python.exe -m src.realtime.live_loop --source 0 --port COM3 --show
```

Replace `COM3` with the Arduino's actual port (Device Manager → Ports). Drop
`--show` for a headless run (e.g. embedded in a car, no monitor).

## Other useful flags

| Flag | Purpose |
|---|---|
| `--source path\to\video.mp4` | Run against a recorded file instead of a live camera |
| `--calib 30` | Override the calibration length (default: whatever `models/classifier_gru.pt` was trained with, currently 60s) |
| `--max-s 120` | Stop after N seconds (useful for a quick test/benchmark) |

## First-time setup (already done on this machine, listed for a fresh clone)

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\download_model.py     # MediaPipe face_landmarker.task
```

`models/classifier_gru.pt` (the deploy model) is already committed to the repo —
nothing to train before testing live.
