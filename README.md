# DriveGuard: Predictive Driver Drowsiness Detection (Cyber-Physical System)

DriveGuard is a cyber-physical system that watches a driver through a webcam, estimates drowsiness, and responds with physical alerts before the driver becomes critically impaired.

**Cyber side:** Each video frame goes through MediaPipe Face Mesh to extract facial features: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), PERCLOS, blink statistics, and head pose. A GRU classifier scores 45-second windows and outputs a drowsiness probability. Each subject is calibrated against a short alert baseline (60 s) before scoring.

**Physical side:** A Python process streams risk levels over serial (pySerial) to an Arduino that drives an RGB LED, a vibration motor, and a buzzer in graduated escalation. A driver acknowledgement button resets the alert.

**Risk levels**
- 0, safe: green LED
- 1, rising risk: yellow LED
- 2, high risk: vibration
- 3, critical: buzzer

## Results

UTA-RLDD, 60 subjects, subject-grouped evaluation.

- **GRU (deployed):** 78.8% accuracy, AUC 0.858 with a 60 s calibration. Earlier numbers using full-video normalization were optimistic and are not reported as headline results.
- **Ensembles** reach about 80% but cost 4x the inference time, so they are reported as a ceiling only.
- **Forecaster:** trained to predict the next 60 s, it barely beats a persistence baseline, because the dataset has no alert-to-drowsy transitions. Lead time is evaluated on synthetic ramped transitions instead, with hard splices as a negative control.

Full numbers and figures are in [`REPORT.md`](REPORT.md) and `results/figures/`.

## Limitations

- Several subjects are near chance for every model, because some show no measurable eye-feature change when labeled drowsy.
- The hardware loop is written and tested in software; the physical Arduino demo is pending parts.

## Stack

Python, PyTorch, MediaPipe, OpenCV, XGBoost, pySerial, Arduino (C++)

## Dataset

[UTA-RLDD face-cropped videos (Kaggle)](https://www.kaggle.com/datasets/mathiasviborg/uta-rldd-videos-cropped-by-faces). The raw data is not included in this repo.

## Usage

```bash
python -m src.realtime.live_loop --source 0 --port COM3   # webcam + Arduino
python -m src.realtime.live_loop --source video.mp4       # video file, no serial
```

Built for the ICPS (Introduction to Cyber-Physical Systems) course.
