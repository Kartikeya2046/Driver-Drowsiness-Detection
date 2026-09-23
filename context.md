# Agent Context: Predictive Driver Drowsiness CPS

Read this before doing any work in this repo. Full phase-by-phase plan is in `plan.md`.

## What we are building

A **cyber-physical system** that predicts driver drowsiness 30–60 seconds before it becomes critical, and responds through physical hardware.

- **Cyber side:** webcam → MediaPipe Face Mesh → facial features (EAR, MAR, PERCLOS, blink stats, head pose) → ML models → risk level.
- **Physical side:** Arduino driving an RGB LED, vibration motor, and buzzer in graduated escalation, plus a driver acknowledgement button.
- Link: Python → pySerial → Arduino.

This is a two-person university course project for **ICPS (Intro to Cyber-Physical Systems)**. The course theme is **predictive technologies using ML/AI**, and the professor requires **some hardware integration**. The "predictive" aspect (forecasting, not just detecting) is the part most likely to be graded — protect it.

## Dataset facts (verified on the actual download 2026-09-23 — supersedes earlier assumptions)

- Source: https://www.kaggle.com/datasets/mathiasviborg/uta-rldd-videos-cropped-by-faces
- Face-cropped version of UTA-RLDD. Faces are cropped; **landmarks/features are NOT extracted** — we do that.
- 60 subjects, **binary labels: non-drowsy / drowsy**. No "low vigilant" class.
- **Subject 42 has no drowsy video** (confirmed: only label `0` folder exists for subject 42).
- Each subject's video is a **single state** per label — no within-video transitions. Source duration is **not** a fixed 10 min; it varies per subject (e.g. subject 01 ≈ 56×10s ≈ 9.3 min, subject 04 ≈ 71×10s ≈ 11.8 min).
- **Already resampled to a fixed 10 fps, already cropped to 224×224.** This Kaggle repackaging is NOT the raw variable-FPS/resolution UTA-RLDD — that description applies to the *original* dataset, not this download. **Plan's "resample by timestamp" step is unnecessary for training data** (still needed for the *live* webcam feed, which won't be a clean 10 fps).
- **Already pre-chunked** into fixed-length clips, in 5 parallel variants: `len5/len10/len20/len30/len60` (chunk length in seconds), same underlying content just cut differently. Use `len60` (fewest files, largest windows) as the primary source for window-based training; other variants are redundant re-cuts, not extra data — don't double-count them as separate samples.
- Folder layout: `data/raw/UTA-RLDD Face Cropped Video/<lenN>/<subject 01-60>/<label 0|10>/<subject>_<label>_<group>_<chunkIdx>.mp4` (len60 filenames omit the group field: `<subject>_<label>_<chunkIdx>.mp4`).
- Label folder mapping: `0` = non_drowsy, `10` = drowsy (not the original UTA-RLDD 0/5/10 three-level scale — only 0 and 10 are present here, consistent with binary labels).
- Full inspection report: `scripts/inspect_data.py` (regeneratable) → `data_inspection_report.json`.

## Core design (decided — don't change without flagging)

1. **Classifier** — drowsy now? Trained on windows (30–60 s) with binary labels. Baseline: XGBoost/RF on window stats; main: small GRU/LSTM/1D-CNN.
2. **Forecaster** — predicts next 30–60 s of features from last 60 s. Self-supervised (no labels). Must be compared against persistence and linear-trend baselines.
3. **Risk fusion** — `p_now` (classifier on current window) and `p_future` (classifier on forecast window) → level 0–3 with hysteresis.
   - 0 safe (green), 1 `p_future` rising (yellow), 2 `p_future` high (vibration), 3 `p_now` high (buzzer).
4. **Lead-time evaluation** — stitched non-drowsy→drowsy sequences per subject, plus self-recorded sessions with KSS (Karolinska Sleepiness Scale) ratings.

## Non-negotiable rules

- **Split by subject, never by window/frame.** 5-fold subject-grouped CV. Leakage makes results meaningless.
- **Resample videos by timestamp** to a fixed rate (~10 fps). Don't use "every Nth frame."
- **Per-subject normalization** of features (from non-drowsy video stats); live system uses a 30 s calibration.
- **One feature-extraction code path** shared by training and real-time inference. No reimplementation.
- **Missing faces:** interpolate gaps < 0.5 s; longer gaps → invalid windows. Never silently zero-fill. Log drop rates.
- **Never commit `data/`** (gitignored). Commit code, small feature files, and small model weights only.
- Exclude subject 42 from classifier evaluation; it may be used for forecaster training.
- Report negative results honestly (e.g. if the forecaster doesn't beat persistence).

## Tech stack

- Python, `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch` (PyTorch only — don't mix in TensorFlow), `pyserial`, `streamlit`.
- Arduino C++ (`hardware/*.ino`).
- Pinned `requirements.txt`.

## Hardware protocol

- Python → Arduino over serial, ASCII line `R:<level>\n`, level 0–3, ~2 Hz.
- Arduino **failsafe**: no message for 2 s → fault blink pattern.
- Button press → Arduino sends `ACK\n` back to Python; log it.
- Vibration motor must be driven through an NPN transistor (2N2222) with a flyback diode — never directly from a GPIO pin.
- Hardware is developed and tested with hardcoded levels independently of the ML models.

## Repo layout

```
data/           raw videos (gitignored)
features/       per-video extracted features
src/features/   landmark → feature code (shared by train + live)
src/models/     classifier, forecaster, risk fusion
src/realtime/   live loop, serial client, dashboard
src/eval/       metrics, lead-time eval, plots
hardware/       Arduino sketch + wiring
scripts/        download_data.py etc.
notebooks/      exploration only — logic lives in src/
```

## Deliverables

- **D1:** real-time feature pipeline + drowsiness classifier + working Arduino graduated alert over serial.
- **D2:** forecasting layer (30–60 s ahead) integrated into the closed loop with failsafe and acknowledgement, evaluated on lead time, false alarms/hour, and latency.

## How to work with the user

- Work autonomously. Make technical decisions yourself and document them briefly (in commit messages or a short `DECISIONS.md`).
- Only interrupt for: genuine blockers, decisions that risk the grade (e.g. dropping the predictive layer, changing the dataset, changing deliverables), or final review.
- The user works in ML/data science and is comfortable with technical detail — no need to over-explain basics.
- **After finishing any task or phase, log it in `PROGRESS.md`** (create it if missing). Each entry: date, phase/task, what was done, key results or numbers, files changed, and any open issues or next steps. Append new entries; never rewrite old ones.

## Current status

- Project idea, dataset, architecture, and plan finalized.
- Next step: Phase 0 (repo setup, dataset download and inspection) → Phase 1 (feature extraction).
