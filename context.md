# Agent Context: Predictive Driver Drowsiness CPS

Read this before doing any work in this repo. Full phase-by-phase plan is in `plan.md`. **Where the last session stopped, the decisions behind it, and the exact next steps: `CHECKPOINT.md`. Read it second, and update it at the end of every session.**

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

1. **Classifier** — drowsy now? 45 s windows / 5 s stride, binary labels. Built and compared: XGBoost (window stats), 1D-CNN, GRU, LSTM (raw sequences), average + stacked ensembles. **GRU is the live/deployed model** (single-model latency); the ensemble is reported as the accuracy ceiling only, not deployed.
2. **Forecaster** — predicts the next 30–60 s of **coarse per-5 s aggregates** (PERCLOS, mean EAR, blink rate, blink duration) from the last 60 s, not raw 10 fps frames. Self-supervised, trained on real within-video data only (never on synthetic ramps — circular). Must be compared against persistence and linear-trend baselines.
3. **Three predictive signals, compared head-to-head** through the same risk fusion: (a) reactive `p_now` only, (b) trend / time-to-threshold on the `p_now` + PERCLOS trajectory, (c) forecaster → classifier on forecast → `p_future`. The predictive claim must be *measured* against (a), not assumed.
4. **Risk fusion** — `p_now` and `p_future` → level 0–3 with hysteresis. Probabilities calibrated (Platt/isotonic) and thresholds tuned on training subjects only.
   - 0 safe (green), 1 `p_future` rising (yellow), 2 `p_future` high (vibration), 3 `p_now` high (buzzer).
5. **Lead-time evaluation** — primary synthetic benchmark: **ramped transitions** (interleaved 5–10 s chunks of a test subject's alert/drowsy footage with a rising drowsy fraction, varied duration and shape). Hard splice (alert video → drowsy video) is a **negative control** only (no precursor exists, so "early" alarms there are false alarms). Real validation: self-recorded sessions with KSS (Karolinska Sleepiness Scale) ratings every 5 min.

## Non-negotiable rules

- **Split by subject, never by window/frame.** 5-fold subject-grouped CV. Leakage makes results meaningless.
- **Resample videos by timestamp** to a fixed rate (~10 fps). Don't use "every Nth frame." (Training data is already 10 fps; this applies to live webcam and self-recorded data.)
- **Per-subject calibration, deployment-faithful** (Phase 2b): subtract the per-channel **median of a calibration window** — training: first N s of the subject's alert video; live: first N s of the session. The first 120 s of every alert video are reserved for calibration and **never appear in evaluation windows**. N ∈ {30, 60, 120} s is an evaluated parameter. **Never normalize against a whole video** — XGBoost went from 84.8% to 75.8–78.1% once made deployable (see `PROGRESS.md`). Code: `src/features/windows.py:calibrate()` / `apply_calibration()`.
- **One feature-extraction code path** shared by training and real-time inference. No reimplementation — the live loop calls the same `calibrate()` / `apply_calibration()` / `window_features()`.
- **All classifiers share one fixed split** (`src/models/training.py:cv_folds`) and one dataset file per calibration length (`features/dataset_cal{N}.npz`, summary stats + sequences aligned by construction). Neural models are seeded and reported as mean ± std over 3 seeds.
- **Missing faces:** interpolate gaps < 0.5 s; longer gaps → invalid windows. Never silently zero-fill. Log drop rates.
- **Never commit `data/`** (gitignored). Commit code, small feature files, and small model weights only.
- Exclude subject 42 from classifier evaluation; it may be used for forecaster training.
- Report negative results honestly (e.g. if the forecaster doesn't beat persistence).

## Tech stack

- Python, `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch` (PyTorch only — don't mix in TensorFlow), `pyserial`, `streamlit`.
- Arduino C++ (`hardware/*.ino`).
- Pinned `requirements.txt`.

## Two environments — which to use when

- **`.venv/` (project root, Python 3.13, CPU):** feature extraction (`src/features/*`), dataset/inspection scripts, anything using MediaPipe/OpenCV. MediaPipe has no GPU benefit here; this env has no CUDA torch.
- **`D:\Anaconda3\envs\btp_lstm_gpu` (Python 3.10, CUDA torch 2.14.0+cu126):** all model training/inference — classifier (GRU/LSTM/1D-CNN), forecaster, anything using `torch`. User's pre-existing env (`btp_lstm_gpu`), had TensorFlow 2.10 + numpy/pandas/sklearn/xgboost already; added CUDA-enabled PyTorch + pyarrow on top. GPU: RTX 3050 6GB, driver supports CUDA 13.1.
- **Rule: prefer GPU over CPU whenever there's a choice** (training, batch inference). CPU is fine for tasks that don't benefit from GPU (data loading, feature extraction, classical sklearn/xgboost on small tabular data — xgboost can optionally use `device="cuda"` if it becomes a bottleneck, but isn't required to).
- Invoke directly by full interpreter path, e.g. `"D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.classifier_gru`, rather than activating — keeps scripts callable from either env unambiguously.
- TensorFlow is present in `btp_lstm_gpu` from prior use but **must not be used** in this project — PyTorch only, per the rule above. Don't import it in project code.

## Hardware protocol

- Python → Arduino over serial, ASCII line `R:<level>\n`, level 0–3, ~2 Hz.
- Arduino **failsafe**: no message for 2 s → fault blink pattern.
- Button press → Arduino sends `ACK\n` back to Python; log it.
- Vibration motor must be driven through an NPN transistor (2N2222) with a flyback diode — never directly from a GPIO pin.
- Hardware is developed and tested with hardcoded levels independently of the ML models.

## Repo layout

```
data/           raw videos (gitignored)
features/       per-video extracted features (committed); dataset_cal{N}.npz (gitignored, rebuild with src.features.build_dataset)
models/         deployable model weights (small, committed); MediaPipe .task (gitignored, scripts/download_model.py)
results/        summary tables (committed); oof_*.npy out-of-fold predictions (gitignored, regenerated by the classifier scripts)
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

## Current status (2026-09-23)

- **Done:** Phase 0 (setup), Phase 1 (feature extraction, 60 subjects), Phase 2 (XGBoost/CNN/GRU/LSTM/ensembles), Phase 2b (deployment-faithful re-evaluation).
- **Headline numbers (deployable protocol, `results/phase2b_summary.csv`):** GRU 78.8% acc / 0.858 AUC at 60 s calibration (mean of 3 seeds); stacked-ensemble ceiling 81.3% / 0.880. The earlier 86–91% Phase 2 numbers are **superseded** (full-video normalization, not deployable) — don't quote them as results.
- **Deploy model:** `models/classifier_gru.pt` = {`state_dict`, `channel_scale`, `calib_s`=60}. Inference: `calibrate()` on the first 60 s → `apply_calibration()` → 45 s window of (ear, mar, pitch, yaw, roll) → divide by `channel_scale` → GRU → sigmoid.
- **Deferred:** hardware loop (user decision; decoupled, serial protocol only). Hardware not yet ordered.
- **Next:** Phase 3 (predictive layer, per the revised `plan.md`). KSS self-recordings should start now (raw webcam video + timestamps + KSS every 5 min).
