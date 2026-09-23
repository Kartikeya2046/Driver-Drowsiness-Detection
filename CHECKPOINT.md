# Session Checkpoint — resume here

**Last session:** 2026-09-23 (paused by user). **Last commit:** `0528124` (Phase 2b). Working tree clean.

Read order when resuming: **`context.md`** (rules, design, envs) → **this file** (where we stopped, why, what's next) → `plan.md` (full phase plan) → `PROGRESS.md` (detailed log with every number).

---

## 1. Where we are

| Phase | State |
|---|---|
| 0 Setup | DONE (hardware not ordered) |
| 1 Feature extraction | DONE: 119 per-frame parquet files in `features/`, 0.1% mean face-drop rate |
| 2 Classifiers | DONE, but numbers superseded by 2b |
| 2 Hardware loop | **DEFERRED by user**: decoupled from software (serial `R:<level>\n` only), pick up any time |
| 2b Deployment-faithful re-eval | **DONE**: GRU 78.8% acc / 0.858 AUC at 60 s calibration (3 seeds) |
| 3 Predictive layer | **NEXT** |
| 4 Real-time, 5 Report | not started |

**Headline numbers to quote (deployable protocol, `results/phase2b_summary.csv`):** GRU 78.8% / 0.858 AUC, video-level 85.9%; stacked ensemble ceiling 81.3% / 0.880. Do **not** quote Phase 2's 86–91%: that was full-video normalization, not deployable.

**Deploy model:** `models/classifier_gru.pt` = `{state_dict, channel_scale, calib_s=60}`.
Inference: `calibrate()` on the first 60 s → `apply_calibration()` → 45 s window of (ear, mar, pitch, yaw, roll) → `/ channel_scale` → `DrowsinessGRU` → sigmoid.

---

## 2. Decisions made this session, and why

| Decision | Why | Who |
|---|---|---|
| Use `len60` dataset variant only | Other `lenN` folders are re-cuts of the same footage, not extra data | User confirmed |
| All torch/xgboost training on GPU (`btp_lstm_gpu` env) | User instruction: GPU whenever there's a choice | User |
| Hardware loop deferred | User doesn't want to do it now; fully decoupled | User |
| **GRU = live model** | Best single model before and after 2b; single-model latency | User ("keep GRU top priority") |
| Ensemble reported as ceiling, **not deployed** | 4× inference cost for +1–2.5 pp | User agreed |
| Per-subject model routing **rejected** | Picking the best model per subject needs that subject's test labels (leakage) and can't be done for an unseen driver | User's idea; declined with explanation, user moved to stacking instead |
| Calibration = median shift of a calibration window, not full-video z-score | Full-video stats aren't available live and inflated accuracy by ~8–13 pp | Found in plan reassessment; user approved 2b |
| **Live calibration = 60 s** | Best GRU AUC; +0.8 pp over 30 s; 120 s adds only +0.1 pp for double the wait | Claude's call, flagged: one flag to change (`--final N`) |
| Phase 3 redesign | Single-state videos: a forecaster never sees onset, and a hard splice has no precursor. So compare reactive vs trend vs forecaster, evaluate on **ramped** synthetic transitions, and use the hard splice as negative control | Plan reassessment; user approved |

---

## 3. Next steps (Phase 3, in order)

1. **Forecaster dataset:** per-5 s aggregate features (reuse `window_features()` with 5 s windows / 5 s stride on calibrated per-frame data). Include subject 42 for forecaster training.
2. **Forecaster:** small seq2seq GRU, 12 steps (60 s) in → 6–12 steps (30–60 s) out. Baselines: persistence and linear trend. Report MAE/RMSE at 10 / 30 / 60 s, subject-grouped CV. Train on real within-video data only, **never on synthetic ramps** (circular).
3. **Ramped transition builder:** per test subject, interleave 5–10 s chunks of their alert and drowsy per-frame features, with the drowsy fraction rising 0 → 100%. Vary duration (2–10 min) and shape (linear/sigmoid). Onset = drowsy fraction first reaches 50%.
4. **Three predictive signals through one risk fusion:** (a) reactive `p_now`, (b) trend / time-to-threshold (Holt or linear on the `p_now` + PERCLOS trajectory), (c) forecaster → GRU on forecast → `p_future`. Calibrate probabilities (Platt/isotonic) and tune thresholds on **training subjects only**.
5. **Lead-time eval:** lead time, false alarms/hour, misses, and a lead-time vs false-alarm curve per signal. Run on ramps (primary) and the hard splice (negative control, expect ≈ 0 lead time).

**User-side tasks, in parallel:**
- **Start KSS self-recordings now**: long rested-to-tired sessions, raw webcam video + timestamps + KSS (1–9) every 5 min. They take calendar time. A small recording script (webcam → mp4 + timestamp CSV + KSS prompt) would help and could be the first thing built next session.
- Order hardware when convenient.

**Open decisions for the user:**
- Is 60 s calibration OK for the demo, or should it be 30 s? GRU loses only 0.8 pp at 30 s.
- Calibration-length curve figure → Phase 5 (data already in `results/phase2b_summary.csv`).

---

## 4. How to resume (commands)

```bash
# CPU env (.venv, Python 3.13): MediaPipe / feature extraction / dataset building
./.venv/Scripts/python.exe -m src.features.build_dataset            # rebuild features/dataset_cal{30,60,120}.npz (~1 min)

# GPU env (btp_lstm_gpu, Python 3.10, CUDA torch): all training
PY="D:/Anaconda3/envs/btp_lstm_gpu/python.exe"
$PY -m src.models.classifier_baseline                                # XGBoost, all calibration lengths (~seconds)
$PY -m src.models.classifier_gru                                     # CV, 3 calib lengths x 3 seeds (~7 min)
$PY -m src.models.classifier_gru --final 60                          # retrain deploy model
PYTHONIOENCODING=utf-8 $PY -m src.models.stacking_ensemble           # ensembles + results/phase2b_summary.csv
```

**Local-only (gitignored) artifacts that exist on this machine:** `features/dataset_cal*.npz`, `results/oof_*.npy`, `models/face_landmarker.task`, `data/raw/`. On a fresh clone, regenerate with `scripts/download_data.py` (needs `~/.kaggle/kaggle.json`), `scripts/download_model.py`, `src.features.extract_features` (~2 h CPU), `src.features.build_dataset`, then the classifier scripts.

---

## 5. Gotchas learned (save time next session)

- **MediaPipe VIDEO mode** needs strictly increasing timestamps per `FaceLandmarker` instance. `FaceDetector` owns the counter; don't pass timestamps in.
- **Windows Task Manager shows ~0% GPU during CUDA training.** Its default graph is the 3D engine. Use `nvidia-smi`.
- **Background job logs look empty.** Python stdout to a file is block-buffered, and `| grep` buffers too. Use `print(..., flush=True)` and avoid piping through grep (or use `grep --line-buffered`). The `±` sign makes grep treat output as binary on Windows; set `PYTHONIOENCODING=utf-8`.
- **GPU-resident training** (whole dataset on the GPU, manual batching) is several times faster than DataLoader here: ~13 s/fold GRU, ~23 s/fold CNN.
- **Median-shifted inputs are in raw units** (EAR ~0.05, pose ~5–8°). NNs need the per-channel `channel_scale`, which is stored in the checkpoint.
- **Hard subjects:** 02 is near chance for every model; 09/15/60 show no EAR drop when "drowsy". This is label/physiology variability, not a bug. Report it as a limitation.
- **User asks for progress on long jobs frequently.** Give short, concrete status (files done / N, current model), and make logs readable from the start.
