# Project Plan: Predictive Driver Drowsiness System with Physical Alert Actuator

ICPS (Intro to Cyber-Physical Systems) course project. Theme: predictive technologies using ML/AI, with hardware integration.

---

## 0. Status (reassessed 2026-09-23 — details in `PROGRESS.md`)

| Phase | Status |
|---|---|
| 0 — Setup | DONE (hardware not yet ordered) |
| 1 — Feature extraction | DONE (60 subjects, 0.1% mean face-drop rate) |
| 2 — Classifiers | DONE; the original 86–91% numbers were optimistic (full-video normalization) and are superseded by Phase 2b |
| 2 — Hardware loop | DEFERRED (user decision; fully decoupled, serial protocol only) |
| 2b — Deployment-faithful re-evaluation | DONE — GRU 78.8% acc / 0.858 AUC (60 s calibration, 3 seeds); stacked ensemble ceiling 81.3% / 0.880. Deploy model: `models/classifier_gru.pt` (60 s calibration) |
| 3 — Predictive layer | **NEXT** — revised design below (forecaster target, ramped transitions, trend comparator) |
| 4, 5 | Not started; small additions below |

---

## 1. Key Dataset Constraint (shapes the whole design)

Dataset: [UTA-RLDD Videos Cropped By Faces](https://www.kaggle.com/datasets/mathiasviborg/uta-rldd-videos-cropped-by-faces)

- Face-cropped version of the UTA Real-Life Drowsiness Dataset.
- 60 subjects, **two classes only: non-drowsy and drowsy** (the original "low vigilant" class is not included).
- Subject 42 has **no drowsy video**.
- Each subject/label is a single continuous recording in **one state** (≈4–18 min, typically ~9 min), pre-cut into 60 s chunks (`len60` variant used). Videos do **not** show a transition from alert to drowsy.
- "Preprocessed" = face already cropped to 224×224 and resampled to a fixed 10 fps. Landmark and feature extraction still had to be done (Phase 1).

**Consequence:** "predict onset" labels cannot be derived by looking ahead inside a video. The design uses two models:

- **Classifier**: learns what drowsy features look like (uses binary labels).
- **Forecaster**: learns how feature streams evolve over time (self-supervised, no labels needed).
- Running the classifier on the forecasted future window makes the system predictive.

---

## 2. System Architecture

```
                         ┌──────────── OFFLINE (training) ────────────┐
 UTA-RLDD cropped videos → MediaPipe → per-frame features → windows → train Forecaster + Classifier
                         └────────────────────────────────────────────┘

                         ┌──────────── ONLINE (live CPS loop) ────────────────────────────────────┐
 Webcam → MediaPipe Face Mesh → Feature engine → Rolling buffer (last 30–60 s)
                                                        │
                                   ┌────────────────────┴────────────────────┐
                                   ▼                                          ▼
                          Classifier: drowsy NOW?            Forecaster: predict features for next 30–60 s
                                                                               │
                                                                               ▼
                                                               Classifier on forecast → drowsy SOON?
                                   └────────────────────┬────────────────────┘
                                                        ▼
                                       Risk fusion → risk level 0/1/2/3
                                                        ▼
                                   pySerial → Arduino → LED / vibration motor / buzzer
                                                        ▲
                                   Acknowledge button ──┘ (driver feedback, logged)
```

---

## 3. Repository Structure

```
drowsiness-cps/
├── data/                 # raw videos (gitignored)
├── features/             # extracted .parquet/.npy per video
├── src/
│   ├── features/         # landmarks → EAR/MAR/pose, blink detection, PERCLOS
│   ├── models/           # classifier, forecaster, risk fusion
│   ├── realtime/         # live loop, serial client, dashboard
│   └── eval/             # metrics, lead-time evaluation, plots
├── hardware/             # Arduino .ino sketch + wiring diagram
├── notebooks/            # exploration only
├── scripts/
│   └── download_data.py  # Kaggle API download
├── requirements.txt
└── README.md
```

---

## 4. Phases

### Phase 0 — Setup (Week 1) — DONE (hardware order still pending)

- Create repo structure; `data/` in `.gitignore`.
- Download dataset via Kaggle API into `data/raw/`.
- Inspect: folder naming, label-to-file mapping, video FPS and resolution (turned out to be uniform 10 fps / 224×224 in this Kaggle version).
- Environment: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `pyserial`, `streamlit`. Pin versions in `requirements.txt`.
- Order hardware (should arrive by Week 3).

**Output:** repo skeleton, dataset downloaded and inspected, reproducible environment.

### Phase 1 — Feature Extraction (Weeks 2–3) — DONE

Implemented: EAR, MAR, pitch/yaw/roll, PERCLOS, blink rate/duration, gap interpolation, drop-rate report. Not implemented: yawn count, head-nod count, blink amplitude (proxies used: `mar_max`/`mar_std`, `pitch_std`) — optional later; a targeted head-nod feature is the most promising for subjects whose drowsiness doesn't show in EAR. Step 5's normalization baseline is revised in Phase 2b.

1. **Resample by timestamp** to a fixed rate (e.g. 10 fps) — training data is already 10 fps; still required for live webcam and self-recorded data.
2. **MediaPipe Face Mesh** per frame → landmarks.
3. **Per-frame features:** EAR (both eyes averaged), MAR, head pitch/yaw/roll.
4. **Derived temporal features:**
   - Blink rate, mean blink duration, blink amplitude
   - **PERCLOS** — fraction of time eyes ≥80% closed over a window
   - Yawn count/duration, head-nod count
5. **Per-subject normalization** using each subject's non-drowsy video statistics. Live system mirrors this with a 30 s calibration at startup.
6. **Missing faces:** interpolate gaps < 0.5 s; mark longer gaps as invalid windows; log drop rate per video.
7. Save one feature file per video to `features/`.

**Split rule:** split **by subject**, never by window. Use 5-fold subject-grouped cross-validation. Exclude subject 42 from classifier evaluation (may be used for forecaster training).

**Output:** `extract_features.py`, feature files, data-quality report (detection rate, frames per video).

### Phase 2 — Classifier + Hardware Loop → Deliverable 1 (Weeks 3–5)

**Classifier (drowsy now):**
- Window: 30–60 s of features, stride of a few seconds.
- Baseline: window summary statistics (mean/std of EAR, PERCLOS, blink rate, etc.) → XGBoost / Random Forest.
- Sequence model: small GRU/LSTM or 1D-CNN on raw windows.
- Metrics: accuracy, F1, ROC-AUC, per-subject accuracy, video-level accuracy (aggregated windows).

**Hardware (built and tested independently of the model):**
- Arduino Uno (or ESP32)
- RGB LED (green / yellow / red)
- Coin vibration motor via NPN transistor (2N2222) + flyback diode (1N4007) — never drive directly from a pin
- Piezo buzzer
- Push button — driver "I'm awake" acknowledgement
- Serial protocol: Python sends `R:<level>\n` at ~2 Hz
- **Failsafe:** no message for 2 s → Arduino shows fault blink pattern
- Test with hardcoded levels before connecting any model

**D1 demo:** webcam → live features → classifier → Arduino responds to current drowsiness state.

**Status:** classifiers DONE (GRU best single model; stacked ensemble best overall, kept as a reportable ceiling, not deployed). Hardware DEFERRED. Video-level accuracy not yet computed → Phase 2b.

### Phase 2b — Deployment-Faithful Re-evaluation (NEW — do before Phase 3)

**Why:** Phase 2 normalized each subject against their *entire* non-drowsy video (~9 min of known-alert data). The live system only gets a short calibration, and the non-drowsy test windows were normalized with statistics that included themselves. On identical eval windows (first 120 s of each alert video held out), XGBoost:

| Normalization | Deployable | Acc | AUC |
|---|---|---|---|
| Full-video z-score (Phase 2) | no | 0.848 | 0.914 |
| 30 s z-score (original live plan) | yes | 0.738 | 0.807 |
| 120 s z-score | yes | 0.775 | 0.851 |
| 30 s median shift | yes | 0.758 | 0.836 |
| 120 s median shift | yes | **0.781** | **0.867** |
| No normalization | yes | 0.723 | 0.790 |

Every forecaster/risk/lead-time result in Phase 3 sits on top of the classifier, so fix this first.

1. **Normalization = subtract the per-subject median of a calibration window** (location only — std from a few seconds of footage is dominated by how many blinks happened to occur). Calibration = first N s of the non-drowsy video; those frames are excluded from all evaluation windows. Blink threshold comes from the same window.
2. **Calibration length N is a parameter** — evaluate 30 / 60 / 120 s and plot accuracy vs N (a useful report figure). Current evidence favors 120 s (+2.3 pp over 30 s); the live/demo default is picked from this curve.
3. **Re-run XGBoost, CNN, GRU, LSTM and the ensemble** under this protocol. Report both numbers: full-video = optimistic upper bound, calibration = deployable. GRU stays the live model unless the re-run changes the ranking.
4. **Seed everything; report NN results as mean ± std over 3 seeds** — observed run-to-run variance (~1 pp) is as large as some of the model gaps.
5. **Video-level accuracy** (aggregate windows per video) — planned in Phase 2, not yet computed.
6. The live loop must call the **same calibration function** (one code path).

**Results (DONE 2026-09-23; full table in `results/phase2b_summary.csv`):** mean ± std over 3 seeds (XGBoost is deterministic, so no std), 11,214 windows / 59 subjects, identical eval windows for every calibration length.

| Model | 30 s acc / AUC | 60 s acc / AUC | 120 s acc / AUC |
|---|---|---|---|
| XGBoost | 0.758 / 0.836 | 0.753 / 0.841 | 0.781 / 0.867 |
| CNN | 0.751 / 0.823 | 0.745 / 0.811 | 0.734 / 0.800 |
| **GRU** | 0.780 / 0.848 | **0.788 / 0.858** | 0.789 / 0.850 |
| LSTM | 0.723 / 0.786 | 0.736 / 0.800 | 0.758 / 0.817 |
| Avg ensemble | 0.800 / 0.881 | 0.800 / 0.881 | 0.801 / 0.881 |
| Stacked ensemble | 0.796 / 0.873 | 0.799 / 0.873 | **0.813 / 0.880** |

- GRU stays the best single model and the live model. It's nearly insensitive to calibration length (30 → 120 s = +0.9 pp, about one seed std), so **60 s is the live default** (best GRU AUC, half the wait of 120 s).
- CNN and LSTM drop below XGBoost under the deployable protocol. The Phase 2 CNN ≈ GRU result depended on full-video normalization.
- Ensembles are still the ceiling (+1–2.5 pp acc, +2–3 pp AUC over GRU), reported only, not deployed.
- Video-level accuracy (118 videos): GRU 0.859 at 60 s; best is the stacked ensemble at 60 s (0.895) and XGBoost at 120 s (0.890).

### Phase 3 — Predictive Layer → Core of Deliverable 2 (Weeks 5–8)

**Why the original design needs adjusting (reassessment 2026-09-23):**
- Every training video is single-state, so a forecaster trained on them only ever sees *within-state* fluctuation, never onset. It will tend toward persistence, and `p_future` ≈ `p_now`. The forecaster stays (Deliverable 2 requires feature-trajectory forecasting), but its value has to be **measured against simpler predictors**, not assumed.
- A hard splice (end of alert video → start of drowsy video) has no precursor signal, so no method can show genuine lead time on it; any "early" alarm there is a false alarm. It can't be the primary lead-time benchmark.

**Forecaster (feature trajectory):**
- Forecast **coarse rolling aggregates**, not raw 10 fps frames: per-5 s window features (PERCLOS, mean EAR, blink rate, mean blink duration). Raw EAR 30–60 s ahead is blink noise and inherently unforecastable.
- Input: last 60 s (12 steps). Output: next 30–60 s (6–12 steps). Small seq2seq GRU. (Transformer dropped — no benefit expected at this data scale.)
- Trained on real within-video data from all videos (including subject 42); no labels needed. **Do not train on the synthetic ramps below** — they're our own construction, so learning them and then evaluating on them is circular.
- Must beat **persistence** and **linear trend** baselines on MAE/RMSE at 10 s, 30 s, 60 s. If not, report honestly.

**Three predictive signals, compared head-to-head through the same risk fusion:**
1. **Reactive only** — `p_now`, no prediction. The baseline every lead-time claim is measured against.
2. **Trend / time-to-threshold** — Holt (or linear) fit on the recent `p_now` and PERCLOS trajectory, extrapolated to when it crosses the alert threshold. Promoted from "fallback" (risk table) to a first-class comparator: cheap, interpretable, and it directly outputs "critical in ~X s".
3. **Forecaster → classifier on forecast window → `p_future`** — the original design.

**Risk fusion:**
- `p_now` = classifier on current window; `p_future` = from signal 2 or 3.
- Levels:
  - 0 = safe (green)
  - 1 = `p_future` rising (yellow)
  - 2 = `p_future` high (vibration)
  - 3 = `p_now` high (buzzer)
- Hysteresis: require several consecutive readings before escalating/de-escalating.
- **Calibrate probabilities** (Platt/isotonic on training folds) before thresholding, and **tune thresholds on training subjects only** against a false-alarm-per-hour budget — never on test subjects.

**Lead-time evaluation (key predictive metric):**
1. **Ramped synthetic transitions (primary synthetic benchmark):** per test subject, interleave 5–10 s chunks of their own alert and drowsy footage, with the drowsy fraction rising 0 → 100% over a ramp. This mirrors real onset, where lapses (long blinks, microsleeps) become more frequent over minutes rather than switching instantly. Vary the ramp duration (2–10 min) and shape (linear/sigmoid) so no method is tuned to one profile. Onset = when the drowsy fraction first reaches 50%; lead time = onset − first escalation. Built at the feature level from the per-frame parquet files (no video re-encoding). Caveat to state in the report: the ramps are designed by us and are gradual by construction, which favors trend methods, so they're reported alongside KSS sessions.
2. **Hard splice (original stitched sequences) — negative control only:** expected lead time ≈ 0; anything earlier counts as a false alarm.
3. **Self-recorded KSS sessions — the only source of real transitions:** record team members in long sessions from rested to tired, logging **Karolinska Sleepiness Scale (KSS, 1–9)** every 5 minutes. **Start now**: record raw webcam video + timestamps + KSS log to disk and process offline later through the same pipeline (needs Phase 1 step 1 resampling). This takes calendar time and genuinely tired sessions, and doesn't need Phase 4.

Metrics: lead time (seconds of warning before onset), false alarms per hour (on pure-alert segments), missed detections, plus a lead-time vs false-alarm trade-off curve for each of the three signals.

### Phase 4 — Real-Time Integration (Weeks 8–9)

- Reuse the exact training feature code (no reimplementation — avoid train/serve mismatch).
- Startup: calibration (length chosen in Phase 2b, same function as training) → continuous loop.
- **Domain-shift check:** training frames are 224×224 face crops; webcam frames are full-frame. Crop the face bounding box to 224×224 before landmarking (mirrors training), and compare live EAR/MAR distributions against training ones.
- MediaPipe every frame; inference every 1–2 s; serial writes at 2 Hz.
- Measure end-to-end latency (frame → actuator) and FPS.
- Log features, `p_now`, `p_future`, risk level, button presses to CSV.
- Dashboard (Streamlit or OpenCV overlay): live EAR/PERCLOS, current vs forecast risk, alert level.

### Phase 5 — Evaluation, Report, Demo (Weeks 10–11)

- Results: all classifiers (XGBoost / CNN / GRU / LSTM / ensemble) with optimistic vs deployable numbers side by side; accuracy vs calibration length; forecaster vs persistence/linear; lead time and false alarms for reactive vs trend vs forecaster (ramped + hard splice + self-recorded); system latency.
- Ablations: reactive only is now a built-in comparator (Phase 3). No per-subject normalization is already measured (Phase 2b table). Remaining if time allows: different horizons.
- Per-subject variability: some subjects show no EAR change when drowsy (see `PROGRESS.md`). Report it as a limitation, along with subject 02 being near chance for every model.
- **Demo script:** calibrate → alert (green) → slow blinks and nodding (yellow → vibration) → eyes closed (buzzer) → press button → back to green.
- Limitations: binary labels, self-reported source labels, stitched transitions, lighting sensitivity.

---

## 5. Deliverables

- **Deliverable 1:** Real-time facial feature pipeline (EAR, MAR, PERCLOS, head pose) with a drowsiness classifier trained on UTA-RLDD, driving a graduated Arduino alert system (LED, vibration, buzzer) over serial.
- **Deliverable 2:** Predictive layer that forecasts facial-feature trajectories to estimate drowsiness risk 30–60 s ahead, integrated into a closed-loop CPS with failsafe and driver acknowledgement, evaluated on lead time, false-alarm rate, and latency using stitched and self-recorded sessions.

---

## 6. Hardware List (approx. ₹800–1200)

- Arduino Uno or ESP32
- Common-cathode RGB LED + resistors
- Coin vibration motor
- 2N2222 transistor, 1N4007 diode, 1 kΩ resistor
- Active piezo buzzer
- Push button
- Breadboard + jumper wires
- USB webcam (or laptop camera)

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Forecaster doesn't beat persistence | Report honestly; the trend signal is already a first-class comparator, so the predictive layer doesn't depend on the forecaster winning |
| Predictive signals show no lead time over reactive | Compare all three honestly on ramps + KSS; a clear negative result is still a valid finding |
| Inflated classifier scores (full-video normalization — found 2026-09-23) | Phase 2b: calibration-window normalization, report deployable numbers |
| No KSS sessions recorded (time) | Start recording now; ramp results are still reportable with their caveat |
| Webcam domain shift (full frame vs 224×224 crops) | Face-crop before landmarking; compare feature distributions (Phase 4) |
| Poor landmarks with glasses / bad lighting | Log detection rate; use even front lighting for demo |
| Timeline slips | D1 is self-contained and demoable by Week 5; Phase 3 is additive |

Week numbers are indicative — adjust to semester length. Phase order matters more than exact timing.
