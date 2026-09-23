# Project Plan: Predictive Driver Drowsiness System with Physical Alert Actuator

ICPS (Intro to Cyber-Physical Systems) course project. Theme: predictive technologies using ML/AI, with hardware integration.

---

## 1. Key Dataset Constraint (shapes the whole design)

Dataset: [UTA-RLDD Videos Cropped By Faces](https://www.kaggle.com/datasets/mathiasviborg/uta-rldd-videos-cropped-by-faces)

- Face-cropped version of the UTA Real-Life Drowsiness Dataset.
- 60 subjects, **two classes only: non-drowsy and drowsy** (the original "low vigilant" class is not included).
- Subject 42 has **no drowsy video**.
- Each video is a single ~10-minute clip in **one state**. Videos do **not** show a transition from alert to drowsy.
- "Preprocessed" = face already cropped. Landmark and feature extraction still has to be done.

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

### Phase 0 — Setup (Week 1)

- Create repo structure; `data/` in `.gitignore`.
- Download dataset via Kaggle API into `data/raw/`.
- Inspect: folder naming, label-to-file mapping, video FPS and resolution (varies — recorded on personal phones/webcams).
- Environment: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `pyserial`, `streamlit`. Pin versions in `requirements.txt`.
- Order hardware (should arrive by Week 3).

**Output:** repo skeleton, dataset downloaded and inspected, reproducible environment.

### Phase 1 — Feature Extraction (Weeks 2–3)

1. **Resample by timestamp** to a fixed rate (e.g. 10 fps) — source FPS varies.
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

### Phase 3 — Predictive Layer → Core of Deliverable 2 (Weeks 5–8)

**Forecaster (feature trajectory):**
- Input: last 60 s of features. Output: next 30–60 s of key features (EAR, PERCLOS, blink rate).
- Model: seq2seq GRU/LSTM; small Transformer if time allows.
- Trained on all videos (including subject 42); no labels needed.
- Must beat baselines: **persistence** (future = present) and **linear trend extrapolation**. If not, report honestly.
- Metrics: MAE/RMSE at 10 s, 30 s, 60 s horizons.

**Risk fusion:**
- `p_now` = classifier on current window; `p_future` = classifier on forecast window.
- Levels:
  - 0 = safe (green)
  - 1 = `p_future` rising (yellow)
  - 2 = `p_future` high (vibration)
  - 3 = `p_now` high (buzzer)
- Hysteresis: require several consecutive readings before escalating/de-escalating.

**Lead-time evaluation (key predictive metric):**
1. **Stitched sequences:** per test subject, join end of non-drowsy video to start of drowsy video; measure how early the system escalates before the join. State clearly that the boundary is artificial.
2. **Self-recorded sessions (strongly recommended):** record team members in long sessions from rested to tired. Log **Karolinska Sleepiness Scale (KSS, 1–9)** self-ratings every 5 minutes. Provides real transitions and a small real-world validation set.

Metrics: lead time (seconds of warning before state change), false alarms per hour, missed detections.

### Phase 4 — Real-Time Integration (Weeks 8–9)

- Reuse the exact training feature code (no reimplementation — avoid train/serve mismatch).
- Startup: 30 s calibration → continuous loop.
- MediaPipe every frame; inference every 1–2 s; serial writes at 2 Hz.
- Measure end-to-end latency (frame → actuator) and FPS.
- Log features, `p_now`, `p_future`, risk level, button presses to CSV.
- Dashboard (Streamlit or OpenCV overlay): live EAR/PERCLOS, current vs forecast risk, alert level.

### Phase 5 — Evaluation, Report, Demo (Weeks 10–11)

- Results: baseline vs GRU classifier; forecaster vs persistence/linear; lead time and false alarms (stitched + self-recorded); system latency.
- Ablations (if time): no forecaster (reactive only), no per-subject normalization, different horizons.
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
| Forecaster doesn't beat persistence | Report honestly; fall back to trend extrapolation of PERCLOS as predictive layer |
| Poor landmarks with glasses / bad lighting | Log detection rate; use even front lighting for demo |
| Timeline slips | D1 is self-contained and demoable by Week 5; Phase 3 is additive |

Week numbers are indicative — adjust to semester length. Phase order matters more than exact timing.
