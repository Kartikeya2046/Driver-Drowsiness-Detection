# Session Checkpoint — resume here

**Last session:** 2026-09-29. **Last commit:** `47073ab` (Phase 5 report). Working tree: `hardware_list.md` untracked (prior session, never committed — ask user), `data_inspection_report.json` has a stray pre-existing whitespace diff (not from this project's work, left alone).

Read order when resuming: **`context.md`** (rules, design, envs) → **this file** → `REPORT.md` (consolidated results) → `PROGRESS.md` (detailed log, every number, every decision with rationale) → `plan.md` (full phase plan, now mostly historical).

---

## 1. Where we are

| Phase | State |
|---|---|
| 0-2b | DONE (see REPORT.md §2) |
| 3 Predictive layer | **DONE** — forecaster, ramps, risk fusion, lead-time eval (REPORT.md §3) |
| 4 Real-time | **DONE (software side)** — live loop, domain-shift check, latency, dashboard (REPORT.md §4). Hardware loop still not ordered. |
| 5 Report | Consolidated report + figures done (`REPORT.md`, `results/figures/`). Physical demo script blocked on hardware. |

**Deploy model:** `models/classifier_gru.pt`, reactive-only (no forecaster in the live loop — see REPORT.md §3 for why). Live loop: `python -m src.realtime.live_loop --source 0 --port COM3` (or a video file path instead of `0`; omit `--port` to log serial instead of sending).

**Standing rule (2026-09-29, memory `feedback_autonomous_decisions.md`): full automode.** Weigh alternatives, decide, implement, test, commit, and move to the next step without stopping to ask — including between phases. Only stop for genuine blockers (info only the user has, hardware they need to act on) or decisions that change user-facing behavior.

---

## 2. What's actually blocking further progress

1. **Hardware not ordered.** `hardware_list.md` has the exact parts list. Until it arrives: can't build/test the Arduino sketch, can't measure true actuator latency, can't run the physical demo script (calibrate → green → yellow/vibration → buzzer → button → green).
2. Nothing else is blocked. If the user wants more work before hardware arrives, options (not yet started, not requested):
   - A second webcam recording (different lighting/person) to strengthen the domain-shift n=1 finding.
   - Arduino sketch itself (`hardware/*.ino`) can be written and tested with hardcoded levels *without* waiting for the ML side — it's decoupled by design (serial protocol only). Could be started once parts arrive, or the sketch code itself could be drafted now against the documented protocol even without hardware to flash it to (untestable without the board, but the code could exist).

---

## 3. How to resume (commands)

```bash
# CPU env (.venv, Python 3.13): MediaPipe / feature extraction / dataset building / eval
./.venv/Scripts/python.exe -m src.features.build_dataset
./.venv/Scripts/python.exe -m src.features.build_forecast_dataset 60
./.venv/Scripts/python.exe -m src.eval.ramps
./.venv/Scripts/python.exe -m src.eval.domain_shift <path/to/recording.mp4>
./.venv/Scripts/python.exe -m src.realtime.live_loop --source 0 --show   # live camera, on-screen overlay, log-only serial
./.venv/Scripts/python.exe -m src.eval.make_report_figures

# GPU env (btp_lstm_gpu, Python 3.10, CUDA torch): all training
PY="D:/Anaconda3/envs/btp_lstm_gpu/python.exe"
$PY -m src.models.classifier_gru --final 60      # retrain deploy model
$PY -m src.models.forecaster                      # forecaster CV (not deployed, see REPORT.md)
PYTHONIOENCODING=utf-8 $PY -u -m src.eval.lead_time   # full lead-time eval (slow, ~10 min)
```

**Local-only (gitignored) artifacts on this machine:** `features/dataset_cal*.npz`, `features/forecast_cal*.npz`, `features/ramps_cal*.npz`, `features/webcam_demo.parquet`, `results/oof_*.npy`, `results/live_log_*.csv`, `models/face_landmarker.task`, `data/raw/`. Regenerate per the commands above / `scripts/download_data.py` + `scripts/download_model.py` + `src.features.extract_features` (~2h CPU) for a fresh clone.

---

## 4. Gotchas learned this session (add to the running list in the old checkpoint entries)

- `np.load(...)['key']` on an `NpzFile` **re-decompresses per access** — looping and indexing into the same key repeatedly (e.g. inside a generator) OOMs. Read each array out once (`{k: z[k] for k in z.files}`) before looping.
- `torch.load` under PyTorch 2.6+ defaults to `weights_only=True` and will reject a checkpoint containing a numpy array (our `channel_scale`). Pass `weights_only=False` for our own trusted checkpoints.
- A naive linear-trend-on-classifier-output signal is **not fixable by damping or smoothing alone** at this window/noise level — tried both, still unusable. Don't re-attempt without a fundamentally different estimator (e.g. Kalman/particle filter) if this comes up again.
- Comparing feature distributions across domains (e.g. webcam vs. training) must use **calibrated** features, not raw ones — raw pooling across many training subjects mixes person-to-person variation into what's supposed to be a camera-only check.
- When interpreting a real recording's model output against an "expected narrative," don't assume a simple single-hump timeline — ask the user what they actually did before concluding the model is wrong.
