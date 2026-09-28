# ICPS Drowsiness Detection — Results Report

Consolidates Phases 0–4. Every number here is reproduced in `PROGRESS.md` with the
date it was produced and the exact command; this is the summary for the final
writeup/demo, not a new source of truth. Figures are in `results/figures/`.

## 1. Pipeline

UTA-RLDD webcam recordings (60 subjects, binary alert/drowsy labels) → MediaPipe
face landmarks → EAR/MAR/head-pose per frame → per-subject calibration (median
shift from a known-alert window) → 45 s windows → classifier → risk level → serial
protocol to an Arduino alert loop (LED/vibration/buzzer).

## 2. Classifiers (Phase 2 / 2b)

**Headline: report the deployable numbers, not the optimistic ones.** Phase 2
normalized each subject against their entire non-drowsy video, including
statistics a live system can't have in advance. Phase 2b redid every model with
only a short calibration window (median shift, 30/60/120 s), which is what the
live loop actually gets. Every model lost 8–13 points of accuracy between the two.

| Model | Optimistic acc (Phase 2, full-video norm) | Deployable acc (Phase 2b, 60 s calib) | Deployable AUC |
|---|---|---|---|
| XGBoost | 86.2% | 75.3% | 0.841 |
| 1D-CNN | 87.4% | 74.5% | 0.811 |
| **GRU (deployed)** | 88.9% | **78.8%** | **0.858** |
| LSTM | 86.3% | 73.6% | 0.800 |
| Avg ensemble | 90.7% | 80.0% | 0.881 |
| Stacked ensemble | 91.2% | 79.9% | 0.873 |

GRU is the best single model on every metric, before and after the deployable
re-evaluation, and is nearly flat across calibration length (+0.9 pp from 30→120 s
— see `results/figures/calibration_curve.png`). Ensembles add +1–2.5 pp over GRU
but at 4x inference cost; reported as a ceiling, not deployed. **Deploy choice:
GRU, 60 s calibration** (best AUC, half the wait of 120 s).

**Video-level accuracy** (aggregating window probabilities per video): GRU 82.5% /
85.9% / 84.5% at 30/60/120 s; best overall is the stacked ensemble at 60 s (89.5%).

**Per-subject variability, not a bug:** subjects 02, 09, 15, 24, 60 are near chance
for every model tried. Confirmed at the raw signal level — some subjects show no
EAR drop at all when labeled drowsy (physiological/behavioral variability in how
drowsiness manifests, not an extraction or normalization defect). Report as a
dataset limitation.

## 3. Forecaster and lead-time evaluation (Phase 3)

**Forecaster barely beats persistence.** A GRU trained to predict the next 60 s of
5 s-aggregate features beats a flat "mean of the last 60 s" baseline by only ~2–5%
MAE, and is no better (sometimes worse) on the two features that matter most
(`ear_mean`, `perclos`) at 30–60 s horizons. Linear trend extrapolation is worse
than persistence at every horizon (`results/figures/forecaster_baselines.png`).
Root cause: every training video is single-state (alert *or* drowsy, never both),
so there's no real transition for a forecaster to learn from — this is a property
of the dataset, not a fixable modeling issue.

**Ramped synthetic transitions** (1,239 sequences, 59 subjects, alert→drowsy
ramps of 2/5/10 min plus a hard-splice negative control) were built to evaluate
lead time properly, since the dataset itself has no transitions.

**A naive linear "is p_now rising" trend signal is structurally unusable** —
confirmed by trying to fix it twice (Holt-style damping, then EWMA smoothing
before the slope fit) and finding it *still* can't hold false alarms under 4.5/hour
at even the strictest threshold. p_now is itself a noisy classifier output; a
12-point slope amplifies that noise past saturation regardless of damping. Not
deployed.

**Lead-time results** (`results/figures/leadtime_curve.png`,
`leadtime_vs_duration.png`), at a practical 3 false-alarms/hour budget:

| Signal | 10-min ramp hit rate | 10-min ramp lead time | 2-min ramp lead time | Hard splice (negative control) |
|---|---|---|---|---|
| Reactive only | 62% | −10 s | negative | negative |
| Forecaster (fc30) | 79% | **+45 s** | negative | negative |
| Step-GRU on aggregates (no forecast) | 86% | **+50 s** | negative | negative |
| Reactive + forecaster (fused) | 78% | +35 s | negative | negative |

**No signal gives positive lead time on fast (2 min) transitions** — that's simply
faster than any window-based method can react to. On slow (10 min) transitions,
~35–50 s of genuine lead time is achievable. **The forecaster does not clearly
beat a plain classifier on 5 s aggregates** (`step_now`) — the same "forecaster
doesn't add much" conclusion as the MAE result, now confirmed at the level that
actually matters (lead time). The hard splice shows negative lead time for every
signal, confirming nothing hallucinates advance warning where none exists.

**Deploy decision: reactive GRU only**, optionally blended with the forecaster
(`fused_rf`) for a modest gain on slow onsets. The forecaster/step-GRU combo is
not wired into the live loop — the added complexity (two more models in the
inference path) isn't justified by ~35–50 s of extra lead time that only shows up
on slow transitions. Documented as an evidence-based Phase 3→4 scope decision, not
an oversight.

## 4. Real-time system (Phase 4)

- **Webcam domain-shift check:** training frames are 224×224 face crops; a raw
  webcam frame isn't. `src/features/webcam.py` crops to the face bounding box
  before landmarking. Compared against training on **calibrated** features (the
  only fair comparison — pooling raw features across 59 different training
  subjects would mix person-to-person variation into what's supposed to be a
  camera check). Result on one 12.3-min real recording: **EAR (the dominant
  classifier signal) matches training well** (webcam/train std ratio 95%); MAR
  and head pitch/yaw show less variation in this session than in training (11%,
  64%, 57% of training std respectively) — likely camera angle and/or one
  scripted take rather than 59 people's natural range. **n=1 spot check, not a
  validated general claim.**
- **Live loop:** calibrate (60 s) → forward-fill short face-loss gaps → 45 s
  window → GRU → risk level → serial at ~2 Hz, with ACK polling and CSV logging.
  Runs against a live camera or a recorded file through the same code path.
  Verified end-to-end on the user's real recording (12.3 min, acted
  alert→drowsy→alert): 633 decisions, no crashes, and **the reactive signal
  tracked the acted-out drowsy/alert state reasonably well** once the actual
  acting timeline was confirmed (not the simple single-hump pattern first
  assumed — see `PROGRESS.md` 2026-09-29 correction entry).
- **Latency:** mean 37.7 ms/tick, p95 52.4 ms, max 68.7 ms, against a 100 ms
  (10 fps) budget — **0% of ticks over budget**. CPU sustains ~26 fps, well above
  the 10 fps processing target, so the software pipeline is not the bottleneck.
  True actuator response time is untested — the Arduino hardware hasn't arrived.
- **Live dashboard:** on-frame overlay (EAR, PERCLOS, p_now, color-coded alert
  level), `--show` flag. Chosen over a separate Streamlit app — no new
  dependency, fits directly into the loop already running.

## 5. What's not done / can't be done yet

- **Hardware loop** (Arduino LED/vibration/buzzer): parts list ready
  (`hardware_list.md`), not ordered. The serial protocol is fully implemented and
  tested in log-only mode; the physical actuator side and the demo script
  (calibrate → green → yellow/vibration → buzzer → button → green) are pending
  the hardware arriving.
- **KSS self-recordings** (long rested→tired sessions with periodic self-rated
  sleepiness): skipped by user decision in favor of one shorter demo/domain-check
  recording. Lead time is therefore validated on synthetic ramps only, not real
  gradual-onset footage — stated as a limitation, not a gap to silently paper
  over.
- **Multi-session domain validation:** the domain-shift and live-loop checks are
  both n=1 (one person, one session). A second recording (different
  lighting/person if possible) would strengthen the claim but isn't required for
  the deliverables.

## 6. Limitations (for the write-up)

- Labels are self-reported and per-video binary (alert/drowsy), not continuous or
  physiologically verified — inherited from UTA-RLDD, not fixable in this project.
- Training clips are stitched from pre-chunked segments of one longer recording
  per subject/label; no natural alert→drowsy transitions exist in the source data
  (hence the synthetic ramp construction for lead-time evaluation).
- Some subjects show no measurable EAR change between labeled states (Section 2).
- MediaPipe landmark quality is sensitive to lighting and glasses; not
  systematically stress-tested here beyond the one domain-shift recording.
- Deployable accuracy (78.8%) is meaningfully lower than the optimistic Phase 2
  number (88.9%) — report the deployable number as the headline result.
