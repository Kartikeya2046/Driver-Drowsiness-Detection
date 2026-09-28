# Progress Log

## 2026-09-23 — Phase 0: Repo setup, environment, dataset download & inspection

**Done:**
- Repo skeleton created per plan.md layout (`data/`, `features/`, `src/{features,models,realtime,eval}`, `hardware/`, `scripts/`, `notebooks/`). Git initialized.
- `.gitignore`: excludes `data/`, venv, caches, `kaggle.json`.
- `requirements.txt` pinned. Original plan versions (numpy 1.26.4, mediapipe 0.10.14) don't have Python 3.13 wheels on this machine — resolved to latest mutually-compatible set instead: opencv-python 5.0.0.93, mediapipe 1.0.1, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1, torch 2.14.0 (CPU), pyserial 3.5, streamlit 1.64.0, kaggle 2.2.4, pyarrow 25.0.1, matplotlib 3.11.2. All installed and import-verified in `.venv`.
- `scripts/download_data.py` — Kaggle API download + unzip into `data/raw/`. Ran successfully with user-provided `kaggle.json`.
- `scripts/inspect_data.py` — inspects folder layout, label mapping, video fps/resolution/duration, chunk counts per subject. Output saved to `data_inspection_report.json`.

**Key findings (corrected `context.md` dataset facts section accordingly):**
- Dataset is **already resampled to a fixed 10 fps** and cropped to 224×224 — not variable FPS/resolution as originally assumed (that description applies to the raw UTA-RLDD, not this Kaggle repackaging). **Phase 1's "resample by timestamp" step is unnecessary for training data**, still needed for live webcam inference.
- Dataset is **already pre-chunked** into fixed-length clips, provided in 5 redundant variants: `len5/len10/len20/len30/len60` (same content, different cut lengths). Plan should use one variant (`len60` likely best for 30-60s windows) as the source, not treat variants as extra data.
- Folder layout: `data/raw/UTA-RLDD Face Cropped Video/<lenN>/<subject 01-60>/<label 0|10>/<subject>_<label>_<group>_<chunkIdx>.mp4`.
- Confirmed: 60 subjects, binary labels (`0`=non_drowsy, `10`=drowsy), subject 42 missing the drowsy label — all match plan's stated facts.
- Chunk/source duration varies per subject (not a fixed ~10 min) — e.g. subject 01 ≈ 9.3 min, subject 04 ≈ 11.8 min.

**Files changed:** `.gitignore`, `requirements.txt`, `scripts/download_data.py`, `scripts/inspect_data.py`, `context.md` (dataset facts corrected), `PROGRESS.md` (new), directory skeleton with `.gitkeep` placeholders.

**Open issues / next steps:**
- Hardware not yet ordered (plan Phase 0 item — outside this session's scope, user-side action).

## 2026-09-23 — Phase 1: Feature extraction pipeline

**Decided:** use `len60` variant as the sole source (user confirmed) — fewest files, largest pre-cut windows, no double-counting the redundant length variants.

**Done:**
- `models/download_model.py` (actually `scripts/download_model.py`) — fetches MediaPipe FaceLandmarker `.task` model (not bundled in mediapipe 1.0.1's new Tasks API; old `mp.solutions.face_mesh` is gone in this version). Model gitignored, small refetch.
- `src/features/landmarks.py` — pure functions: EAR (6-point, both eyes averaged), MAR, head pose (pitch/yaw/roll) from MediaPipe's facial transformation matrix. Single source of truth, no duplication between training/live paths.
- `src/features/detector.py` — `FaceDetector` wrapper around MediaPipe Tasks `FaceLandmarker`, VIDEO running mode with `output_facial_transformation_matrixes=True`. Owns its own monotonic timestamp counter internally (see bug below) so any caller processing multiple clips through one instance is automatically safe.
- `src/features/extract_features.py` — per subject/label: stitches len60 chunks back into one continuous per-frame timeline (chunks are just splits of one source video), interpolates missing-face gaps <0.5s, marks longer gaps invalid (never zero-fills), saves one parquet per subject/label to `features/`.
- `src/features/windows.py` + `build_windows.py` — window-level derived features (PERCLOS, blink rate/duration via per-subject-calibrated EAR threshold, pitch/yaw variability), per-subject normalization (z-score against each subject's own non_drowsy baseline — mirrors the live system's 30s calibration), 45s windows / 5s stride, subject-grouped output with `usable_for_classifier` flag (False for subject 42).

**Bug found and fixed:** MediaPipe's VIDEO-mode `detect_for_video` requires a strictly monotonically increasing timestamp on one `FaceLandmarker` instance. Initial version threaded per-call timestamps from the caller, reset to 0 per subject/label — crashed with "Input timestamp must be monotonically increasing" as soon as a second subject/label was processed through the shared detector. Root-caused and fixed at the source: `FaceDetector` now owns an internal always-incrementing counter, so `detect()` takes no timestamp argument and no caller can violate the invariant.

**Running:** full extraction across all 119 subject/label videos (~2hr+ estimated from a 3-video smoke test — MediaPipe per-frame inference is the bottleneck, ~600 frames/video × 119). Will update this entry with final drop-rate stats once complete.

**Open issues / next steps:**
- Once extraction finishes: run `build_windows.py`, inspect `windows.parquet` and drop-rate report, then move to Phase 2 (classifier baseline + hardware loop).
- Blink threshold uses 15th-percentile EAR from each subject's non-drowsy baseline — reasonable default, not validated against ground-truth blink annotations (UTA-RLDD doesn't provide them). Worth a sanity plot once data lands.

## 2026-09-23 — GPU environment set up for model training

**Decided (user request):** use local RTX 3050 6GB for all torch-based training/inference from here on, CPU only where there's no GPU alternative (feature extraction, data loading).

**Done:**
- Found user's pre-existing `btp_lstm_gpu` conda env (`D:\Anaconda3\envs\btp_lstm_gpu`, Python 3.10) — had TensorFlow 2.10 GPU + numpy/pandas/sklearn/xgboost from prior work, no PyTorch.
- Installed CUDA-enabled PyTorch (2.14.0+cu126) and pyarrow into it. Verified with an actual GPU matmul (`torch.cuda.is_available()` True, device correctly identifies "NVIDIA GeForce RTX 3050 6GB Laptop GPU").
- Documented the two-environment split in `context.md`: `.venv` (CPU, Python 3.13) for MediaPipe/feature extraction, `btp_lstm_gpu` (CUDA torch, Python 3.10) for all model training. TensorFlow is present in `btp_lstm_gpu` from prior use but must not be imported — PyTorch only per project rules.

**Open issues / next steps:**
- Phase 2/3 model training scripts (`src/models/*`) should be run via `"D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m ...`, not the project `.venv`.
- xgboost baseline classifier could optionally use `device="cuda"` if training time on window-summary-stat features becomes a bottleneck — not required, data is small (a few thousand windows).

## 2026-09-23 — Phase 2: baseline classifier trained (XGBoost, GPU)

**Done:**
- `src/models/classifier_baseline.py` — XGBoost (`device="cuda"`) on window summary-stat features (EAR/MAR mean/std/min/max, pitch/yaw std, PERCLOS, blink rate/duration). Subject-grouped 5-fold CV via `StratifiedGroupKFold`, with an explicit assertion that no subject appears in both train and test of a fold (leakage guard). Excludes subject 42 (via `usable_for_classifier` flag from Phase 1).
- Results: **86.2% accuracy, 0.852 F1, 0.917 ROC-AUC** (mean across 5 folds, std ~0.03-0.04). Per-subject accuracy within folds ranges widely (min 0.53-0.69 for hard subjects, up to ~0.90+ mean) — some subjects are much harder to classify than others, worth a closer look before D1 demo (could be lighting/glasses/individual expressiveness).
- Final model trained on all usable windows, saved to `models/classifier_baseline.json` (gitignored alongside the MediaPipe model — small, easily regenerated by rerunning the script).

**Open issues / next steps:**
- Sequence model (GRU/LSTM/1D-CNN) on raw windows not yet built — plan's "sequence model" comparison point for Phase 2.
- Hardware loop (Arduino + serial protocol) not yet built — the other half of Deliverable 1.

## 2026-09-23 — Phase 2: 1D-CNN sequence model trained (GPU), compared to XGBoost baseline

**Refactor first:** extracted the per-subject normalization + baseline-loading loop (previously duplicated logic waiting to happen between the summary-stat window builder and a new raw-sequence builder) into `src/features/windows.py:iter_normalized_subject_labels()`. `build_windows.py` now just calls it; verified byte-identical window counts before/after (12,634, same per-subject/label breakdown) as a regression check.

**Done:**
- `src/features/build_sequences.py` — same window/stride/normalization as `build_windows.py`, but keeps raw per-frame arrays (450 frames x 5 channels: ear/mar/pitch/yaw/roll) instead of collapsing to summary stats. Output: `features/sequences.npz` (13MB, gitignored — trivially regeneratable from the already-committed `features/*.parquet`, no reason to bloat the repo with a derived cache).
- `src/models/classifier_cnn.py` — small 1D-CNN (3 conv blocks, ~last channel 64, global avg pool, FC head, ~tens of thousands of params — kept small given ~12.6k windows / 60 subjects to limit overfitting risk). Same `StratifiedGroupKFold(seed=0)` subject-grouped 5-fold CV as the XGBoost baseline (identical fold assignments, for a fair comparison), same leakage assertion. Trained on GPU (confirmed via `nvidia-smi`: ~1GB VRAM, ~28% util during training — note Windows Task Manager's default "GPU" graph tracks the 3D engine, not Compute, so it can show ~0% for real CUDA work; `nvidia-smi` is the reliable check).

**Results — CNN vs XGBoost baseline:**
| | acc | f1 | auc |
|---|---|---|---|
| XGBoost (window stats) | 0.862 | 0.852 | 0.917 |
| 1D-CNN (raw sequences) | 0.865 | 0.844 | 0.928 |

Essentially comparable — CNN slightly ahead on accuracy/AUC, XGBoost slightly ahead on F1. Not a large win from raw sequence learning over hand-crafted summary stats at this data scale, which is itself an honest finding worth stating in the report rather than picking whichever model looks best.

**Interesting:** the two models disagree on which subjects are hardest. XGBoost's worst 5 were 24, 09, 02, 48, 59 (see prior investigation — weak/inverted EAR signal). CNN's worst 5 are 02, 54, 28, 23, 24 — only 02 and 24 overlap. The CNN does noticeably *better* than XGBoost on subjects 09/60/15/48/59 (the ones with weak EAR separation), suggesting it's picking up temporal/sequential patterns the window-summary-stats can't capture — plausible since raw sequences retain information (e.g. exact timing/shape of eye closures, not just aggregate PERCLOS) that summary stats discard. Worth a line in the final report; not chasing further right now.

**Open issues / next steps:**
- Hardware loop (Arduino + serial protocol) — the remaining piece of Deliverable 1.
- `models/classifier_cnn.pt` (116K) and `models/classifier_baseline.json` (324K) both committed — small, per project rule.

## 2026-09-23 — GRU and LSTM added, full 4-model comparison

**Refactor:** extracted the CV/train/eval loop shared by CNN/GRU/LSTM into `src/models/sequence_training.py:run_cv()` — a model constructor + name is all each of `classifier_cnn.py`, `classifier_gru.py`, `classifier_lstm.py` now provide. Re-ran CNN after the refactor as a regression check; results shifted slightly (86.5%->87.4% acc) purely from unseeded model-training randomness (weight init/batch shuffling) — the CV split itself is seeded and identical, so this is expected run-to-run variance, not a refactor bug.

**Full comparison (subject-grouped 5-fold CV, mean across folds):**
| Model | Accuracy | F1 | ROC-AUC | CV std (acc) |
|---|---|---|---|---|
| XGBoost (window stats) | 0.862 | 0.852 | 0.917 | 0.036 |
| 1D-CNN (raw sequences) | 0.874 | 0.856 | 0.927 | 0.025 |
| **GRU (raw sequences)** | **0.889** | **0.883** | **0.937** | **0.019** |
| LSTM (raw sequences) | 0.863 | 0.848 | 0.917 | 0.050 |

**GRU is the best model on every metric** and the most consistent across folds (lowest std). LSTM had one unstable fold (fold 1: acc 0.790, auc 0.848, notably below its other folds) - same architecture pattern as GRU (single recurrent layer, hidden_size=64) but LSTM's extra gate/cell-state seems more sensitive to this particular split or init; not investigated further, noted as an observation.

**User idea considered and declined:** per-subject model routing (use whichever of the 4 models scores best for each specific subject, use that one for them). Explained why this doesn't work: choosing per-subject winners requires seeing each subject's held-out test accuracy, i.e. using test-set labels to pick the model - that's leakage, and the resulting "ensemble" accuracy wouldn't be a valid generalization estimate. It's also undeployable: a real new driver has no ground truth to know in advance which model suits them. Proposed a legitimate alternative instead - a standard prediction-averaging/stacking ensemble across all 4 models (no subject-specific routing, no leakage) - pending user decision on whether to build it.

**Open issues / next steps:**
- **Hardware loop deferred by user request (2026-09-23)** — user confirmed it's fully decoupled from the software (only touchpoint is the serial protocol, `R:<level>\n` at ~2Hz + `ACK\n` on button press; Arduino side is built/tested with hardcoded levels, no model needed). Parking as an open task, pick up whenever — doesn't block any other Phase 2/3 work.

## 2026-09-23 — Stacking ensemble built and evaluated

**Done:**
- Added `t_start` to `features/sequences.npz` (`build_sequences.py`) as an explicit join key — `windows.parquet` (XGBoost's summary-stat features) and `sequences.npz` (raw arrays for CNN/GRU/LSTM) are two independently-built files; rather than assume their row order matches, `src/models/stacking_ensemble.py:load_aligned()` merges them on `(subject, label_binary, t_start)` with `validate="one_to_one"` plus an explicit row-count assertion, so a silent misalignment would raise instead of silently corrupting every downstream number.
- `src/models/stacking_ensemble.py`: honest (non-leaky) stacking — every window's 4 base-model predictions come from a fold where that window's subject was held out during that model's training; the logistic-regression meta-learner is then evaluated via the same folds (meta-trained only on other folds' OOF predictions).
- Also computed a zero-training unweighted-average ensemble as a baseline comparison to the learned stacking layer.

**Results (subject-grouped 5-fold CV, all via OOF predictions):**
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| XGBoost | 0.862 | 0.854 | 0.915 |
| CNN | 0.869 | 0.851 | 0.930 |
| GRU | 0.891 | 0.884 | 0.937 |
| LSTM | 0.873 | 0.864 | 0.924 |
| Average ensemble (unweighted mean) | 0.907 | 0.900 | 0.960 |
| Stacked ensemble (logistic regression) | 0.912 | 0.908 | 0.959 |

Base-model numbers closely match the earlier standalone runs (small run-to-run variance from unseeded training, same as before) — a useful sanity check that the merge-based alignment is correct, not just assumed.

**Both ensembling approaches meaningfully beat every individual model, including GRU** (+2pp acc / +2pp AUC over GRU alone for the average ensemble). Almost all of the gain comes from the free unweighted average; the learned stacking meta-learner only adds ~0.5pp on top — an honest finding that the extra complexity buys comparatively little here.

**Notable:** the meta-learner consistently weights XGBoost highest and GRU lowest, despite GRU being the best individual model (e.g. fold 0 weights: xgboost=3.30, cnn=2.29, lstm=1.97, gru=1.42). Not a bug — a stacking meta-learner weights models by how much *unique* signal they add beyond the others, not standalone accuracy. XGBoost (tree-based, hand-crafted summary stats) is architecturally the most different from the three neural sequence models, so it contributes the most complementary information; GRU's errors likely correlate heavily with CNN/LSTM's, so it gets down-weighted.

**Decision (user-directed):** GRU remains the priority model for the live/real-time system (single-model inference, latency-sensitive). The ensemble result (91.2% acc / 0.959 AUC) is documented as the reportable "ceiling if 4x inference cost weren't a constraint" rather than adopted for deployment — no code wires the ensemble into the live loop.

Worst subjects for the stacked ensemble: 02 (0.51, near chance — hardest across every architecture tried, worth a mention as a genuinely hard case), 60, 15, 24, 23.

## 2026-09-23 — Investigated low-accuracy subjects (out-of-fold analysis)

Computed out-of-fold predictions across all 5 CV folds to get true per-subject
accuracy (not just fold minimums). Worst: subject 24 (53%), 09 (57%), 02
(60%), 48 (60%), 59 (63%), 60 (64%), 15 (65%). Saved to `per_subject_accuracy.csv`
(not committed — quick analysis artifact, regeneratable).

**Root cause, confirmed at the raw (unnormalized) per-frame level, not a
pipeline bug:**
- Subjects 09, 60, 15: raw drowsy-video mean EAR is roughly equal to, or
  even *higher* (more open) than, their own non-drowsy baseline EAR. E.g.
  subject 60: alert 0.194 vs "drowsy" 0.206. No eye-closure signal to learn
  from for these subjects, sometimes inverted.
- Subject 24 (worst overall): raw signal direction is correct (0.189 ->
  0.163, ~14% drop, comparable magnitude to well-classified subjects) and
  windows are class-balanced (208 drowsy / 196 non-drowsy), but still
  near-chance — likely high within-video variance swamping the mean
  difference at the 45s-window level. This subject's recording is unusually
  long (~18 min vs the typical ~9 min), consistent with more internal
  variability within the single labeled state.
- Verified extraction quality isn't the cause: drop rates for all these
  subjects are ~0% (see `feature_extraction_report.csv`).

**Conclusion:** real inter-subject variability in how drowsiness manifests
physically — some people show it through eye closure (PERCLOS-dominant,
classify well), others don't show much visible eye-closure at all. This
matches the plan's already-documented limitation (self-reported, per-video
binary labels, no objective ground truth). Not a bug to fix; a negative
result to report honestly. Possible future angles if it matters for the
final report: richer features (head-nod rate is currently only pitch_std,
could be more targeted), or the sequence model may pick up temporal
patterns EAR/PERCLOS summary stats miss — but not chasing this now,
86% aggregate accuracy stands as the honest baseline number.

## 2026-09-23 — Plan reassessment (plan.md updated)

**Key finding — Phase 2 accuracy is optimistic.** Phase 2 normalized each subject against their *entire* non-drowsy video, but the live system only gets a short calibration (and non-drowsy test windows were normalized with stats that included themselves). Quick XGBoost check on identical eval windows (first 120 s of every alert video held out), same subject-grouped CV:

| Normalization | Deployable | Acc | AUC |
|---|---|---|---|
| Full-video z-score (Phase 2) | no | 0.848 | 0.914 |
| 30 s z-score (original live plan) | yes | 0.738 | 0.807 |
| 60 s z-score | yes | 0.757 | 0.826 |
| 120 s z-score | yes | 0.775 | 0.851 |
| 30 s median shift | yes | 0.758 | 0.836 |
| 120 s median shift | yes | 0.781 | 0.867 |
| No normalization | yes | 0.723 | 0.790 |

Takeaways: calibration is still worth ~+6 pp over none; median shift beats z-score at equal length; longer calibration helps. Honest deployable XGBoost ≈ 78%, not 86%. All Phase 2 models share this normalization, so all their numbers are upper bounds. (Throwaway inline script, not committed — reproduced by Phase 2b.)

**plan.md changes:**
- Status table at top; corrected dataset facts (10 fps / 224×224, 4–18 min recordings).
- **New Phase 2b (next):** median-shift calibration-window normalization, calibration length 30/60/120 s as an evaluated parameter, re-run all classifiers + ensemble reporting optimistic vs deployable numbers, seed NN runs (mean ± std over 3 seeds), add video-level accuracy.
- **Phase 3 redesign:** forecaster predicts coarse per-5 s aggregates instead of raw 10 fps frames (raw EAR is unforecastable blink noise); Transformer dropped; three predictive signals compared head-to-head (reactive / trend time-to-threshold / forecaster); probability calibration + thresholds tuned on training subjects only; **ramped synthetic transitions** (interleaved alert/drowsy chunks with rising drowsy fraction) replace the hard splice as the primary lead-time benchmark, since a hard splice has no precursor and can't show genuine lead time (kept as a negative control); KSS self-recording should **start now** (raw video + timestamps, processed offline later).
- Phase 4: face-crop to 224×224 before landmarking + feature-distribution check (webcam domain shift). Phase 5 and risk table updated to match.

**Open issues / next steps:**
- `context.md` still says "live system uses a 30 s calibration" — now superseded by Phase 2b's evaluated calibration length. Not edited (user limited this change to plan.md); needs updating once Phase 2b picks the length.
- Calibration length for the live/demo system (30 vs 120 s) is a user-facing trade-off (accuracy vs startup wait) — decide after the Phase 2b curve.
- Hardware still deferred; KSS recordings not yet started.

## 2026-09-23 — Phase 2b: deployment-faithful re-evaluation (DONE)

**Pipeline changes:**
- `src/features/windows.py`: replaced full-video z-score (`normalize_frame_df` / `iter_normalized_subject_labels`) with `calibrate()` (per-channel median + blink threshold from a calibration window) and `apply_calibration()` (median shift) — the same pair the live loop will call. `iter_calibrated_videos(calib_s)` reserves the first 120 s of every alert video for calibration and never evaluates it, so eval windows are identical across calibration lengths. `make_windows()` now returns summary stats **and** raw sequences from the same window boundaries.
- `src/features/build_dataset.py` (replaces `build_windows.py` + `build_sequences.py`): one `features/dataset_cal{N}.npz` per N ∈ {30, 60, 120}, holding tabular + sequence features aligned by construction (removes the `t_start` merge). 11,214 windows / 59 subjects each (subject 42 excluded: no drowsy video).
- `src/models/training.py` (was `sequence_training.py`): shared `load_dataset` / `cv_folds` (one fixed split + leakage assertion) / `evaluate` (acc, f1, auc, **video-level acc** — aggregate window probs per video, 118 videos). NN training is seeded, keeps the dataset GPU-resident (~13–23 s per fold, several times faster than the DataLoader version), and divides each channel by its training-set std (median-shifted inputs are in raw units: EAR ~0.05 vs pose ~5–8°). OOF probabilities saved to `results/oof_*.npy` (gitignored). CLI: `--calib 30 60 120`, `--final N`.
- `classifier_baseline.py` / `_cnn` / `_gru` / `_lstm` use the shared harness. `stacking_ensemble.py` now just combines saved OOFs (no retraining) and writes `results/phase2b_summary.csv`.
- Removed stale artifacts built with the old normalization: `features/windows.parquet`, `features/sequences.npz`, `models/classifier_{baseline.json,cnn.pt,lstm.pt}`, old `classifier_gru.pt`.

**Results (mean ± std over 3 seeds; XGBoost deterministic):**
| Model | 30 s acc / AUC | 60 s acc / AUC | 120 s acc / AUC |
|---|---|---|---|
| XGBoost | 0.758 / 0.836 | 0.753 / 0.841 | 0.781 / 0.867 |
| CNN | 0.751±0.001 / 0.823 | 0.745±0.012 / 0.811 | 0.734±0.004 / 0.800 |
| GRU | 0.780±0.006 / 0.848 | 0.788±0.007 / 0.858 | 0.789±0.006 / 0.850 |
| LSTM | 0.723±0.019 / 0.786 | 0.736±0.005 / 0.800 | 0.758±0.009 / 0.817 |
| Avg ensemble | 0.800 / 0.881 | 0.800 / 0.881 | 0.801 / 0.881 |
| Stacked ensemble | 0.796 / 0.873 | 0.799 / 0.873 | 0.813 / 0.880 |

Video-level acc: GRU 0.825 / 0.859 / 0.845; best overall is the stacked ensemble at 60 s (0.895) and XGBoost at 120 s (0.890).

**Findings:**
- Versus Phase 2 (optimistic): GRU 0.889 → 0.788, stacked ensemble 0.912 → 0.813. Every model lost ~8–13 pp once normalization was made deployable.
- GRU is still the best single model and barely depends on calibration length (+0.9 pp from 30 → 120 s, about one seed std). XGBoost benefits most from longer calibration (+2.3 pp).
- CNN and LSTM fell below XGBoost. The Phase 2 "CNN ≈ GRU" result relied on full-video normalization. LSTM also has the largest seed variance (±0.019 at 30 s).
- Ensembles still add +1–2.5 pp acc and +2–3 pp AUC over GRU. Reported as the ceiling, not deployed.

**Decision:** live calibration = **60 s** (best GRU AUC, +0.8 pp over 30 s, half the wait of 120 s). Deploy model trained on all data: `models/classifier_gru.pt` = {`state_dict`, `channel_scale`, `calib_s`=60} (68K, committed).

**Docs:** `plan.md` (status, Phase 2b results) and `context.md` (core design, calibration rule, repo layout, current status) updated per user request.

**Open issues / next steps:**
- Phase 3 (predictive layer) is next.
- Calibration-length curve figure → Phase 5 report figures (data in `results/phase2b_summary.csv`).
- KSS self-recordings not started; hardware still deferred and not ordered.

## 2026-09-23 — Session paused: checkpoint saved

- Added `CHECKPOINT.md`: current state, every decision made this session with its rationale and who made it, ordered Phase 3 next steps, user-side tasks (KSS recordings, hardware order), open decisions (60 s vs 30 s calibration), resume commands, local-only artifacts and how to regenerate them, and gotchas learned.
- `context.md` now points to `CHECKPOINT.md` as the second file to read, to be updated at the end of every session.
- Cross-session preferences (GPU-first training, honest evaluation, progress visibility) saved to Claude's memory outside the repo.

**Next session starts at:** `CHECKPOINT.md` §3, step 1 (forecaster dataset), optionally preceded by a KSS recording script so self-recordings can begin.

## 2026-09-24 — Hardware shopping list

- Added `hardware_list.md`: exact parts for the Arduino alert loop (Uno R3, common-cathode RGB LED, 3 V coin motor + 2N2222 + 1N4007 + 1 kΩ, 5 V active buzzer, 6×6 mm button, breadboard, jumpers), each with spec, search terms, look-alikes to avoid, and an on-arrival check.
- Camera: the laptop's built-in webcam replaces the USB webcam (user decision). The domain-shift check (Phase 4) matters more now because a laptop camera sees the face from a low angle.
- Next: user orders parts; hardware loop is still deferred until they arrive.

## 2026-09-28 — Phase 3 step 1: forecaster dataset; KSS decision

- **Decision (user):** skip long KSS self-recordings. Middle route: one 5–10 min normal webcam clip in Phase 4 (domain-shift check + demo), optional KSS ratings if tired. Report lead time as validated on synthetic ramps only.
- `src/features/build_forecast_dataset.py` → `features/forecast_cal60.npz`: per-5 s aggregates (same 11 features as the classifier), 60 s median-shift calibration, 12,158 steps over 118 videos (48–216 steps each, 55% drowsy), no NaNs. Rows are grouped by `video_id`, ordered by `step`.
- `iter_calibrated_videos(include_unpaired=True)` added. Subject 42's alert video is 60 s, shorter than the 120 s calibration reserve, so it contributes nothing (the "include 42" plan item is moot). Classifier datasets rebuilt: identical counts (11,214 windows).
- Next: forecaster (12 steps in → 6–12 out) with persistence/linear-trend baselines.

## 2026-09-28 — Phase 3 step 2: forecaster + baselines

`src/models/forecaster.py` (GPU env): GRU encoder (hidden 64) → 12×11 offset from the input-window mean; 60 s in → 60 s out; smooth-L1, 30 epochs; subject-grouped 5-fold CV × 3 seeds; ~9k stride-1 samples; standardized per training fold. `results/phase3_forecaster.csv`. Model on all data: `python -m src.models.forecaster --final` → `models/forecaster.pt`.

Error in training-std units (lower is better; MAE over all 11 features):
| | 10 s | 30 s | 60 s |
|---|---|---|---|
| GRU | 0.445 | 0.475 | 0.485 |
| mean of input window | 0.469 | 0.485 | 0.494 |
| last value | 0.504 | 0.574 | 0.578 |
| linear trend | 0.583 | 0.782 | 1.042 |

Key features (MAE, GRU vs mean baseline): ear_mean 0.477/0.535/0.561 vs 0.504/0.532/0.555; perclos 0.539/0.603/0.608 vs 0.532/0.557/0.565.

**Finding:** the forecaster beats the mean baseline by only ~5% at 10 s and ~2% at 60 s overall, and is *no better* on ear_mean / perclos at 30–60 s (worse on perclos). Linear trend extrapolation is worse than persistence at every horizon. Reason: videos are single-state, so the best predictor of the next minute is "what the last minute looked like"; there are no transitions to learn from. The forecaster cannot be expected to add lead time on real-data training alone; step 4 will test whether it adds anything on ramps.

## 2026-09-29 — Phase 3 step 3: ramped transition builder

`src/eval/ramps.py` → `features/ramps_cal60.npz` (`load_ramps()` yields subject, duration, shape, rep, onset_s, ear_threshold, frames). Per subject, from their own 60 s-calibrated per-frame data: 180 s alert lead-in → ramp of 2 / 5 / 10 min (linear or sigmoid) where each 5–10 s chunk is drowsy with probability p(t) rising 0→1 → 120 s drowsy tail. Hard splice = duration 0 (negative control). 3 random draws per config → 21 sequences × 59 subjects = 1,239 sequences, 204.7 h.
- Onset = nominal time p(t)=0.5 (ramp midpoint; splice: 180 s), not the realized-fraction crossing, which is noisy with 5–10 s chunks.
- Self-check (alert=0, drowsy=1 stand-ins, 200 draws): lead-in is exactly 0, tail exactly 1, mean fraction at midpoint 0.48 (linear) / 0.54 (sigmoid).
- Chunks are drawn with replacement from a subject's own footage, so ramps of one subject share source frames; downstream evaluation must stay subject-grouped (train on other subjects, score this subject's ramps).

## 2026-09-29 — Autonomous decisions rule adopted

**User rule:** weigh alternatives independently, choose, implement, test, keep going — ask only for information only the user has. Saved to Claude memory (`feedback_autonomous_decisions.md`) for future sessions.

## 2026-09-29 — Phase 3 step 4-5: risk fusion, lead-time evaluation (DONE)

`src/eval/lead_time.py` (GPU env). Per subject-grouped fold (same folds as the classifier): retrains the GRU classifier, the forecaster, and a new **step-GRU** (a GRU classifier reading 9x5s aggregate blocks = 45s, the only architecture that can score a forecast output). Streams every held-out subject's real alert video and their ramps through 5s decisions, Platt-calibrates each classifier's raw output using *other* folds' OOF predictions, computes:
- `reactive` = calibrated p_now (raw-frame GRU, 45s window)
- `trend` = damped/smoothed trend on p_now (see below)
- `fc30`/`fc60` = forecaster's 30s/60s-ahead prediction -> step-GRU
- `step_now` = step-GRU on the last 45s of blocks, no forecast (control, isolates whether *forecasting* adds anything beyond the step-GRU architecture itself)
- `fused_mean`/`fused_max`/`fused_rf` = combinations

Alarm = signal >= threshold on 2 consecutive decisions, 60s refractory. Threshold per (signal, budget) chosen from *other* folds' real alert videos only; lead time and hit rate scored on held-out ramps. `results/phase3_leadtime.csv` (operating points at 1/3/6 FA/h), `results/phase3_curve.csv` (full threshold sweep).

**Bug fixed:** `load_ramps()` re-decompressed the 61MB `frames` array from the npz on every single yielded item (NpzFile lazy-decompresses per key-access) — 1,239x redundant decompression, OOM'd at ~150MB/access. Fixed by reading each array out of the NpzFile once before the loop.

**Finding — naive linear trend-on-p_now is unusable, root-caused and fixed twice, still unusable:**
1. First version (OLS slope over 12 raw p_now values, extrapolated 87.5s ahead) never dropped below 8.27 FA/h at *any* threshold up to 0.995 — it saturates at ~1 constantly. Root cause: p_now is already a noisy classifier output; extrapolating a slope fit to 12 noisy points 17.5 steps ahead amplifies that noise past the [0,1] clip bound routinely.
2. Applied Holt-style geometric damping (phi=0.9) instead of a flat 12-step jump: floor dropped to 5.58 FA/h, still unusable.
3. Applied EWMA smoothing (alpha=0.3) to p_now *before* fitting the slope (fixing the actual root cause — noisy input, not just the extrapolation multiplier): floor dropped to 4.5 FA/h. Still can't hit even a 6/h budget reliably.
**Conclusion:** a slope-based "is p_now rising" signal is structurally unreliable at this window/noise level, independent of damping or smoothing — p_now swings enough on true non-drowsy footage that its derivative is mostly noise. Not deployed. `fused_max`/plain `fused_mean` inherit this instability; kept in the CSV as a documented negative result, not recommended.

**Results at practical operating points (median lead time in s, +=before onset; hit rate; ramp duration = time from 0% to 100% drowsy):**
| Signal | Budget (FA/h) | actual FA/h | hit@120s | lead@120s | hit@300s | lead@300s | hit@600s | lead@600s |
|---|---|---|---|---|---|---|---|---|
| reactive | 3 | 4.14 | 0.58 | -45 | 0.58 | -35 | 0.62 | -10 |
| fc30 (forecaster) | 3 | 3.42 | 0.69 | -25 | 0.76 | -7.5 | 0.79 | **+45** |
| step_now (control) | 3 | 3.60 | 0.78 | -35 | 0.82 | -5 | 0.86 | **+50** |
| fused_rf (reactive+fc30, recommended) | 3 | 3.24 | 0.70 | -30 | 0.75 | -10 | 0.78 | +35 |

**Negative control confirmed:** the hard splice (dur=0, no precursor) shows negative median lead for every signal (-20 to -50s) — no signal hallucinates advance warning where there is none, as expected.

**Headline finding:** no signal gives positive lead time on fast transitions (2 min ramps: all median leads negative — the transition is faster than any window-based method can anticipate). On slow transitions (10 min ramps, plausible for real drowsiness onset), several signals give ~35-50s of genuine lead time at a 3 FA/h budget, with step_now (the aggregate classifier alone, no forecast) matching or beating the forecaster-based signals. **The forecaster does not clearly add value over just running a classifier on 5s aggregates reactively** — same conclusion as step 2 (forecaster ~ mean baseline), now confirmed at the level that matters (lead time), not just forecast MAE.

**Recommendation for the report/demo:** deploy `reactive` (fine-grained, lowest latency) for the primary alert, optionally blended with `fc30` (`fused_rf`) for a small lead-time gain on slow onsets — both calibrated per the CSV. Do not deploy `trend` or a forecaster-only signal; the step-GRU-on-aggregates result suggests any future work should look at longer classifier context windows before building a heavier forecaster.

**Open issues / next steps:** Phase 3 is functionally complete (dataset, forecaster, ramps, fusion, lead-time eval all done and honestly reported). Remaining before Phase 4: pick final deploy signal (recommend `reactive` alone, given `fused_rf`'s modest gain vs added complexity — GRU forecaster + step-GRU in the loop). Phase 4 (real-time loop, webcam domain check, one 5-10 min recording) is next.

## 2026-09-29 — Phase 4 step 1: face-crop webcam pipeline + domain-shift check

**User recorded** a 12.3 min full-frame webcam clip (1920x1080, 29.7 fps), acting out alert -> drowsy -> alert: `C:\Users\HP VICTUS\Pictures\Camera Roll\WIN_20260929_00_39_13_Pro.mp4` (not in the repo, personal footage).

- `src/features/webcam.py`: `process_webcam_video()` — nearest-frame subsample to 10 fps, then per frame: crop to the face's bounding box (landmarks + 40% margin, squared) and resize to 224x224 before running the same `frame_features()` training uses. **Two separate `FaceDetector` instances, not one, and this is deliberate**: the bbox pass is IMAGE mode (each frame's box is unrelated to the last, so no reason to feed it to a temporal tracker); the crop pass is VIDEO mode (consecutive 224x224 crops *are* a smooth video, same framing convention as training data, so VIDEO mode's tracking is valid there, matching how `extract_features.py` runs on training clips). Mixing both passes into one VIDEO-mode instance would have fed it alternating raw-frame/crop images and broken its continuity assumption.
- `src/eval/domain_shift.py`: runs the above, saves `features/webcam_demo.parquet` (gitignored, personal footage-derived), compares against training. **0% face-drop rate** over the whole 12.3 min clip.
- **Methodology self-correction:** first pass compared *raw* features pooled across all 59 training subjects against the webcam session — flawed, since pooling raw values across different people mixes genuine person-to-person variation into what's supposed to be a camera/framing check, swamping any real signal (e.g. raw EAR looked wildly different, but that's just this person's resting eye shape). Redid the comparison on **calibrated** features (median-shifted from each session's own first 60 s, calib_s=60, reusing `iter_calibrated_videos`) — the same feature space the classifier actually trains and infers on, and the only fair apples-to-apples comparison.

**Calibrated comparison (webcam session vs. all training videos' calibrated features):**
| Feature | webcam std | train std | ratio |
|---|---|---|---|
| ear | 0.046 | 0.048 | 95% — matches well |
| mar | 0.005 | 0.047 | 11% — far narrower |
| pitch | 4.64 | 7.26 | 64% |
| yaw | 4.54 | 7.94 | 57% |
| roll | 6.00 | 5.51 | 109% |

**Finding:** EAR — the dominant drowsiness signal per the earlier per-subject investigation — transfers well after calibration; the webcam pipeline is trustworthy for the feature that matters most. MAR and head pitch/yaw show noticeably less variation in this session than in training, consistent with a lower-angle laptop camera and/or one person's own range of motion during a single scripted demo rather than 59 different people's natural behavior. **Caveat: n=1 person, n=1 session** — this is a spot-check, not a validated general claim; report it as a limitation, not a conclusion.

**Not committed yet.** Next: pick the reactive signal's deploy path (Phase 3 recommendation), then wire the calibration -> loop -> serial protocol for the live system, using this same `webcam.py` preprocessing.

## 2026-09-29 — Phase 4 step 2: live loop built and run end-to-end on real footage

- `src/realtime/risk.py`: p_now -> alert level 0-3 (thresholds 0.3/0.5/0.75, a starting point for the demo, not yet tuned against `phase3_curve.csv`). `demo()` self-check.
- `src/realtime/serial_client.py`: wraps pyserial; with no port (hardware not ordered yet) or pyserial/port unavailable, logs `[serial] R:<level>` instead of raising — the software loop is fully runnable and testable without the Arduino attached. `demo()` self-check.
- `src/realtime/live_loop.py`: calibrate (first `calib_s`, from the checkpoint) -> forward-fill short face-loss gaps (<0.5s, matches training's interpolation cutoff; longer gaps clear the window rather than alert on stale data) -> 45s window -> deploy GRU -> `risk_level()` -> serial at ~2Hz, `ACK` polling, CSV log of every decision (`results/live_log_<ts>.csv`, gitignored). Works against a live webcam (`--source 0`) or a video file (`--source path`), same code path — reuses `src.features.webcam.LiveFeatureExtractor` (refactored `process_webcam_video` to use it too, one implementation of frame->feature instead of two).
- **Bug fixed:** `torch.load` on `models/classifier_gru.pt` failed under PyTorch 2.6+'s new `weights_only=True` default (the checkpoint's `channel_scale` is a numpy array). Fixed with `weights_only=False` — it's our own checkpoint, not an untrusted download.
- **Ran end-to-end against the user's real recording** (12.3 min, acted alert->drowsy->alert): 60s calibration, then 633 decisions over ~11.5 min, no crashes, serial/CSV logging worked. Level distribution: 0:77, 1:78, 2:59, **3:419 (66%)**.

**Finding — the model's output does not clearly track the intended narrative on this clip.** Aligning p_now against raw EAR/pitch over time: p_now is high (0.85-0.99, i.e. "drowsy") through most of the first ~7 minutes, dips low (0.13-0.46, i.e. "alert") for a stretch around t=550-705s, then jumps back to 0.998 in the final 30s. Raw EAR barely moves the whole session (0.27-0.35), consistent with the domain-shift check's earlier flag (webcam pitch/yaw variance is narrower than training's) — there may simply not be enough signal in this one take for the classifier to key on, or the model's 78.8% cross-validated accuracy just doesn't hold up on an unseen real-world subject/camera. **Not chasing this by retuning thresholds until it "looks right"** — that would be fitting the demo, not evaluating it. Logged honestly; loop mechanics (calibration, buffering, inference, risk mapping, serial, logging) are verified working, the model's real-world accuracy on this clip is not.

**Open question for the user:** what was the actual timeline of the acting (roughly when did alert / drowsy / alert start, in clip time)? Needed to know whether p_now's dip-then-rise pattern is inverted relative to what was intended, or whether the "acting" happened at different times than assumed.

**Not committed yet.**

## 2026-09-29 — Correction: live-loop finding was based on a wrong assumption

**User confirmed the p_now trace is fairly correct**, not inverted. My previous entry assumed a single alert(start)->drowsy(middle)->alert(end) hump; the user's actual acting had a more layered timeline (drowsy for most of the first ~7 min of decisions, a genuine alert recovery ~9-12 min, drowsy again in the final 30s) that matches what the model produced. **Retracting the "doesn't track the narrative" concern** — on this one clip, the reactive GRU signal does track the acted state reasonably well end-to-end (webcam capture -> face-crop -> calibration -> GRU -> risk level), which is a positive result for the live loop, not a negative one. Domain-shift caveats (narrower MAR/pitch/yaw variance than training, n=1 session) still stand as limitations to mention in the report, but are not contradicted by this result.

## 2026-09-29 — Design decision: live loop stays reactive-only, no forecaster

Evaluated wiring the forecaster + step-GRU into `live_loop.py` for `p_future` (plan's D2 checklist item) against keeping it reactive-only (current state). Per Phase 3's own lead-time evaluation: `fc30` (forecaster) beat `reactive` by only ~35-50s of lead time, only on slow (10 min) transitions, and `step_now` (a classifier on aggregates, no forecast at all) matched or beat it. Wiring the forecaster in doubles the models running in the loop (forecaster GRU + a persisted step-GRU classifier, neither of which currently has a `--final` deploy checkpoint saved) for a gain that's modest and concentrated in a transition speed real driving may or may not produce. **Decision: keep the deployed loop reactive-only** (simpler, lower latency, matches the Phase 3 recommendation). The forecaster stays available as a Phase 5 ablation/comparison result, not a deployed component. Revisit only if the user wants that extra lead time enough to justify the added complexity and latency.

## 2026-09-29 — Phase 4 step 3: latency/FPS measurement

Added `LiveLoop.latency_report()`: wall-clock time per processing tick (frame -> face-crop -> landmark -> feature -> decision-if-due -> serial write), i.e. frame-to-serial-write latency; the actuator's own response after that (Arduino digitalWrite/PWM) is untestable without the hardware attached (not ordered yet), but is expected to be sub-millisecond and not the bottleneck.

3-minute benchmark (CPU env, 1,201 ticks @ 10 fps target): **mean 37.7ms, p95 52.4ms, max 68.7ms**, against a 100ms/tick budget (10 fps) — **0% of ticks over budget**. Sustainable throughput ≈ 26 fps, well above the 10 fps target, so the CPU pipeline is not a bottleneck for a live camera feed either. `--max-s` flag added to `live_loop.py` for short benchmark runs without processing a whole recording.

## 2026-09-29 — Phase 4 step 4: live overlay dashboard

`draw_overlay()` + `--show` in `live_loop.py`: on-frame EAR, PERCLOS (over the current 45s window), p_now, and alert level, with a color-coded border (green/yellow/orange/red matching the risk levels). Chosen over a separate Streamlit app: no new process/dependency, fits directly in the loop already running, and a physical demo (laptop + screen) doesn't need a browser. Sanity-checked `draw_overlay()` directly on a dummy frame (this environment has no display to open a real cv2 window) — **user should try `--source 0 --show` on their own machine** to see the live window.

**Phase 4 is now functionally complete**: face-crop webcam pipeline, domain-shift check, live loop (calibration -> GRU -> risk -> serial, works with or without hardware attached), latency/FPS measurement (well within budget), on-frame dashboard. Not yet possible: true actuator latency and the physical demo script (hardware not ordered/arrived).

## 2026-09-29 — Phase 5 step 1: consolidated report + figures

`src/eval/make_report_figures.py`: 4 figures from existing result tables (no new experiments) — `results/figures/{calibration_curve,forecaster_baselines,leadtime_curve,leadtime_vs_duration}.png`.
`REPORT.md`: consolidated Phase 2-4 results (optimistic vs deployable classifier numbers, forecaster vs baselines, lead-time/false-alarm results, domain-shift + live-loop findings, latency), what's not done and why (hardware, KSS recordings), and a limitations section for the write-up. Everything traces back to the dated `PROGRESS.md` entry that produced it.

**Remaining for Phase 5:** the physical demo script needs the hardware (not ordered). Per-subject variability write-up and different-horizon ablations are already covered in REPORT.md / Phase 3 results; no further analysis planned unless the user asks.

## 2026-09-29 — Live-loop accuracy fixes: threshold recentering + closure override

**User reported live testing was bad: stuck at high alert, unresponsive to real drowsiness, felt laggy.** Before changing anything, ruled out a live-pipeline bug: ran the exact same buffering/scaling/model code from `live_loop.py` against known-labeled lab data (subjects 01/05/20). Found the same failure modes reproduced on lab data (same domain the model was trained/validated on):
- Subject 20's own known-**alert** footage scored p_now=0.816 — already "level 3" while genuinely alert.
- Subject 05 barely separated alert (0.307) from drowsy (0.242) at all — inverted.
Confirms this is the known per-subject variability + ~79% accuracy ceiling (already documented), not a coding bug — the live pipeline is correct.

**Two evidence-targeted fixes, no retraining needed:**
1. **`risk.recenter()`** (logit-space): the 60s calibration window already collected for feature calibration is now also scored by the model itself (last 45s of it), giving each person's own baseline p_now. Live decisions are recentered so that baseline maps to a fixed low target (0.10) instead of wherever the raw model happened to put it. Verified on the same lab subjects: subject 20's alert-footage level dropped 3→2 (no longer pinned red), subject 01 (already good) unaffected, subject 05 (genuinely low-signal, not a bias problem) correctly *not* "fixed" — recentering can't manufacture a discrimination the model doesn't have. `p_now_raw` and `p_now` (recentered) both logged for transparency.
2. **`risk.apply_closure_override()`**: the GRU reasons over a 45s window, so a few seconds of closed eyes barely moves it — inherent to a model trained/evaluated at that window length, not a bug. Sustained eye closure (tracked every tick from the calibrated EAR threshold, ~100ms resolution) now forces level ≥2 past 1.5s and level 3 past 3.0s, independent of the GRU. Fixed cutoffs, not tuned against real closure-duration data (none exists yet) — `ponytail:` comment marks this for retuning once real sessions are available.

Both are pure functions in `risk.py` with `demo()` self-checks (recenter identity-at-baseline, override never-downward, threshold crossings).

**Re-verified end-to-end on the user's real recording** (3 min excerpt): level distribution shifted from {0:1, 1:2, 2:8, 3:65} (85% level-3) to **{0:19, 1:6, 2:16, 3:35} (46% level-3)** — same footage, same model, meaningfully less "stuck high". Latency unaffected (mean 35.7ms, still well under the 100ms budget).

**Not fully solved and won't overclaim it:** genuine low-signal subjects (like 05) and the underlying ~79% ceiling are real, data-level limits — no threshold engineering fixes those. Recentering only removes *systematic bias*, not classification *noise*. User should re-test live and report whether the remaining behavior is acceptable or still needs work.
