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
