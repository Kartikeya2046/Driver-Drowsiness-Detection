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
