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
- Phase 1 (feature extraction) plan needs a small update before starting: drop the resample-by-timestamp step for training data (keep it for the live path), and decide which `lenN` variant(s) to actually load. Recommend `len60` as primary since it best matches the 30-60s window design; flag to user before locking this in since it affects Phase 1 design.
- Hardware not yet ordered (plan Phase 0 item — outside this session's scope, user-side action).
- Have not yet committed to git — pending user confirmation on first commit.
