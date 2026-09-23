"""Baseline drowsiness classifier: window summary stats -> XGBoost (GPU).
Same fixed subject-grouped folds and metrics as the sequence models. Deterministic
(no subsampling), so one run per calibration length instead of 3 seeds.

Usage: run via the GPU conda env:
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.classifier_baseline
"""
import numpy as np
import xgboost as xgb

from src.models.training import CALIB_LENGTHS_S, RESULTS_DIR, cv_folds, evaluate, fmt, load_dataset, oof_path


def make_xgb():
    return xgb.XGBClassifier(
        device="cuda", tree_method="hist", n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss"
    )


def run_cv(calib_s: int) -> dict:
    d = load_dataset(calib_s)
    X, y, subjects = d["X_tab"], d["y"], d["subjects"]
    oof = np.zeros(len(y))
    for tr, te in cv_folds(y, subjects):
        clf = make_xgb()
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(oof_path("xgboost", calib_s), oof)
    m = evaluate(y, oof, subjects)
    print(f"== xgboost cal={calib_s}s: {fmt(m)}", flush=True)
    return m


if __name__ == "__main__":
    for n in CALIB_LENGTHS_S:
        run_cv(n)
