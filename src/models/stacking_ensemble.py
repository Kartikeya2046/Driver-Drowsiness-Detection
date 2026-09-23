"""Ensembles + Phase 2b summary table, built from saved out-of-fold predictions
(results/oof_*.npy) — no retraining. All models share one fixed split and one
dataset file per calibration length, so OOF rows are aligned by construction.

Stacking stays non-leaky: every OOF probability came from a model that never saw
that window's subject, and the logistic-regression meta-learner for each fold is
fit only on the other folds' OOF rows.

Run after classifier_baseline / _cnn / _gru / _lstm:
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.stacking_ensemble
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.training import CALIB_LENGTHS_S, RESULTS_DIR, SEEDS, cv_folds, evaluate, load_dataset, oof_path

NN_MODELS = ("cnn", "gru", "lstm")


def stack(P: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, te in folds:
        out[te] = LogisticRegression().fit(P[tr], y[tr]).predict_proba(P[te])[:, 1]
    return out


def main() -> None:
    rows = []
    for calib_s in CALIB_LENGTHS_S:
        d = load_dataset(calib_s)
        y, subjects = d["y"], d["subjects"]
        folds = cv_folds(y, subjects)
        xgb_oof = np.load(oof_path("xgboost", calib_s))
        rows.append({"model": "xgboost", "calib_s": calib_s, "seed": 0, **evaluate(y, xgb_oof, subjects)})
        for seed in SEEDS:
            nn_oof = {m: np.load(oof_path(m, calib_s, seed)) for m in NN_MODELS}
            for m, p in nn_oof.items():
                rows.append({"model": m, "calib_s": calib_s, "seed": seed, **evaluate(y, p, subjects)})
            P = np.column_stack([xgb_oof, *nn_oof.values()])
            rows.append({"model": "avg_ensemble", "calib_s": calib_s, "seed": seed, **evaluate(y, P.mean(1), subjects)})
            rows.append({"model": "stacked_ensemble", "calib_s": calib_s, "seed": seed, **evaluate(y, stack(P, y, folds), subjects)})

    per_seed = pd.DataFrame(rows)
    metrics = ["acc", "f1", "auc", "video_acc"]
    summary = per_seed.groupby(["model", "calib_s"], sort=False)[metrics].agg(["mean", "std"]).fillna(0.0)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    out = RESULTS_DIR / "phase2b_summary.csv"
    summary.round(4).to_csv(out, index=False)

    for calib_s in CALIB_LENGTHS_S:
        print(f"\n=== calibration {calib_s}s (mean ± std over seeds; xgboost deterministic) ===")
        for _, r in summary[summary["calib_s"] == calib_s].iterrows():
            print(f"{r['model']:17s} " + "  ".join(f"{m}={r[m + '_mean']:.3f}±{r[m + '_std']:.3f}" for m in metrics))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
