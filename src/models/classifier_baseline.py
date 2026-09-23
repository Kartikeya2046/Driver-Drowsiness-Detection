"""Baseline drowsiness classifier: window summary stats -> XGBoost (GPU).
Subject-grouped 5-fold CV — never split by window/frame (leakage).

Usage: run via the GPU conda env, e.g.
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.classifier_baseline
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "features"
FEATURE_COLS = [
    "ear_mean", "ear_std", "ear_min",
    "mar_mean", "mar_std", "mar_max",
    "pitch_std", "yaw_std",
    "perclos", "blink_rate_per_min", "mean_blink_duration_s",
]


def main() -> None:
    df = pd.read_parquet(FEATURES_DIR / "windows.parquet")
    df = df[df["usable_for_classifier"]].reset_index(drop=True)

    X = df[FEATURE_COLS].to_numpy()
    y = df["label_binary"].to_numpy()
    groups = df["subject"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        assert set(groups[train_idx]).isdisjoint(groups[test_idx]), "subject leakage across folds"

        clf = xgb.XGBClassifier(
            device="cuda", tree_method="hist",
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss",
        )
        clf.fit(X[train_idx], y[train_idx])

        pred = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]

        acc = accuracy_score(y[test_idx], pred)
        f1 = f1_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], proba)

        test_df = df.iloc[test_idx].copy()
        test_df["pred"] = pred
        per_subj_acc = test_df.groupby("subject").apply(
            lambda g: accuracy_score(g["label_binary"], g["pred"]), include_groups=False
        )

        print(
            f"fold {fold}: n_test={len(test_idx)}, subjects={sorted(set(groups[test_idx]))}, "
            f"acc={acc:.3f} f1={f1:.3f} auc={auc:.3f} "
            f"per_subject_acc[min={per_subj_acc.min():.3f}, mean={per_subj_acc.mean():.3f}]"
        )
        fold_metrics.append({"fold": fold, "acc": acc, "f1": f1, "auc": auc})

    metrics_df = pd.DataFrame(fold_metrics)
    print("\n=== 5-fold subject-grouped CV summary ===")
    print(metrics_df[["acc", "f1", "auc"]].agg(["mean", "std"]))

    # final model on all usable data, for use in the live loop
    final_clf = xgb.XGBClassifier(
        device="cuda", tree_method="hist",
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss",
    )
    final_clf.fit(X, y)
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "classifier_baseline.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_clf.save_model(str(model_path))
    print(f"\nSaved final model (trained on all usable windows) to {model_path}")


if __name__ == "__main__":
    main()
