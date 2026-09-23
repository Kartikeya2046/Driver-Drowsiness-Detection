"""Stacking ensemble over XGBoost + CNN + GRU + LSTM.

Honest (non-leaky) stacking: for every window, each base model's prediction
comes from a fold where that window's subject was held out. A meta-learner
(logistic regression) is then trained on those out-of-fold base predictions,
itself evaluated via the same subject-grouped folds so the meta-learner also
never sees a fold's subjects during its own fitting.

windows.parquet (XGBoost's summary-stat features) and sequences.npz (raw
arrays for CNN/GRU/LSTM) are two independently-built files - aligned here by
an explicit (subject, label_binary, t_start) merge instead of assuming row
order matches, since a silent misalignment would corrupt every result without
raising an error.

Usage: run via the GPU conda env:
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.stacking_ensemble
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.models.classifier_baseline import FEATURE_COLS
from src.models.classifier_cnn import DrowsinessCNN
from src.models.classifier_gru import DrowsinessGRU
from src.models.classifier_lstm import DrowsinessLSTM
from src.models.sequence_training import DEVICE, train_one_fold

FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "features"
SEQ_MODELS = {"cnn": DrowsinessCNN, "gru": DrowsinessGRU, "lstm": DrowsinessLSTM}
BASE_MODEL_NAMES = ["xgboost", "cnn", "gru", "lstm"]


def load_aligned():
    win = pd.read_parquet(FEATURES_DIR / "windows.parquet")
    win = win[win["usable_for_classifier"]].reset_index(drop=True)

    seq = np.load(FEATURES_DIR / "sequences.npz", allow_pickle=True)
    seq_mask = seq["usable"].astype(bool)
    seq_df = pd.DataFrame(
        {
            "subject": seq["subjects"][seq_mask],
            "label_binary": seq["y"][seq_mask],
            "t_start": seq["t_start"][seq_mask],
            "seq_idx": np.arange(seq_mask.sum()),
        }
    )
    X_seq_all = seq["X"][seq_mask]

    merged = win.merge(seq_df, on=["subject", "label_binary", "t_start"], how="inner", validate="one_to_one")
    assert len(merged) == len(win), f"alignment dropped rows: {len(win)} -> {len(merged)}"

    X_tab = merged[FEATURE_COLS].to_numpy()
    X_seq = X_seq_all[merged["seq_idx"].to_numpy()]
    y = merged["label_binary"].to_numpy()
    groups = merged["subject"].to_numpy()
    return X_tab, X_seq, y, groups, merged


def xgb_oof_fold(X_tab_train, y_train, X_tab_test):
    clf = xgb.XGBClassifier(
        device="cuda", tree_method="hist", n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss"
    )
    clf.fit(X_tab_train, y_train)
    return clf.predict_proba(X_tab_test)[:, 1]


def main() -> None:
    X_tab, X_seq, y, groups, merged = load_aligned()
    n = len(y)
    print(f"Aligned dataset: {n} windows, {len(np.unique(groups))} subjects")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(cv.split(X_tab, y, groups))

    oof_proba = np.zeros((n, len(BASE_MODEL_NAMES)))
    fold_id = np.zeros(n, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(folds):
        assert set(groups[train_idx]).isdisjoint(groups[test_idx]), "subject leakage across folds"
        fold_id[test_idx] = fold
        print(f"\n--- fold {fold} (n_test={len(test_idx)}) ---")

        oof_proba[test_idx, 0] = xgb_oof_fold(X_tab[train_idx], y[train_idx], X_tab[test_idx])
        print(f"  xgboost done")

        for col, name in enumerate(["cnn", "gru", "lstm"], start=1):
            _, proba = train_one_fold(
                SEQ_MODELS[name], X_seq[train_idx], y[train_idx].astype(np.float32), X_seq[test_idx], y[test_idx]
            )
            oof_proba[test_idx, col] = proba
            print(f"  {name} done")

    print("\n=== Base model OOF performance (sanity check vs earlier standalone runs) ===")
    for col, name in enumerate(BASE_MODEL_NAMES):
        pred = (oof_proba[:, col] >= 0.5).astype(int)
        print(
            f"{name}: acc={accuracy_score(y, pred):.3f} f1={f1_score(y, pred):.3f} "
            f"auc={roc_auc_score(y, oof_proba[:, col]):.3f}"
        )

    # simple average ensemble (no training, sanity baseline)
    avg_proba = oof_proba.mean(axis=1)
    avg_pred = (avg_proba >= 0.5).astype(int)
    print("\n=== Average ensemble (unweighted mean of 4 OOF probabilities) ===")
    print(f"acc={accuracy_score(y, avg_pred):.3f} f1={f1_score(y, avg_pred):.3f} auc={roc_auc_score(y, avg_proba):.3f}")

    # stacked meta-learner: logistic regression on OOF probs, evaluated via the
    # same folds (meta-train on other folds' OOF probs, meta-test on this fold's)
    stack_pred = np.zeros(n, dtype=int)
    stack_proba = np.zeros(n)
    fold_metrics = []
    for fold in range(5):
        meta_train = fold_id != fold
        meta_test = fold_id == fold
        meta = LogisticRegression()
        meta.fit(oof_proba[meta_train], y[meta_train])
        proba = meta.predict_proba(oof_proba[meta_test])[:, 1]
        stack_proba[meta_test] = proba
        stack_pred[meta_test] = (proba >= 0.5).astype(int)
        acc = accuracy_score(y[meta_test], stack_pred[meta_test])
        f1 = f1_score(y[meta_test], stack_pred[meta_test])
        auc = roc_auc_score(y[meta_test], proba)
        fold_metrics.append({"fold": fold, "acc": acc, "f1": f1, "auc": auc})
        print(f"stack fold {fold}: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}, meta weights={meta.coef_.round(3)}")

    metrics_df = pd.DataFrame(fold_metrics)
    print("\n=== Stacked ensemble (logistic regression meta-learner) ===")
    print(metrics_df[["acc", "f1", "auc"]].agg(["mean", "std"]))
    print(f"Overall: acc={accuracy_score(y, stack_pred):.3f} f1={f1_score(y, stack_pred):.3f} auc={roc_auc_score(y, stack_proba):.3f}")

    merged["stack_correct"] = stack_pred == y
    per_subj = merged.groupby("subject")["stack_correct"].mean().sort_values()
    print(f"\nWorst 5 subjects (stacked ensemble): {per_subj.head(5).to_dict()}")


if __name__ == "__main__":
    main()
