"""Shared CV harness for every classifier (XGBoost, CNN, GRU, LSTM).

Deployment-faithful protocol (Phase 2b): features/dataset_cal{N}.npz, where N is
the calibration length in seconds; subject-grouped 5-fold CV with one fixed
split; out-of-fold probabilities saved to results/ so ensembles combine them
without retraining.

Model scripts call main(), e.g.:
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.classifier_gru            # CV, all N, 3 seeds
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.classifier_gru --final 120  # train deploy model
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CALIB_LENGTHS_S = (30, 60, 120)
SEEDS = (0, 1, 2)
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3


def load_dataset(calib_s: int):
    return np.load(FEATURES_DIR / f"dataset_cal{calib_s}.npz")


def cv_folds(y: np.ndarray, subjects: np.ndarray) -> list:
    """The one fixed split every model uses (same seed + same row order -> same folds)."""
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, subjects))
    for tr, te in folds:
        assert set(subjects[tr]).isdisjoint(subjects[te]), "subject leakage across folds"
    return folds


def evaluate(y: np.ndarray, proba: np.ndarray, subjects: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    videos = pd.DataFrame({"s": subjects, "y": y, "p": proba}).groupby(["s", "y"], as_index=False)["p"].mean()
    return {
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, proba),
        "video_acc": float(((videos["p"] >= 0.5).astype(int) == videos["y"]).mean()),
    }


def fmt(m: dict) -> str:
    return " ".join(f"{k}={v:.3f}" for k, v in m.items())


def oof_path(name: str, calib_s: int, seed: int | None = None) -> Path:
    suffix = "" if seed is None else f"_seed{seed}"
    return RESULTS_DIR / f"oof_{name}_cal{calib_s}{suffix}.npy"


def fit_torch(model_ctor, X: np.ndarray, y: np.ndarray, seed: int):
    """Returns (model, channel_scale). Inputs are median-shifted but still in raw
    units (EAR ~0.05, pose ~10 deg), so each channel is divided by its training-set std."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    scale = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    Xt = torch.from_numpy(X / scale).float().to(DEVICE)
    yt = torch.from_numpy(y).float().to(DEVICE)

    model = model_ctor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=gen).to(DEVICE)
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model, scale


@torch.no_grad()
def predict_torch(model, scale: np.ndarray, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X / scale).float().to(DEVICE))).cpu().numpy()


def run_cv(model_ctor, name: str, calib_s: int) -> list[dict]:
    d = load_dataset(calib_s)
    X, y, subjects = d["X_seq"], d["y"], d["subjects"]
    folds = cv_folds(y, subjects)
    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for seed in SEEDS:
        oof = np.zeros(len(y))
        for tr, te in folds:
            model, scale = fit_torch(model_ctor, X[tr], y[tr], seed)
            oof[te] = predict_torch(model, scale, X[te])
        np.save(oof_path(name, calib_s, seed), oof)
        m = evaluate(y, oof, subjects)
        results.append(m)
        print(f"{name} cal={calib_s}s seed={seed}: {fmt(m)}", flush=True)
    return results


def save_final(model_ctor, name: str, calib_s: int) -> None:
    d = load_dataset(calib_s)
    model, scale = fit_torch(model_ctor, d["X_seq"], d["y"], seed=0)
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"classifier_{name}.pt"
    torch.save({"state_dict": model.state_dict(), "channel_scale": scale, "calib_s": calib_s}, path)
    print(f"Saved {path} (calib_s={calib_s})")


def main(model_ctor, name: str) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, nargs="+", default=list(CALIB_LENGTHS_S))
    ap.add_argument("--final", type=int, metavar="CALIB_S", help="train the deploy model on all data instead of CV")
    args = ap.parse_args()
    if args.final:
        save_final(model_ctor, name, args.final)
        return
    for calib_s in args.calib:
        res = pd.DataFrame(run_cv(model_ctor, name, calib_s))
        print(f"== {name} cal={calib_s}s mean±std over {len(SEEDS)} seeds: "
              + " ".join(f"{k}={res[k].mean():.3f}±{res[k].std():.3f}" for k in res), flush=True)
