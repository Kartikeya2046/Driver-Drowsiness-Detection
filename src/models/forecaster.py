"""Phase 3 forecaster: 60 s of per-5 s aggregate features in -> next 60 s out.

Trained on real within-video data only (never on synthetic ramps). Compared with three
baselines that need no training: last value, mean of the input window, linear trend.
Errors are in units of the training-set feature std, so features are comparable.
Subject-grouped 5-fold CV, 3 seeds; horizons reported at 10 / 30 / 60 s.

Usage (GPU env):
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.forecaster [--final]
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from src.models.training import DEVICE, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, SEEDS, cv_folds

T_IN, T_OUT, STEP_S = 12, 12, 5
HORIZONS = {"10s": 2, "30s": 6, "60s": 12}  # forecast steps (1-indexed)
KEY_FEATURES = ("ear_mean", "perclos")
EPOCHS, BATCH, LR = 30, 128, 1e-3


class Forecaster(nn.Module):
    """GRU encoder -> linear head. Predicts the whole horizon at once as an offset from
    the input-window mean, so an untrained model equals the strong mean baseline."""

    def __init__(self, n_feat: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, T_OUT * n_feat)
        self.n_feat = n_feat

    def forward(self, x):  # x: (B, T_IN, F) standardized
        _, h = self.gru(x)
        delta = self.head(h.squeeze(0)).view(-1, T_OUT, self.n_feat)
        return x.mean(dim=1, keepdim=True) + delta


def make_samples(X: np.ndarray, video_id: np.ndarray):
    """Slide a (T_IN + T_OUT)-step window with stride 1 inside each video.
    Returns inputs, targets, and each sample's row index of its first step."""
    xs, ys, rows = [], [], []
    for v in np.unique(video_id):
        idx = np.flatnonzero(video_id == v)
        if len(idx) < T_IN + T_OUT:
            continue
        w = sliding_window_view(X[idx], T_IN + T_OUT, axis=0).transpose(0, 2, 1)  # (n, T, F)
        xs.append(w[:, :T_IN])
        ys.append(w[:, T_IN:])
        rows.append(idx[: len(w)])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rows)


def baselines(x: np.ndarray) -> dict:
    """x: (N, T_IN, F) -> {name: (N, T_OUT, F)}"""
    n, _, f = x.shape
    mean = np.repeat(x.mean(1, keepdims=True), T_OUT, axis=1)
    last = np.repeat(x[:, -1:], T_OUT, axis=1)
    t = np.arange(T_IN) - (T_IN - 1) / 2
    slope = (t[None, :, None] * (x - x.mean(1, keepdims=True))).sum(1) / (t**2).sum()  # (N, F)
    h = np.arange(1, T_OUT + 1) + (T_IN - 1) / 2  # steps ahead of the window centre
    trend = x.mean(1, keepdims=True) + slope[:, None, :] * h[None, :, None]
    return {"last": last, "mean": mean, "trend": trend}


def fit(x, y, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xt, yt = torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)
    model = Forecaster(x.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt), generator=gen).to(DEVICE)
        for i in range(0, len(xt), BATCH):
            b = perm[i : i + BATCH]
            opt.zero_grad()
            nn.functional.smooth_l1_loss(model(xt[b]), yt[b]).backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(torch.from_numpy(x).to(DEVICE)).cpu().numpy()


def score(pred, y, cols) -> dict:
    """MAE / RMSE at each horizon, averaged over all features and for the key features."""
    err = pred - y  # (N, T_OUT, F)
    out = {}
    for hname, h in HORIZONS.items():
        e = err[:, h - 1]
        out[f"mae_{hname}"] = np.abs(e).mean()
        out[f"rmse_{hname}"] = np.sqrt((e**2).mean())
        for k in KEY_FEATURES:
            out[f"mae_{k}_{hname}"] = np.abs(e[:, cols.index(k)]).mean()
    return out


def load():
    d = np.load(FEATURES_DIR / "forecast_cal60.npz")
    return d["X"], d["video_id"], d["subjects"], d["y"], [str(c) for c in d["cols"]]


def run_cv():
    X, video_id, subjects, y_row, cols = load()
    folds = cv_folds(y_row, subjects)  # row-level split by subject; samples inherit it
    fold_of_subject = {s: i for i, (_, te) in enumerate(folds) for s in np.unique(subjects[te])}
    xs, ys, first = make_samples(X, video_id)
    sample_fold = np.array([fold_of_subject[s] for s in subjects[first]])
    print(f"{len(xs)} samples, {len(folds)} folds", flush=True)

    rows = {}  # model name -> list of per-seed score dicts
    for seed in SEEDS:
        preds = {"gru": np.zeros_like(ys), "last": np.zeros_like(ys), "mean": np.zeros_like(ys), "trend": np.zeros_like(ys)}
        for f in range(len(folds)):
            tr, te = sample_fold != f, sample_fold == f
            # scale from training-fold steps only (training subjects' rows)
            tr_rows = np.isin(subjects, [s for s, i in fold_of_subject.items() if i != f])
            mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
            norm = lambda a: ((a - mu) / sd).astype(np.float32)
            model = fit(norm(xs[tr]), norm(ys[tr]), seed)
            xte = norm(xs[te])
            preds["gru"][te] = predict(model, xte)
            for k, v in baselines(xte).items():
                preds[k][te] = v
            print(f"seed={seed} fold={f} done", flush=True)
        for name, p in preds.items():
            # all folds share per-fold scaling, so score in std units per fold-normalized targets
            yn = np.zeros_like(ys)
            for f in range(len(folds)):
                tr_rows = np.isin(subjects, [s for s, i in fold_of_subject.items() if i != f])
                mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
                m = sample_fold == f
                yn[m] = (ys[m] - mu) / sd
            rows.setdefault(name, []).append(score(p, yn, cols))
    # baselines are deterministic: identical across seeds, std = 0
    table = pd.DataFrame({n: pd.DataFrame(r).mean() for n, r in rows.items()}).T
    std = pd.DataFrame({n: pd.DataFrame(r).std() for n, r in rows.items()}).T
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "phase3_forecaster.csv")
    show = ["mae_10s", "mae_30s", "mae_60s", "rmse_10s", "rmse_30s", "rmse_60s"] + [
        f"mae_{k}_{h}" for k in KEY_FEATURES for h in HORIZONS
    ]
    print(table[show].round(3).to_string())
    print("gru seed std:", std.loc["gru", show].round(3).to_dict())


def save_final():
    X, video_id, *_ , cols = load()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    xs, ys, _ = make_samples(X, video_id)
    norm = lambda a: ((a - mu) / sd).astype(np.float32)
    model = fit(norm(xs), norm(ys), seed=0)
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mu, "std": sd, "cols": cols, "calib_s": 60},
               MODELS_DIR / "forecaster.pt")
    print("Saved models/forecaster.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    save_final() if ap.parse_args().final else run_cv()
