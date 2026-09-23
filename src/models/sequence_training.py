"""Shared subject-grouped 5-fold CV training/eval harness for sequence models
(CNN/GRU/LSTM) on features/sequences.npz. One code path so every architecture
is compared under identical splits/epochs/metrics.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "features"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3


def train_one_fold(model_ctor, X_train, y_train, X_test, y_test) -> tuple[np.ndarray, np.ndarray]:
    model = model_ctor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if epoch == EPOCHS - 1:
            print(f"    final train loss: {total_loss / len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().to(DEVICE))
        proba = torch.sigmoid(logits).cpu().numpy()
    pred = (proba >= 0.5).astype(np.int64)
    return pred, proba


def run_cv(model_ctor, model_name: str) -> None:
    data = np.load(FEATURES_DIR / "sequences.npz", allow_pickle=True)
    X, y, subjects, usable = data["X"], data["y"], data["subjects"], data["usable"]
    mask = usable.astype(bool)
    X, y, subjects = X[mask], y[mask], subjects[mask]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    fold_metrics = []
    subj_correct: dict[str, list[bool]] = {}
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, subjects)):
        assert set(subjects[train_idx]).isdisjoint(subjects[test_idx]), "subject leakage across folds"

        pred, proba = train_one_fold(model_ctor, X[train_idx], y[train_idx], X[test_idx], y[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        f1 = f1_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], proba)
        print(f"fold {fold}: n_test={len(test_idx)}, acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")
        fold_metrics.append({"fold": fold, "acc": acc, "f1": f1, "auc": auc})

        for subj, correct in zip(subjects[test_idx], (pred == y[test_idx])):
            subj_correct.setdefault(subj, []).append(bool(correct))

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n=== 5-fold subject-grouped CV summary ({model_name}) ===")
    print(metrics_df[["acc", "f1", "auc"]].agg(["mean", "std"]))

    per_subj_acc = {s: float(np.mean(v)) for s, v in subj_correct.items()}
    worst = sorted(per_subj_acc.items(), key=lambda kv: kv[1])[:5]
    print(f"\nWorst 5 subjects: {worst}")

    # final model on all usable data
    final_model = model_ctor().to(DEVICE)
    opt = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    final_model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(final_model(xb), yb)
            loss.backward()
            opt.step()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"classifier_{model_name}.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"\nSaved final model to {model_path}")
