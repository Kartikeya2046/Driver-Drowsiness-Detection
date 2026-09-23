"""1D-CNN drowsiness classifier on raw per-frame sequences (features/sequences.npz).
Compares against the XGBoost window-summary-stats baseline. Same subject-grouped
5-fold CV split logic (identical seed/groups -> identical folds, for a fair
apples-to-apples comparison).

Usage: run via the GPU conda env:
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.classifier_cnn
"""
from pathlib import Path

import numpy as np
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


class DrowsinessCNN(nn.Module):
    def __init__(self, n_channels: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):  # x: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # -> (batch, channels, seq_len)
        return self.head(self.net(x)).squeeze(-1)  # logits


def train_one_fold(X_train, y_train, X_test, y_test) -> tuple[np.ndarray, np.ndarray]:
    model = DrowsinessCNN(n_channels=X_train.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if epoch == EPOCHS - 1:
            print(f"    final train loss: {total_loss / len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
        logits = model(X_test_t)
        proba = torch.sigmoid(logits).cpu().numpy()
    pred = (proba >= 0.5).astype(np.int64)
    return pred, proba


def main() -> None:
    data = np.load(FEATURES_DIR / "sequences.npz", allow_pickle=True)
    X, y, subjects, usable = data["X"], data["y"], data["subjects"], data["usable"]

    mask = usable.astype(bool)
    X, y, subjects = X[mask], y[mask], subjects[mask]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    fold_metrics = []
    all_subj_correct = {}
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, subjects)):
        assert set(subjects[train_idx]).isdisjoint(subjects[test_idx]), "subject leakage across folds"

        pred, proba = train_one_fold(X[train_idx], y[train_idx], X[test_idx], y[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        f1 = f1_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], proba)
        print(f"fold {fold}: n_test={len(test_idx)}, acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")
        fold_metrics.append({"fold": fold, "acc": acc, "f1": f1, "auc": auc})

        for subj, correct in zip(subjects[test_idx], (pred == y[test_idx])):
            all_subj_correct.setdefault(subj, []).append(correct)

    import pandas as pd

    metrics_df = pd.DataFrame(fold_metrics)
    print("\n=== 5-fold subject-grouped CV summary (1D-CNN) ===")
    print(metrics_df[["acc", "f1", "auc"]].agg(["mean", "std"]))

    per_subj_acc = {s: float(np.mean(v)) for s, v in all_subj_correct.items()}
    worst = sorted(per_subj_acc.items(), key=lambda kv: kv[1])[:5]
    print(f"\nWorst 5 subjects: {worst}")

    # final model on all usable data
    final_model = DrowsinessCNN(n_channels=X.shape[-1]).to(DEVICE)
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
    model_path = MODELS_DIR / "classifier_cnn.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"\nSaved final model to {model_path}")


if __name__ == "__main__":
    main()
