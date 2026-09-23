"""1D-CNN drowsiness classifier on raw per-frame sequences (features/dataset_cal{N}.npz).

Usage: run via the GPU conda env:
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.classifier_cnn [--calib 30 60 120] [--final N]
"""
import torch.nn as nn

from src.models.training import main


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


if __name__ == "__main__":
    main(DrowsinessCNN, "cnn")
