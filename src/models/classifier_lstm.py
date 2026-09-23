"""LSTM drowsiness classifier on raw per-frame sequences (features/dataset_cal{N}.npz).

Usage: run via the GPU conda env:
  "D:/Anaconda3/envs/btp_lstm_gpu/python.exe" -m src.models.classifier_lstm [--calib 30 60 120] [--final N]
"""
import torch.nn as nn

from src.models.training import main


class DrowsinessLSTM(nn.Module):
    def __init__(self, n_channels: int = 5, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):  # x: (batch, seq_len, channels)
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch, hidden_size)
        return self.head(h_n.squeeze(0)).squeeze(-1)  # logits


if __name__ == "__main__":
    main(DrowsinessLSTM, "lstm")
