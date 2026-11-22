"""Lazy windowing and the LSTM forecaster."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .config import WINDOW


class WindowDataset(Dataset):
    """Windows are sliced on demand, never stacked into one huge array.

    Sample i: input x[i : i+window] (all columns), target x[i+window, 0].
    """

    def __init__(self, x, window: int = WINDOW, start: int = 0, stop: int | None = None):
        self.x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        self.window = window
        stop = len(x) - window if stop is None else stop
        self.indices = range(start, stop)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        return self.x[i : i + self.window], self.x[i + self.window, 0]


class LSTMForecaster(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 80, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)