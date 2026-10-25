"""Hand-written networks (plain nn.Module): MLP classifier and LSTM early-warning classifier."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class MLP(nn.Module):
    """Feed-forward net with Linear -> ReLU -> Dropout blocks; outputs a logit (S1 neuron, S8 dropout)."""

    def __init__(self, d_in: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class LSTMNet(nn.Module):
    """LSTM over the observed prefix (S3). With `static_mode="concat"` the static factors are concatenated
    to every time step and also to the head input; with `"none"` they are ignored (ablation).
    The last valid hidden state (packed sequences) feeds a small head that outputs a logit."""

    def __init__(self, d_seq: int, d_static: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1,
                 static_mode: str = "concat"):
        super().__init__()
        assert static_mode in ("concat", "none")
        self.static_mode = static_mode
        ds = d_static if static_mode == "concat" else 0
        self.lstm = nn.LSTM(d_seq + ds, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden + ds, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq, static, lengths):
        if self.static_mode == "concat":
            seq = torch.cat([seq, static[:, None, :].expand(-1, seq.shape[1], -1)], -1)
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        z = self.drop(h[-1])
        if self.static_mode == "concat":
            z = torch.cat([z, static], -1)
        return self.head(z).squeeze(-1)
