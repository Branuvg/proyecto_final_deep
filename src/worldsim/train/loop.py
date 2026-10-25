"""Generic training loop: AdamW + BCE-with-logits + early stopping on validation loss (S8)."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def fit(model: nn.Module, batch_fn, n_train: int, val_fn, *, lr=1e-3, weight_decay=1e-2, batch_size=512,
        max_epochs=200, patience=15, device=None, verbose=False):
    """`batch_fn(idx)` -> (inputs tuple, target) for training rows; `val_fn()` -> validation BCE (float).
    Returns the model restored to its best validation epoch and the loss history."""
    device = device or get_device(); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(n_train)
        tot = 0.0
        for i in range(0, n_train, batch_size):
            inputs, target = batch_fn(perm[i:i + batch_size])
            loss = lossf(model(*inputs), target)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(target)
        model.eval()
        with torch.no_grad():
            v = val_fn()
        hist.append((tot / n_train, v))
        if verbose and ep % 10 == 0:
            print(f"ep {ep} train {tot/n_train:.4f} val {v:.4f}")
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist
