"""Recurrent neural network (GRU) decoder, built with PyTorch."""
import copy

import numpy as np
import torch
from torch import nn


class GRUDecoder(nn.Module):
    """Reads spike counts one 20 ms bin at a time and outputs hand velocity at every bin.

    A GRU (gated recurrent unit) carries a hidden state forward in time, so each output can
    depend on all earlier spiking but never on future spiking. That makes it causal, like a
    real-time decoder. Unlike the Kalman filter, it learns its own (nonlinear) dynamics.
    """

    def __init__(self, n_inputs, n_hidden=64, dropout=0.0, n_outputs=2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)       # randomly silences inputs during training
        self.gru = nn.GRU(n_inputs, n_hidden, batch_first=True)
        self.readout = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):                        # x: (trials, time bins, neurons)
        h, _ = self.gru(self.dropout(x))         # h: (trials, time bins, hidden units)
        return self.readout(h)                   # (trials, time bins, 2)


def _to_tensors(X, Y, M):
    return (torch.as_tensor(np.nan_to_num(X), dtype=torch.float32),
            torch.as_tensor(np.nan_to_num(Y), dtype=torch.float32),
            torch.as_tensor(M, dtype=torch.bool))


def _masked_mse(pred, target, mask):
    """Mean squared error over scored bins only (padding and out-of-window bins are ignored)."""
    diff = (pred - target)[mask]
    return (diff ** 2).mean() if diff.numel() else pred.sum() * 0.0


def train_gru(X, Y, M, X_val=None, Y_val=None, M_val=None, n_hidden=64, dropout=0.0,
              max_epochs=500, patience=50, fixed_epochs=None, lr=2e-3, weight_decay=1e-4,
              batch_size=16, seed=0):
    """Train a GRUDecoder.

    X : (trials, bins, neurons) standardized spike counts
    Y : (trials, bins, 2) standardized velocity
    M : (trials, bins) True for bins that count toward the loss

    With validation data, training stops once validation loss hasn't improved for `patience`
    epochs, and the weights from the best epoch are kept ("early stopping"). With
    `fixed_epochs`, it trains for exactly that many epochs; this is used for the final fit on
    all training trials, with the epoch count found by early stopping.

    Returns (model, history): history holds per-epoch "train_loss" and "val_loss", and
    "best_epoch".
    """
    assert not np.isnan(np.asarray(Y)[np.asarray(M)]).any(), "missing velocity in a scored bin"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt, Yt, Mt = _to_tensors(X, Y, M)
    has_val = X_val is not None and fixed_epochs is None   # fixed_epochs means: no early stopping
    if has_val:
        Xv, Yv, Mv = _to_tensors(X_val, Y_val, M_val)

    model = GRUDecoder(Xt.shape[2], n_hidden=n_hidden, dropout=dropout, n_outputs=Yt.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    history = {"train_loss": [], "val_loss": []}
    best_loss, best_epoch, best_state = np.inf, 0, None

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(len(Xt))
        total, count = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
            opt.zero_grad()
            loss = _masked_mse(model(Xt[idx]), Yt[idx], Mt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # keeps updates stable
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        history["train_loss"].append(total / count)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = _masked_mse(model(Xv), Yv, Mv).item()
            history["val_loss"].append(val_loss)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= patience:
                break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
    else:
        history["best_epoch"] = n_epochs
    model.eval()
    return model, history


def predict_gru(model, X):
    """Decode: (trials, bins, neurons) standardized spikes -> (trials, bins, 2) standardized velocity."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(np.nan_to_num(X), dtype=torch.float32)).numpy()
