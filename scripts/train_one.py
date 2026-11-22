import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIG_DIR, MODEL_DIR, PRED_DIR, WINDOW, ensure_dirs, set_seed
from src.data import load_channel, load_labels, true_anomalies
from src.model import LSTMForecaster, WindowDataset


def predict(model, x, batch: int = 256):
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in DataLoader(WindowDataset(x), batch_size=batch, shuffle=False):
            outs.append(model(xb).numpy())
    return np.concatenate(outs)


def main(chan: str, epochs: int, patience: int):
    set_seed()
    ensure_dirs()
    train, test = load_channel(chan)
    n_windows = len(train) - WINDOW

    # chronological split; validation starts WINDOW steps later so its inputs
    # never contain a training target
    split = int(n_windows * 0.7)
    train_ds = WindowDataset(train, start=0, stop=split)
    val_ds = WindowDataset(train, start=split + WINDOW, stop=n_windows)
    print(f"{chan}: train windows {len(train_ds)}, val windows {len(val_ds)}, inputs {train.shape[1]}")

    model = LSTMForecaster(train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tl = DataLoader(train_ds, batch_size=64, shuffle=True)
    vl = DataLoader(val_ds, batch_size=256)

    best, bad, hist = float("inf"), 0, []
    ckpt = MODEL_DIR / f"{chan}.pt"
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        tr = tot / len(train_ds)

        model.eval()
        vt = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                vt += loss_fn(model(xb), yb).item() * len(xb)
        va = vt / len(val_ds)
        hist.append((tr, va))
        print(f"epoch {ep:2d}  train {tr:.5f}  val {va:.5f}")

        if va < best - 1e-6:
            best, bad = va, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(ckpt))
    y_hat = predict(model, test)
    y_true = test[WINDOW:, 0]
    err = np.abs(y_true - y_hat)
    np.savez(PRED_DIR / f"{chan}.npz", y_true=y_true, y_hat=y_hat, err=err)
    print(f"mean abs error on test: {err.mean():.4f}")

    anoms = true_anomalies(load_labels(), chan)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    ax[0].plot([h[0] for h in hist], label="train")
    ax[0].plot([h[1] for h in hist], label="validation")
    ax[0].set_title("loss per epoch")
    ax[0].legend()
    t = np.arange(WINDOW, len(test))
    ax[1].plot(t, y_true, linewidth=0.7, label="real")
    ax[1].plot(t, y_hat, linewidth=0.7, label="predicted")
    ax[1].set_title(f"{chan} test: real vs predicted")
    ax[1].legend()
    ax[2].plot(t, err, linewidth=0.7)
    ax[2].set_title("absolute error")
    for a in ax[1:]:
        for s, e, k in anoms:
            a.axvspan(s, e, color="red", alpha=0.25)
        a.set_xlabel("time step")
    fig.tight_layout()
    out = FIG_DIR / f"{chan}_lstm.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chan_id")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--patience", type=int, default=5)
    a = ap.parse_args()
    main(a.chan_id, a.epochs, a.patience)