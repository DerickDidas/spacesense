import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIG_DIR, LABELS_CSV, TEST_DIR, TRAIN_DIR, ensure_dirs


def anomalies_for(labels: pd.DataFrame, chan_id: str):
    """Collect (start, end, type) from ALL rows of this channel."""
    out = []
    for _, row in labels[labels.chan_id == chan_id].iterrows():
        seqs = ast.literal_eval(row.anomaly_sequences)
        # class looks like "[point]" or "[contextual, contextual]" (no quotes)
        kinds = [k.strip() for k in str(row["class"]).strip("[]").split(",")]
        for (start, end), kind in zip(seqs, kinds):
            out.append((start, end, kind))
    return out


def main(chan_id: str) -> None:
    ensure_dirs()
    labels = pd.read_csv(LABELS_CSV)
    anoms = anomalies_for(labels, chan_id)
    spacecraft = labels[labels.chan_id == chan_id].iloc[0].spacecraft

    train = np.load(TRAIN_DIR / f"{chan_id}.npy")[:, 0]
    test = np.load(TEST_DIR / f"{chan_id}.npy")[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(train, linewidth=0.7)
    axes[0].set_title(f"{chan_id} train (normal), {spacecraft}")

    axes[1].plot(test, linewidth=0.7)
    for start, end, kind in anoms:
        axes[1].axvspan(start, end, color="red", alpha=0.3)
        axes[1].text(start, axes[1].get_ylim()[1], kind, fontsize=8, va="top")
    axes[1].set_title(f"{chan_id} test, {len(anoms)} anomaly(ies)")

    for ax in axes:
        ax.set_xlabel("time step")
        ax.set_ylabel("telemetry value")
    fig.tight_layout()
    out = FIG_DIR / f"{chan_id}_overview.png"
    fig.savefig(out, dpi=150)
    print("saved", out, "| anomalies:", anoms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chan_id")
    main(ap.parse_args().chan_id)