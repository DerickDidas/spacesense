"""Loading channels and labels."""
import ast

import numpy as np
import pandas as pd

from .config import LABELS_CSV, TEST_DIR, TRAIN_DIR


def load_labels() -> pd.DataFrame:
    return pd.read_csv(LABELS_CSV)


def parse_classes(text) -> list:
    """'[point, contextual]' -> ['point', 'contextual'] (the CSV has no quotes)."""
    return [k.strip() for k in str(text).strip("[]").split(",")]


def true_anomalies(labels: pd.DataFrame, chan_id: str):
    """[(start, end, kind), ...] from ALL rows of this channel."""
    out = []
    for _, row in labels[labels.chan_id == chan_id].iterrows():
        seqs = ast.literal_eval(row.anomaly_sequences)
        kinds = parse_classes(row["class"])
        out += [(s, e, k) for (s, e), k in zip(seqs, kinds)]
    return out


def spacecraft_of(labels: pd.DataFrame, chan_id: str) -> str:
    return labels[labels.chan_id == chan_id].iloc[0].spacecraft


def load_channel(chan_id: str):
    """Returns (train, test) arrays of shape (time steps, inputs)."""
    return np.load(TRAIN_DIR / f"{chan_id}.npy"), np.load(TEST_DIR / f"{chan_id}.npy")