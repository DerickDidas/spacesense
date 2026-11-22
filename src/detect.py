"""Error smoothing, dynamic threshold, pruning. Uses no labels."""
import numpy as np
import pandas as pd

from .config import WINDOW
from .evaluate import flags_to_sequences, merge_close


def smooth(err, span: int = 30):
    return pd.Series(err).ewm(span=span, adjust=False).mean().to_numpy()


def choose_threshold(e, z_values=np.arange(2.5, 10.01, 0.5)):
    """Pick mean + z*std that removes the most error mean/std per alarm raised."""
    mu, sd = e.mean(), e.std()
    best_z, best_score = z_values[-1], -np.inf
    for z in z_values:
        eps = mu + z * sd
        below = e[e < eps]
        if len(below) == 0:
            continue
        d_mu = (mu - below.mean()) / mu
        d_sd = (sd - below.std()) / sd
        above = e >= eps
        n_above = int(above.sum())
        n_seq = len(flags_to_sequences(above))
        score = (d_mu + d_sd) / (n_above + n_seq ** 2) if n_above else -np.inf
        if score > best_score:
            best_z, best_score = z, score
    return mu + best_z * sd, best_z


def prune(e, seqs, min_drop: float = 0.13):
    """Drop sequences whose peak is not clearly above the next-highest peak."""
    if not seqs:
        return []
    peaks = [e[s : t + 1].max() for s, t in seqs]
    order = np.argsort(peaks)[::-1]
    ranked = [peaks[i] for i in order]
    flagged = np.zeros(len(e), dtype=bool)
    for s, t in seqs:
        flagged[s : t + 1] = True
    ranked.append(e[~flagged].max() if (~flagged).any() else 0.0)
    keep_n = 0
    for i in range(len(ranked) - 1):
        if (ranked[i] - ranked[i + 1]) / ranked[i] >= min_drop:
            keep_n = i + 1
    keep = set(order[:keep_n].tolist())
    return [seqs[i] for i in sorted(keep)]


def detect(err, span: int = 30, min_drop: float = 0.13, merge_gap: int = 50):
    """Returns (sequences in TEST-step coordinates, smoothed error, threshold)."""
    e = smooth(err, span)
    eps, z = choose_threshold(e)
    raw = flags_to_sequences(e >= eps)
    kept = prune(e, raw, min_drop) if min_drop is not None else raw
    kept = [(s + WINDOW, t + WINDOW) for s, t in kept]
    return merge_close(kept, merge_gap), e, eps, z, len(raw)