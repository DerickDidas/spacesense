"""Event-level scoring. All ranges are inclusive [start, end] in test-file steps."""
from dataclasses import dataclass

import numpy as np


def overlaps(a, b) -> bool:
    """True if two inclusive ranges share at least one step."""
    return a[0] <= b[1] and b[0] <= a[1]


@dataclass
class Score:
    tp: int  # labeled anomalies found
    fp: int  # flagged sequences that overlap no labeled anomaly
    fn: int  # labeled anomalies missed

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0  # no flags at all -> 0.0 by convention

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_events(pred_seqs, true_seqs) -> Score:
    """Score flagged sequences against labeled anomalies, event by event."""
    tp = sum(any(overlaps(p, t) for p in pred_seqs) for t in true_seqs)
    fp = sum(not any(overlaps(p, t) for t in true_seqs) for p in pred_seqs)
    fn = len(true_seqs) - tp
    return Score(tp, fp, fn)


def flags_to_sequences(flags, offset: int = 0):
    """Turn a True/False array into [[start, end], ...] runs.

    offset: add this to every position, e.g. the 250-step window length,
    so positions line up with the test file.
    """
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return []
    idx = np.flatnonzero(flags)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [[int(r[0]) + offset, int(r[-1]) + offset] for r in runs]


def detection_delays(pred_seqs, true_seqs):
    """For each labeled anomaly: steps from its start to the first overlapping flag.

    None means the anomaly was missed. Replay-based, so treat as optimistic.
    """
    out = []
    for t in true_seqs:
        hits = [max(p[0], t[0]) - t[0] for p in pred_seqs if overlaps(p, t)]
        out.append(min(hits) if hits else None)
    return out


def false_alarms_per_10k(fp: int, n_steps: int) -> float:
    """False alarms per 10,000 time steps of test data."""
    return fp / n_steps * 10_000

def merge_close(seqs, gap: int):
    """Merge sequences separated by at most `gap` unflagged steps."""
    out = []
    for s in sorted(seqs):
        if out and s[0] - out[-1][1] - 1 <= gap:
            out[-1][1] = max(out[-1][1], s[1])
        else:
            out.append([s[0], s[1]])
    return out