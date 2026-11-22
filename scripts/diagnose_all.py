import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PRED_DIR, WINDOW
from src.data import load_labels, true_anomalies
from src.detect import choose_threshold, smooth
from src.evaluate import flags_to_sequences

CHANNELS = ["P-1", "E-2", "A-2", "T-1", "D-1", "M-1", "M-6", "P-10", "F-7", "C-1"]
labels = load_labels()
print(f"{'chan':6} {'anomaly':14} {'peak_in':>8} {'peak_out':>8} {'thr':>7} {'raw_hit':>7}")
for chan in CHANNELS:
    e = smooth(np.load(PRED_DIR / f"{chan}.npz")["err"])
    eps, _ = choose_threshold(e)
    raw = [(s + WINDOW, t + WINDOW) for s, t in flags_to_sequences(e >= eps)]
    for s, t, kind in true_anomalies(labels, chan):
        m = np.zeros(len(e), dtype=bool)
        m[max(s - WINDOW, 0): max(t - WINDOW + 1, 0)] = True
        if not m.any():
            print(f"{chan:6} {s}-{t:<9} (anomaly inside first {WINDOW} steps)")
            continue
        hit = any(rs <= t and s <= rt for rs, rt in raw)
        print(f"{chan:6} {s}-{t:<9} {e[m].max():8.3f} {e[~m].max():8.3f} {eps:7.3f} {str(hit):>7}")