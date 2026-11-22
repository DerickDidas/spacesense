import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PRED_DIR
from src.data import load_labels, true_anomalies
from src.detect import detect_windowed

chan = sys.argv[1]
err = np.load(PRED_DIR / f"{chan}.npz")["err"]
truth = [(s, t) for s, t, _ in true_anomalies(load_labels(), chan)]
seqs = detect_windowed(err, min_drop=None)[0]

print("true anomalies:", truth)
for s, t in seqs:
    gaps = [0 if (s <= te and ts <= t) else max(ts - t, s - te) for ts, te in truth]
    d = min(gaps)
    note = "overlaps" if d == 0 else f"{d} steps from nearest anomaly"
    print(f"flag {s}-{t} (len {t - s + 1}): {note}")