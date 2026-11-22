import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PRED_DIR, WINDOW
from src.data import load_channel, load_labels, true_anomalies

chan = sys.argv[1] if len(sys.argv) > 1 else "A-2"
d = np.load(PRED_DIR / f"{chan}.npz")
y, yh, err = d["y_true"], d["y_hat"], d["err"]
train, test = load_channel(chan)

# which predicted steps fall inside a labeled anomaly (test step = WINDOW + index)
inside = np.zeros(len(y), dtype=bool)
for s, e, _ in true_anomalies(load_labels(), chan):
    lo, hi = max(s - WINDOW, 0), min(e - WINDOW + 1, len(y))
    inside[lo:hi] = True

print(f"{chan}: {len(y)} predicted steps, {inside.sum()} inside anomalies")
print(f"LSTM mean abs error        : {err.mean():.4f}")
print(f"'repeat last value' error  : {np.abs(y[1:] - y[:-1]).mean():.4f}")
print(f"'always train mean' error  : {np.abs(y - train[:, 0].mean()).mean():.4f}")
print(f"LSTM error outside anomaly : {err[~inside].mean():.4f}")
print(f"LSTM error inside anomaly  : {err[inside].mean():.4f}")
print(f"ratio inside/outside       : {err[inside].mean() / err[~inside].mean():.2f}")
print(f"99th percentile of error   : {np.percentile(err, 99):.4f}")

p = np.abs(y[1:] - y[:-1])          # 'repeat last value' error per step
pin = inside[1:]
print(f"repeat-last ratio inside/outside: {p[pin].mean() / p[~pin].mean():.2f}")