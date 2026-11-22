import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PRED_DIR, WINDOW
from src.data import load_labels, true_anomalies
from src.detect import choose_threshold, smooth
from src.evaluate import flags_to_sequences

chan = sys.argv[1] if len(sys.argv) > 1 else "A-2"
err = np.load(PRED_DIR / f"{chan}.npz")["err"]
e = smooth(err)
eps, z = choose_threshold(e)
truth = [(s, t) for s, t, _ in true_anomalies(load_labels(), chan)]

inside = np.zeros(len(e), dtype=bool)
for s, t in truth:
    inside[max(s - WINDOW, 0): t - WINDOW + 1] = True

print(f"threshold {eps:.4f}")
print(f"max smoothed error inside anomaly : {e[inside].max():.4f}")
print(f"max smoothed error outside anomaly: {e[~inside].max():.4f}")
for s, t in flags_to_sequences(e >= eps):
    print(f"raw sequence test steps {s + WINDOW}-{t + WINDOW}, peak {e[s:t + 1].max():.4f}")
top = np.argsort(e)[::-1][:5]
print("5 highest smoothed errors at test steps:", sorted((top + WINDOW).tolist()))
