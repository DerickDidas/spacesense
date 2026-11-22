import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FIG_DIR, PRED_DIR, WINDOW
from src.data import load_channel, load_labels, true_anomalies
from src.detect import detect
from src.evaluate import detection_delays, false_alarms_per_10k, score_events

chan = sys.argv[1] if len(sys.argv) > 1 else "A-2"
d = np.load(PRED_DIR / f"{chan}.npz")
_, test = load_channel(chan)
truth = [(s, e) for s, e, _ in true_anomalies(load_labels(), chan)]

seqs, e, eps, z, n_raw = detect(d["err"])
sc = score_events(seqs, truth)
print(f"{chan}: threshold {eps:.4f} (mean + {z} std)")
print(f"sequences before pruning: {n_raw}, after pruning/merging: {len(seqs)}")
print("flagged:", seqs)
print("true   :", truth)
print(f"tp {sc.tp}  fp {sc.fp}  fn {sc.fn}  precision {sc.precision:.2f}  recall {sc.recall:.2f}  f1 {sc.f1:.2f}")
print(f"false alarms per 10k steps: {false_alarms_per_10k(sc.fp, len(test)):.2f}")
print("detection delays:", detection_delays(seqs, truth))

t = np.arange(WINDOW, len(test))
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(t, d["y_true"], linewidth=0.7)
ax[0].set_title(f"{chan} test telemetry")
ax[1].plot(t, e, linewidth=0.8, label="smoothed error")
ax[1].axhline(eps, color="k", linestyle="--", label="threshold")
for a in ax:
    for s, en in truth:
        a.axvspan(s, en, color="red", alpha=0.25)
    for s, en in seqs:
        a.axvspan(s, en, ymin=0, ymax=0.08, color="blue", alpha=0.8)
ax[1].legend()
ax[1].set_xlabel("time step (red = true anomaly, blue bar = flagged)")
fig.tight_layout()
out = FIG_DIR / f"{chan}_detect.png"
fig.savefig(out, dpi=150)
print("saved", out)