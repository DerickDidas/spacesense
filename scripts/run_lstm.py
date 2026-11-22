import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_one
from src.config import METRICS_CSV, PRED_DIR, ensure_dirs
from src.data import load_channel, load_labels, spacecraft_of, true_anomalies
from src.detect import detect, detect_windowed
from src.evaluate import false_alarms_per_10k, score_events

CHANNELS = ["P-1", "E-2", "A-2", "T-1", "D-1", "M-1", "M-6", "P-10", "F-7", "C-1"]
VARIANTS = {
    "lstm_pruned": lambda err: detect(err, min_drop=0.13),
    "lstm_no_prune": lambda err: detect(err, min_drop=None),
    "lstm_win_pruned": lambda err: detect_windowed(err, min_drop=0.13),
    "lstm_win_no_prune": lambda err: detect_windowed(err, min_drop=None),
}
COLS = ["run_id", "channel", "spacecraft", "model", "n_anomalies", "n_flagged",
        "tp", "fp", "fn", "precision", "recall", "f1", "fa_per_10k", "merge_gap"]

ensure_dirs()
labels = load_labels()
run_id = time.strftime("%Y%m%d-%H%M%S")
rows = []
for chan in CHANNELS:
    if not (PRED_DIR / f"{chan}.npz").exists():
        print(f"--- training {chan}")
        train_one.main(chan, epochs=35, patience=5)
    err = np.load(PRED_DIR / f"{chan}.npz")["err"]
    _, test = load_channel(chan)
    truth = [(s, t) for s, t, _ in true_anomalies(labels, chan)]
    for name, fn in VARIANTS.items():
        seqs, *_ = fn(err)
        sc = score_events(seqs, truth)
        rows.append({
            "run_id": run_id, "channel": chan, "spacecraft": spacecraft_of(labels, chan),
            "model": name, "n_anomalies": len(truth), "n_flagged": len(seqs),
            "tp": sc.tp, "fp": sc.fp, "fn": sc.fn,
            "precision": round(sc.precision, 3), "recall": round(sc.recall, 3),
            "f1": round(sc.f1, 3),
            "fa_per_10k": round(false_alarms_per_10k(sc.fp, len(test)), 2),
            "merge_gap": 50,
        })

new_file = not METRICS_CSV.exists()
with open(METRICS_CSV, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    if new_file:
        w.writeheader()
    w.writerows(rows)

print(f"\n{'channel':8} {'model':14} {'tp':>3} {'fp':>3} {'fn':>3}")
for r in rows:
    print(f"{r['channel']:8} {r['model']:14} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3}")
for name in VARIANTS:
    sel = [r for r in rows if r["model"] == name]
    tp, fp, fn = (sum(r[k] for r in sel) for k in ("tp", "fp", "fn"))
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    print(f"TOTAL {name}: tp {tp} fp {fp} fn {fn}  precision {p:.2f}  recall {rc:.2f}")