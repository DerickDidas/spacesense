import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import PRED_DIR, WINDOW
from src.data import load_channel, load_labels, spacecraft_of, true_anomalies
from src.detect import detect_windowed

OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = sys.argv[1:] or ["P-1", "E-2", "A-2", "T-1", "D-1", "M-1", "M-6", "P-10", "F-7", "C-1"]
labels = load_labels()
index = []

for chan in CHANNELS:
    d = np.load(PRED_DIR / f"{chan}.npz")
    _, test = load_channel(chan)
    seqs, e, *_ = detect_windowed(d["err"], min_drop=None)

    pred = [None] * WINDOW + [round(float(v), 4) for v in d["y_hat"]]
    err = [None] * WINDOW + [round(float(v), 4) for v in e]
    truth = [{"start": int(s), "end": int(t), "kind": k} for s, t, k in true_anomalies(labels, chan)]

    payload = {
        "channel": chan,
        "spacecraft": spacecraft_of(labels, chan),
        "window": WINDOW,
        "method": "windowed_no_prune",
        "values": [round(float(v), 4) for v in test[:, 0]],
        "pred": pred,
        "err": err,
        "flags": [{"start": int(s), "end": int(t)} for s, t in seqs],
        "truth": truth,
    }
    (OUT / f"{chan}.json").write_text(json.dumps(payload))
    index.append({"channel": chan, "spacecraft": payload["spacecraft"],
                  "n_steps": len(test), "n_truth": len(truth), "n_flags": len(seqs)})
    print(f"{chan}: {len(test)} steps, {len(truth)} true, {len(seqs)} flagged")

(OUT / "index.json").write_text(json.dumps(index, indent=1))