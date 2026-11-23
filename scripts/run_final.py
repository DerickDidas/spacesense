import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_one
from src.config import PRED_DIR, RESULTS_DIR, WINDOW
from src.data import load_channel, load_labels, spacecraft_of, true_anomalies
from src.detect import detect, detect_windowed
from src.evaluate import score_events

DEV = ["P-1", "E-2", "A-2", "T-1", "D-1", "M-1", "M-6", "P-10", "F-7", "C-1"]
VARIANTS = {
    "lstm_pruned": lambda e: detect(e, min_drop=0.13),
    "lstm_no_prune": lambda e: detect(e, min_drop=None),
    "lstm_win_pruned": lambda e: detect_windowed(e, min_drop=0.13),
    "lstm_win_no_prune": lambda e: detect_windowed(e, min_drop=None),
}

labels = load_labels()
chans = sorted(set(labels["chan_id"]) - set(DEV))
missing = [c for c in chans if not (PRED_DIR / f"{c}.npz").exists()]
if missing:
    print("no predictions yet for:", missing)

ch_rows, an_rows = [], []
for chan in chans:
    if chan in missing:
        continue
    err = np.load(PRED_DIR / f"{chan}.npz")["err"]
    train, test = load_channel(chan)
    n_w = len(train) - WINDOW
    short = (n_w - (int(n_w * 0.7) + WINDOW)) < train_one.MIN_VAL
    truth = []
    for s, t, k in true_anomalies(labels, chan):
        truth.append((s, t, k if isinstance(k, str) else ",".join(k)))
    spans = [(s, t) for s, t, _ in truth]
    sc_name = spacecraft_of(labels, chan)
    for name, fn in VARIANTS.items():
        seqs = fn(err)[0]
        sc = score_events(seqs, spans)
        ch_rows.append(dict(channel=chan, spacecraft=sc_name, short_train=short, variant=name,
                            tp=sc.tp, fp=sc.fp, fn=sc.fn, steps=len(test)))
        for s, t, k in truth:
            starts = [fs for fs, fe in seqs if fs <= t and s <= fe]
            an_rows.append(dict(channel=chan, spacecraft=sc_name, short_train=short, variant=name,
                                kind=k, caught=bool(starts),
                                delay=max(0, min(starts) - s) if starts else np.nan))

ch = pd.DataFrame(ch_rows)
an = pd.DataFrame(an_rows)
ch.to_csv(RESULTS_DIR / "final_by_channel.csv", index=False)
an.to_csv(RESULTS_DIR / "final_by_anomaly.csv", index=False)


def table(d):
    g = d.groupby("variant")[["tp", "fp", "fn", "steps"]].sum()
    den = (g.tp + g.fp).where((g.tp + g.fp) > 0)
    g["precision"] = g.tp / den
    g["recall"] = g.tp / (g.tp + g.fn)
    g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall)
    g["fa_per_10k"] = g.fp / g.steps * 10000
    return g.drop(columns="steps").round(2)


groups = [("ALL final channels", ch),
          ("excluding short-training channels", ch[~ch.short_train]),
          ("SMAP only", ch[ch.spacecraft == "SMAP"]),
          ("MSL only", ch[ch.spacecraft == "MSL"])]
for title, d in groups:
    print(f"\n== {title} ({d.channel.nunique()} channels)")
    print(table(d).to_string())

an["point"] = an.kind.str.contains("point")
t = an.groupby(["variant", "point"]).caught.agg(["sum", "count"])
t["rate"] = (t["sum"] / t["count"]).round(2)
print("\n== caught anomalies by type (point=True / contextual=False)")
print(t.to_string())
print("\n== median first-alert delay in steps (caught anomalies only)")
print(an[an.caught].groupby("variant").delay.median().to_string())
print("\nshort-training channels:", sorted(ch[ch.short_train].channel.unique()))