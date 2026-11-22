import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import iforest_flags, zscore_flags
from src.config import METRICS_CSV, ensure_dirs, set_seed
from src.data import load_channel, load_labels, spacecraft_of, true_anomalies
from src.evaluate import false_alarms_per_10k, flags_to_sequences, merge_close, score_events

CHANNELS = ["P-1", "E-2", "A-2", "T-1", "D-1", "M-1", "M-6", "P-10", "F-7", "C-1"]
MERGE_GAP = 50  # fixed, same for every baseline, never tuned on labels
MODELS = {"zscore": zscore_flags, "iforest": iforest_flags}


def main():
    set_seed()
    ensure_dirs()
    labels = load_labels()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for chan in CHANNELS:
        train, test = load_channel(chan)
        tr, te = train[:, 0], test[:, 0]
        true_seqs = [[s, e] for s, e, _ in true_anomalies(labels, chan)]
        for name, fn in MODELS.items():
            seqs = merge_close(flags_to_sequences(fn(tr, te)), MERGE_GAP)
            sc = score_events(seqs, true_seqs)
            rows.append(
                dict(
                    run_id=run_id, channel=chan, spacecraft=spacecraft_of(labels, chan),
                    model=name, n_anomalies=len(true_seqs), n_flagged=len(seqs),
                    tp=sc.tp, fp=sc.fp, fn=sc.fn,
                    precision=round(sc.precision, 3), recall=round(sc.recall, 3),
                    f1=round(sc.f1, 3),
                    fa_per_10k=round(false_alarms_per_10k(sc.fp, len(te)), 2),
                    merge_gap=MERGE_GAP,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_CSV, mode="a", header=not METRICS_CSV.exists(), index=False)
    print(df.to_string(index=False))

    tot = df.groupby(["model", "spacecraft"])[["tp", "fp", "fn"]].sum()
    tot["precision"] = (tot.tp / (tot.tp + tot.fp)).round(3)
    tot["recall"] = (tot.tp / (tot.tp + tot.fn)).round(3)
    print("\nTotals:\n", tot)


if __name__ == "__main__":
    main()