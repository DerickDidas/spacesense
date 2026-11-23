import pandas as pd

ch = pd.read_csv("results/final_by_channel.csv")
for v in ["lstm_win_no_prune", "lstm_win_pruned"]:
    d = ch[ch.variant == v].sort_values("fp", ascending=False)
    print(f"\n== {v}: channels with the most false alarms")
    print(d[["channel", "spacecraft", "short_train", "tp", "fp", "fn", "steps"]].head(10).to_string(index=False))
    if d.fp.sum():
        print(f"share of all false alarms from the top 5 channels: {d.fp.head(5).sum() / d.fp.sum():.0%}")
d = ch[ch.variant == "lstm_win_pruned"]
print("\n== windowed+pruning: channels with labeled anomalies but nothing caught")
print(d[(d.tp == 0) & (d.fn > 0)][["channel", "spacecraft", "short_train", "fn"]].to_string(index=False))