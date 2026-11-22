import sys
import pandas as pd

df = pd.read_csv("results/metrics.csv")
df = df[df["run_id"] == df["run_id"].iloc[-1]]
model = sys.argv[1] if len(sys.argv) > 1 else "lstm_win_no_prune"
cols = ["channel", "n_anomalies", "n_flagged", "tp", "fp", "fn", "fa_per_10k"]
print(df[df["model"] == model][cols].to_string(index=False))