import pandas as pd

df = pd.read_csv("results/metrics.csv")
latest = df.groupby("model")["run_id"].transform("max")
df = df[df["run_id"] == latest]          # newest run of each model
g = df.groupby("model")[["n_anomalies", "tp", "fp", "fn"]].sum()
den_p = (g.tp + g.fp).where((g.tp + g.fp) > 0)
g["precision"] = g.tp / den_p
g["recall"] = g.tp / (g.tp + g.fn)
g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall)
print(g.round(2).to_string())