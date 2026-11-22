# SpaceSense

Anomaly detection on public NASA spacecraft telemetry (SMAP and MSL), with a replay dashboard.
An LSTM learns what normal sensor behavior looks like and flags deviations. I compare it against
simple baselines and evaluate it honestly, including where it fails.

> This is a replay of publicly released data with precomputed results. It is not live, not
> connected to any spacecraft, and not a NASA project.

**Demo:** `<DEMO_URL>` (replay view and operations console)

## What it does

- Trains one LSTM forecaster per channel on normal telemetry (plus command columns).
- Turns prediction error into alerts: EWMA smoothing, then a threshold computed per trailing window.
- Scores alerts against the expert-labeled anomalies, event by event.
- Replays the results in a browser: fleet status, incident log, per-channel detail.

## Data

NASA SMAP (Soil Moisture Active Passive) and MSL (Curiosity) telemetry with anomalies labeled by
JPL engineers from incident reports, released with Hundman et al., "Detecting Spacecraft
Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" (KDD 2018). 82 channels,
105 labeled anomalies. Each channel's first column is the telemetry value; the other columns
encode commands sent to the spacecraft.

## Method

| Step | Choice |
|---|---|
| Model | LSTM, 2 layers x 80 units, dropout 0.3, reads the last 250 steps, predicts the next value |
| Training | Adam 1e-3, batch 64, early stopping (patience 5), chronological 70/30 validation split |
| Error | Absolute prediction error, EWMA-smoothed (span 105) |
| Threshold | Per trailing window of 2,100 steps (stride 70): mean + z x std, z in 2.5 to 10 |
| Scoring | Event level: an anomaly is caught if any alert overlaps it; an alert that overlaps nothing is a false alarm |

Deviations from the original paper: one-step-ahead prediction (the paper predicts 10 steps),
early-stopping patience 5 (paper: 10), validation split 0.3 (paper: 0.2).

## Results
Isolation Forest has the highest recall (12 of 16) but raises 115 false alarms (precision 0.09);
the windowed LSTM catches 11 with 17 false alarms. With 16 anomalies, the one-event difference in
recall is not meaningful, while the false-alarm difference is.

### Development set (10 channels, 16 labeled anomalies)

The labels of these channels were examined while building the method, so these numbers are for
comparing variants, not a final benchmark.

| Method | Caught | False alarms | Precision | Recall |
|---|---|---|---|---|
| Z-score (train mean/std) | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Isolation Forest | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| LSTM, global threshold, pruning | 2 | 0 | 1.00 | 0.12 |
| LSTM, global threshold | 6 | 4 | 0.60 | 0.38 |
| LSTM, windowed threshold, pruning | 5 | 4 | 0.56 | 0.31 |
| LSTM, windowed threshold | 11 | 17 | 0.39 | 0.69 |

The windowed threshold finds more anomalies and raises more false alarms. Pruning removes
false alarms and also most true detections in this setup.

### Final evaluation (remaining channels, untouched during development)

*Pending.* All variants will be reported, whichever one looks best.

## What I found

- **Repeat-last-value beats the LSTM on raw error** (0.095 vs 0.135 on A-2), but its error is lower
  inside the anomaly (the signal goes flat), while the LSTM's is higher. Raw prediction error is
  the wrong yardstick for this task.
- **A single global threshold hides weaker anomalies** behind the strongest event in a file
  (P-1, T-1, C-1) and fails when a long anomaly inflates the mean and standard deviation (D-1).
  Per-window thresholds fixed several of these, and detected A-2 about 16 steps after it began,
  compared with about 100 for the global threshold.
- **Some anomalies are not detectable by this model** (F-7, M-1, P-10: another region has a
  higher error than the labeled anomaly).

## Limitations

- Many false alarms are short blips (1 to 10 steps) far from any labeled anomaly.
- Event-level scoring is generous to long anomalies. D-1's anomaly spans 3,258 steps and about
  380 are flagged; it still counts as caught.
- Dataset quirks: P-2 appears twice in the label file, T-10 has data but no label, and
  M-1 is about half anomalous, so "normal" is hard to define there.
- Results come from one model per channel and one seed.

## Run it

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_one.py A-2
python scripts/detect_one.py A-2
python scripts/export_replay.py
python -m http.server 8000 --directory dashboard
```

The data itself is not in this repository. Download SMAP/MSL from the Kaggle mirror or the
telemanom repository, and see `src/config.py` for the expected folder layout.

## Credits

Dataset and method: Hundman, Constantinou, Laporte, Colwell, Soderstrom, *Detecting Spacecraft
Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*, KDD 2018. Code here is my own
implementation.