# SpaceSense

Anomaly detection on public NASA spacecraft telemetry (SMAP and MSL), with a replay dashboard.
An LSTM learns what normal sensor behavior looks like and flags deviations. I compare it against
simple baselines and evaluate it honestly, including where it fails.

> This is a replay of publicly released data with precomputed results. It is not live, not
> connected to any spacecraft, and not a NASA project.

**Demo:** [SpaceSense](https://derickdidas.github.io/spacesense/)

## What it does

- Trains one LSTM forecaster per channel on normal telemetry (plus command columns).
- Turns prediction error into alerts: EWMA smoothing, then a threshold computed per trailing window.
- Scores alerts against the expert-labeled anomalies, event by event.
- Replays the results in a browser: fleet status, incident log, per-channel detail.

## Data

NASA SMAP (Soil Moisture Active Passive) and MSL (Curiosity) telemetry with anomalies labeled by
JPL engineers from incident reports, released with Hundman et al., "Detecting Spacecraft
Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" (KDD 2018). 82 channels of data;
81 have labels (T-10 has data but no label row), covering 105 labeled anomalies. Each channel's
first column is the telemetry value; the other columns encode commands sent to the spacecraft.

## Method

| Step | Choice |
|---|---|
| Model | LSTM, 2 layers x 80 units, dropout 0.3, reads the last 250 steps, predicts the next value |
| Training | Adam 1e-3, batch 64, up to 35 epochs, early stopping (patience 5) on a chronological 70/30 validation split |
| Error | Absolute prediction error, EWMA-smoothed (span 105) |
| Threshold | Per trailing window of 2,100 steps (stride 70): mean + z x std, z in 2.5 to 10 |
| Pruning | Optional: drop an alert sequence if its peak is less than 13% above the next-highest peak |
| Scoring | Event level: an anomaly is caught if any alert overlaps it; an alert that overlaps nothing is a false alarm |

Deviations from the original paper: one-step-ahead prediction (the paper predicts 10 steps),
early-stopping patience 5 (paper: 10), validation split 0.3 (paper: 0.2). Eleven channels have
training files too short for a validation split (fewer than 50 validation windows); those are
trained for a fixed 35 epochs without validation.

## How the evaluation was done

The 81 labeled channels were split once. **10 development channels** were used to build and
compare methods, and their labels were examined along the way. The other **71 channels** were not
touched until every method and setting was fixed. All variants are reported, not only the best one.

## Results

### Development set (10 channels, 16 labeled anomalies)

These numbers are for comparing variants, not a final benchmark.

| Method | Caught | False alarms | Precision | Recall |
|---|---|---|---|---|
| Z-score (train mean/std) | 5 | 39 | 0.11 | 0.31 |
| Isolation Forest | 12 | 115 | 0.09 | 0.75 |
| LSTM, global threshold, pruning | 2 | 0 | 1.00 | 0.12 |
| LSTM, global threshold | 6 | 4 | 0.60 | 0.38 |
| LSTM, windowed threshold, pruning | 5 | 4 | 0.56 | 0.31 |
| LSTM, windowed threshold | 11 | 17 | 0.39 | 0.69 |

Isolation Forest has the highest recall (12 of 16) but raises 115 false alarms (precision 0.09);
the windowed LSTM catches 11 with 17 false alarms. With 16 anomalies, the one-event difference in
recall is not meaningful, while the false-alarm difference is. The z-score baseline misses most
contextual anomalies, because their values stay inside the range seen in training.

### Final evaluation (71 held-out channels, 89 labeled anomalies)

| Variant | Caught | Missed | False alarms | Precision | Recall | F1 | False alarms per 10k steps |
|---|---|---|---|---|---|---|---|
| LSTM, global threshold, pruning | 26 | 63 | 1 | 0.96 | 0.29 | 0.45 | 0.02 |
| LSTM, global threshold | 52 | 37 | 16 | 0.76 | 0.58 | 0.66 | 0.36 |
| LSTM, windowed threshold, pruning | 51 | 38 | 15 | 0.77 | 0.57 | 0.66 | 0.33 |
| LSTM, windowed threshold | 71 | 18 | 156 | 0.31 | 0.80 | 0.45 | 3.47 |

The baselines have not yet been run on these 71 channels, so this table has no baseline rows.

**The development-set ranking did not hold.** The windowed threshold without pruning had the best
F1 on the 10 development channels (0.50). On the held-out channels it has the highest recall and
by far the lowest precision. Two variants tie for the best F1 (0.66) and differ by one caught
anomaly. The choice between variants is an operating point (how many false alarms a team will
tolerate), not a winner. With 89 anomalies, recall has roughly ±10 points of sampling error.

Other results on the held-out channels:

- SMAP is easier than MSL (windowed threshold with pruning: F1 0.73 vs 0.47).
- Contextual anomalies are harder than point anomalies. Across the four variants, 18% to 70% of
  contextual anomalies were caught, against 36% to 86% of point anomalies. Many "point" anomalies
  in this dataset are long, which makes event-level scoring generous.
- The median delay from the start of an anomaly to the first alert is about 50 to 60 steps for
  every variant.
- The 11 short-training channels do not drive the result; leaving them out does not change the
  ranking of the variants.

## What I found

- **Repeat-last-value beats the LSTM on raw error** (0.095 vs 0.135 on A-2), but its error is lower
  inside the anomaly (the signal goes flat), while the LSTM's is higher. Raw prediction error is
  the wrong yardstick for this task.
- **A single global threshold hides weaker anomalies** behind the strongest event in a file
  (P-1, T-1, C-1) and fails when a long anomaly inflates the mean and standard deviation (D-1).
  Per-window thresholds fixed several of these, and detected A-2 about 16 steps after it began,
  compared with about 100 for the global threshold.
- **Without pruning, windowed thresholds produce many false alarms.** They are spread over many
  channels (the five worst account for 29%) and grow with file length, because each trailing
  window nominates its own largest spikes. Pruning removes most of them but also removes true
  detections when the anomaly is not clearly above the normal error.
- **Some anomalies are not detectable by this model**: on F-7, M-1 and P-10, another region has a
  higher error than the labeled anomaly.
- **Possible undertraining.** Seven of the eight channels whose training hit the 35-epoch cap with
  loss still falling (F-1, F-4, M-2, M-3, M-4, T-2, T-4) had no anomaly caught by the windowed
  variant with pruning. This is a hypothesis, not a result; I have not tested longer training.

## Limitations

- Many false alarms are short blips (1 to 10 steps) far from any labeled anomaly.
- Event-level scoring is generous to long anomalies. D-1's anomaly spans 3,258 steps and about
  380 are flagged; it still counts as caught.
- The 11 channels with very short training files get weak models; some have test errors of 0.3 to
  0.5. They are included and flagged, and results are also reported without them.
- The development set (16 anomalies) ranked the variants differently from the held-out set, so
  small-sample comparisons here should be read with caution.
- Dataset quirks: P-2 appears twice in the label file, T-10 has data but no label, and
  M-1 is about half anomalous, so "normal" is hard to define there.
- One model per channel and one seed. No confidence intervals beyond the rough estimate above.
- Not directly comparable with the paper's reported numbers: different channel handling, a
  simplified model setup and my own scoring code.

## Run it

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_one.py A-2          # train one channel
python scripts/detect_one.py A-2         # detect and score one channel
python scripts/train_all.py              # train the 71 held-out channels (hours)
python scripts/run_final.py              # score all four variants on them
python scripts/export_replay.py          # export dashboard data
python -m http.server 8000 --directory dashboard
```

The data itself is not in this repository. Download SMAP/MSL from the Kaggle mirror or the
telemanom repository, and see `src/config.py` for the expected folder layout.

## Repository layout

- `src/` data loading, model, detection, event-level scoring (with unit tests in `tests/`)
- `scripts/` training, evaluation and export scripts
- `results/` metrics tables and figures (model weights and predictions are not committed)
- `dashboard/` the replay page and the operations console

## Credits

Dataset and method: Hundman, Constantinou, Laporte, Colwell, Soderstrom, *Detecting Spacecraft
Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*, KDD 2018. Code here is my own
implementation.
