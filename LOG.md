# SpaceSense project log

Short notes after each phase: what worked, what broke, what I decided, what is next.

## Phase 0: Setup (2026-11-22)

**Done**
- Created project folder `F:\SpaceSense` with `.venv`, `src/`, `scripts/`, `tests/`, `notebooks/`, `results/figures/`, `data/raw/`.
- Python 3.14, with numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, torch 2.15.0, matplotlib 3.11.2, plotly 7.1.0, streamlit 1.67.0. Versions saved in `requirements.txt`.
- `.gitignore` excludes `data/`, `.venv/`, `*.zip`, `labeled_anomalies.csv`, `results/models/`, `results/preds/`.
- `src/config.py` holds the seed (42), window length (250), and all data and result paths.
- Local git repository only; not pushed to GitHub yet (decided to build first).

**Data**
- Source: Kaggle copy of the NASA SMAP/MSL telemetry (`archive.zip`), extracted to `data/raw/archive/`.
- Layout is nested: `data/raw/archive/data/data/{train,test}/`, one `.npy` file per channel (82 channels).
- Labels file: `labeled_anomalies.csv` (columns: chan_id, spacecraft, anomaly_sequences, class, num_values).
- The archive also contains the original authors' pre-trained outputs (`2018-05-19_15.00.10/`: Keras models, predictions, smoothed errors). **Reference only**; my results will come from my own models.

**Decisions**
- Use my own implementation; do not reuse the authors' models or predictions as results.
- No GitHub push until the core pipeline works.

## Phase 1: Explore the data (2026-11-22)

**Labels file checks**
- 82 rows: 55 SMAP, 27 MSL. Total anomalies: 69 SMAP + 36 MSL = 105, matching the paper.
- `P-2` appears twice (rows 17 and 51, both SMAP, both "point"). Rule: always collect anomalies from ALL rows of a channel, never only the first.
- `T-10` has a data file but no row in the labels file. Check: compare `num_values` of the two P-2 rows with the test lengths of P-2 and T-10. (Result: to fill in.)
- The `class` column is stored unquoted (`[point]`), so it cannot be parsed with `ast.literal_eval`. Parse it by stripping brackets and splitting on commas.

**Channel plots** (`scripts/plot_channel.py`, saved in `results/figures/`)
- `E-2` (SMAP, point): the normal cycle is a sharp rise then decay. In the labeled region (steps 5598 to 6995, ~1,400 steps) the signal climbs to a sustained plateau near 1.0 (about double a normal test peak), then runs smaller, irregular cycles with a floor near -1. "Point" does not mean a single time step.
- `A-2` (SMAP, contextual): a regular oscillation throughout. In the labeled region (steps 4450 to 4560, ~110 steps) the oscillation stops and the line sits flat near -0.4. The value is normal; the behavior is not.
- `M-1` (MSL, contextual): command-like square pulses. In the labeled region (steps 1110 to 2250) the pulses continue but dip only to ~0.5 instead of ~-1, with a flat stretch near 0.35. The labeled region covers about half the test file (~2,270 steps), which is a concrete example of high anomaly density.

**Observations to use later**
- In all three channels the anomalous values lie inside the range seen in training, so a plain z-score is likely to miss them. Prediction to test in Phase 3.
- `E-2` train peaks (~0.9) are higher than normal test peaks (~0.5). Possible cause of false alarms; mention in limitations.
- `M-1` looks command-driven, so the command columns may matter. Experiment planned: LSTM with and without command columns.
- Labeled regions are long and fuzzy, which supports event-level scoring.

**Next**
- Phase 2: write event-level scoring (`src/evaluate.py`) and tests (`tests/test_scoring.py`) before any model.


## Phase 2: Scoring (2026-11-22)
- Wrote src/evaluate.py: event-level scoring, flags_to_sequences with offset, detection delays, false alarms per 10k steps.
- 12 tests in tests/test_scoring.py pass.
- Convention: precision = 0.0 when nothing is flagged.

## Phase 3: Baselines (2026-11-22)
- Settings: z-score k=3 on train mean/std; Isolation Forest on rolling features (window 50), threshold = 1st percentile of train scores; merge gap 50 for both. 10 channels, 16 anomalies.
- z-score: found 5/16, 39 false alarms. Isolation Forest: found 12/16, 115 false alarms (precision ~0.09).
- Prediction confirmed: z-score flagged nothing on E-2, A-2, M-1 (anomalous values lie inside the training range).
- Isolation Forest flags too many sequences (up to ~41 false alarms per 10k steps), and missed M-1's long anomaly while flagging the first half of the test file.
- Caveat: 16 anomalies is too few for stable percentages; rerun on all channels later.
- Target for the LSTM: keep high recall with far fewer false alarms, especially on contextual anomalies.


## Pause point (2026-11-22): final-set training not started yet

State:
- Dev set = 10 channels (P-1, E-2, A-2, T-1, D-1, M-1, M-6, P-10, F-7, C-1). Labels already looked at, so they are NOT a clean test set.
- Final set = the other channels in labeled_anomalies.csv (about 72). Untouched. Nothing tuned on them.
- Dev totals (event level, 16 anomalies): global+pruning tp2/fp0/fn14; global no pruning 6/4/10; windowed+pruning 5/4/11; windowed no pruning 11/17/5 (precision 0.39, recall 0.69).
- Windowed = paper-style: trailing 2100-step window, stride 70, EWMA span 105, threshold per window. Global = one threshold per file, span 30.
- Dev misses: weaker anomaly beside a stronger one (P-1, T-1, C-1), threshold inflated by a long anomaly (D-1), another region with higher error (F-7, M-1, P-10).
- False alarms concentrated in A-2 (6) and P-1 (4). Open question: some may be smoothing lag just after an anomaly (check with scripts/show_flags.py).

To resume:
1. Settle the scoring rule on the dev channels first (tolerance for flags next to an anomaly). Do not touch final-set results before that.
2. Run `python scripts/train_all.py` (resumable, skips channels with results/preds/*.npz, about 8-10 h; sleep off, charger on). Errors go to results/train_errors.txt.
3. Score all four variants on the final set, report all four, say which was best on dev. No tuning after seeing the final set.

Deviations from the original (README): predicts 1 step ahead (original 10), patience 5 (original 10), validation split 0.3 (original 0.2).


## Phase 5 closing + Phase 8 v1
- Scoring rule kept as strict overlap. Of the 17 windowed false alarms only 1 is near an anomaly (P-1, 54 steps after the end); the rest are 1-10 step blips far from any anomaly. No tolerance rule, no minimum-length filter (would be tuning on labels).
- A-2 windowed detection fires 16 steps after the anomaly starts (global detector: about 100).
- README limits: event-level scoring is generous for long anomalies (D-1: 3,258 steps, about 380 covered, counts as caught); P-1 misses 2 of 3 anomalies with nothing flagged near them.
- Dashboard v1: static page, dashboard/index.html + dashboard/data/*.json from scripts/export_replay.py. Detector shown: windowed, no pruning.


## Phase 6 setup: final-set training
- 71 final channels (all labeled channels minus the 10 dev ones). train_all.py took 145 min for 41 channels (the earlier 8-10 h estimate was too high).
- 9 channels crashed: no validation windows left (training file too short for the 250-step validation gap). Fix: fewer than 50 validation windows (MIN_VAL) = train on all windows for fixed 35 epochs, no validation. Applied to T-12 and T-13 too (19 validation windows) and retrained. Decided from the crash, before any final-set detection result.
- Expectations to check (not conclusions): train/test shift on P-14, P-15; relative thresholds on near-constant channels (G-2, R-1, P-4, T-5, G-3) may raise false alarms; S-1 did not converge; F-1, F-4, M-2, M-3, T-2, T-4 hit the 35-epoch cap while still improving.
- Pre-declared: report all four variants, all channels and excluding short-training channels, by spacecraft and by anomaly type. No tuning after seeing the final numbers.


## Phase 6: final evaluation (71 channels, 89 labeled anomalies, untouched until now)
- Global pruned 26 tp / 1 fp / 63 fn; global 52/16/37; windowed pruned 51/15/38; windowed 71/156/18.
- Dev-set ranking did not hold: windowed/no pruning was best F1 on dev (0.50) but on final it has the best recall (0.80) and worst precision (0.31, 3.47 false alarms per 10k steps). Global and windowed+pruning tie at F1 0.66. All four reported; no tuning after seeing results.
- SMAP easier than MSL (windowed pruned F1 0.73 vs 0.47). Contextual anomalies harder than point (caught 0.42-0.70 vs 0.68-0.86). Median first-alert delay about 50 steps.
- Excluding the 11 short-training channels does not change the ranking.
- Not yet done: baselines (z-score, Isolation Forest) on the final set; failure analysis of false-alarm channels (scripts/top_fp.py).