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