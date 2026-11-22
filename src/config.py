"""Central settings. Change values here, not inside other files."""
import random
from pathlib import Path

import numpy as np

# --- reproducibility and model settings ---
SEED = 42          # fixed everywhere so results can be reproduced
WINDOW = 250       # LSTM input length in time steps (JPL paper setting)

# --- folders ---
ROOT = Path(__file__).resolve().parents[1]           # F:\SpaceSense
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# the extracted archive is nested: data/raw/archive/data/data/{train,test}
DATA_ROOT = RAW_DIR / "archive" / "data" / "data"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

# the original authors' pre-trained outputs: REFERENCE ONLY, never your own results
REFERENCE_DIR = DATA_ROOT / "2018-05-19_15.00.10"

# --- labels ---
LABELS_CSV = ROOT / "labeled_anomalies.csv"

# --- outputs ---
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
PRED_DIR = RESULTS_DIR / "preds"       # saved predictions and errors (git-ignored)
MODEL_DIR = RESULTS_DIR / "models"     # saved model weights (git-ignored)
METRICS_CSV = RESULTS_DIR / "metrics.csv"   # one row per run and channel


def ensure_dirs() -> None:
    """Create output folders if they do not exist yet."""
    for p in (FIG_DIR, PRED_DIR, MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Fix Python, NumPy and (if installed) PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass