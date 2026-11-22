import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import SEED


def zscore_flags(train_x, test_x, k: float = 3.0):
    """Flag test steps more than k standard deviations from the TRAIN mean."""
    mu, sd = train_x.mean(), train_x.std()
    sd = sd if sd > 0 else 1e-9
    return np.abs(test_x - mu) / sd > k


def _features(x, window: int = 50):
    s = pd.Series(x)
    r = s.rolling(window, min_periods=1)
    f = pd.DataFrame(
        {"value": s, "mean": r.mean(), "std": r.std().fillna(0.0), "range": r.max() - r.min()}
    )
    return f.to_numpy()


def iforest_flags(train_x, test_x, window: int = 50, q: float = 1.0):
    """Isolation Forest on rolling features, fit on train only.

    Threshold = the q-th percentile of TRAIN scores (no labels involved),
    so about q% of normal training steps would be flagged by construction.
    """
    model = IsolationForest(n_estimators=100, random_state=SEED)
    model.fit(_features(train_x, window))
    threshold = np.percentile(model.score_samples(_features(train_x, window)), q)
    return model.score_samples(_features(test_x, window)) < threshold
