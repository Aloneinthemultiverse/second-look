"""Probability calibration, and the reason we use Platt rather than isotonic.

Threshold selection by expected rupee cost requires actual probabilities, not
ranking scores -- multiplying a ranking score by a cost is meaningless. So a
calibration step is mandatory.

Which calibrator matters more than it looks. Measured on the held-out test slice:

    calibrator     PR-AUC   ROC-AUC      ECE   unique scores
    none           0.5095    0.8870   0.0067         111,512
    platt          0.5095    0.8870   0.0032         111,512
    isotonic       0.4881    0.8866   0.0049             167

Isotonic regression is a step function. It collapsed 111,512 distinct scores
into 167 levels; scores tied inside a level have no ordering, so ranking
metrics fall -- PR-AUC dropped 0.0214. Platt scaling is a strictly monotonic
sigmoid, so ranking is preserved exactly, and here it also calibrated better.

We found this by measuring PR-AUC on both sides of the calibrator instead of
assuming calibration is free.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DEFAULT_CALIBRATOR = "platt"
_EPS = 1e-6


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_calibrator(raw_scores, y, kind: str = DEFAULT_CALIBRATOR):
    """Return a callable mapping raw booster scores -> calibrated probabilities.

    'platt'    strictly monotonic sigmoid; ranking mathematically unchanged
    'isotonic' step function; fits the probability curve freely but ties scores
    'none'     pass-through, for measuring what calibration actually buys
    """
    if kind == "none":
        return lambda s: np.asarray(s)
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(raw_scores, y).predict
    if kind == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(
            _logit(raw_scores).reshape(-1, 1), y)
        return lambda s: lr.predict_proba(_logit(s).reshape(-1, 1))[:, 1]
    raise ValueError(f"unknown calibrator: {kind!r}")


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: mean |observed rate - predicted| across bins."""
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return float(np.mean(np.abs(frac_pos - mean_pred)))
