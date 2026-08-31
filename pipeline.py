"""Shared training/scoring routine, so the analysis scripts don't duplicate it.

Caches the test-set labels and amounts to disk, because reloading the 683MB CSV
for every experiment is the slowest part of the loop.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

import config
import data
import features
from model import PARAMS, NUM_ROUNDS

CACHE = config.ARTIFACTS / "test_arrays.npz"


def train_and_score(include_drift: bool = False, include_latency: bool = False):
    """Fit detector + calibrator on a chosen feature set; score the test slice.

    include_drift    -> add the D_* day-offset columns
    include_latency  -> add the V_* offline aggregates (NOT computable in 50ms)
    """
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    cols = list(audit["online"])
    if include_drift:
        cols += audit["excluded_drift"]
    if include_latency:
        cols += audit["excluded_latency"]

    train_df, calib_df, test_df = data.temporal_split(raw)
    X_tr, y_tr, _, cats = features.build(train_df, cols)
    X_ca, y_ca, _, _ = features.build(calib_df, cols, cats)
    X_te, y_te, amt_te, _ = features.build(test_df, cols, cats)

    booster = lgb.train(
        PARAMS,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=NUM_ROUNDS,
        valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(booster.predict(X_ca), y_ca)
    p_te = iso.predict(booster.predict(X_te))

    np.savez(CACHE, y=y_te, amount=amt_te)
    return {
        "n_features": len(cols),
        "scores": p_te,
        "y": y_te,
        "amount": amt_te,
        "pr_auc": float(average_precision_score(y_te, p_te)),
    }


def load_test_arrays():
    """(y, amount) for the test slice, from cache if available."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["y"], z["amount"]
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    _, _, test_df = data.temporal_split(raw)
    _, y, amt, _ = features.build(test_df, audit["online"])
    np.savez(CACHE, y=y, amount=amt)
    return y, amt
