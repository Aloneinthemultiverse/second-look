"""Build the online feature matrix.

Only columns classified as `online` by data.audit_leakage() are used. Categorical
columns are cast to pandas `category` dtype so LightGBM can split on them
natively without one-hot expansion.
"""
from __future__ import annotations

import pandas as pd

import config
import data

TARGET = "isFraud"
AMOUNT = "TransactionAmt"


def _categoricals(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if df[c].dtype == object]


def build(df: pd.DataFrame, feature_cols: list[str], categories: dict | None = None):
    """Return (X, y, amount_inr, categories).

    `categories` is the category mapping learned on TRAIN and reused verbatim on
    calibration/test, so unseen values become NaN rather than silently shifting
    the encoding between splits.
    """
    X = df[feature_cols].copy()
    learned = {}
    for col in _categoricals(df, feature_cols):
        if categories is None:
            X[col] = X[col].astype("category")
            learned[col] = X[col].cat.categories
        else:
            X[col] = pd.Categorical(X[col], categories=categories[col])
            learned[col] = categories[col]

    y = df[TARGET].astype(int).to_numpy()
    amount_inr = (df[AMOUNT].astype(float) * config.USD_TO_INR).to_numpy()
    return X, y, amount_inr, learned


def load_splits():
    """Load, split temporally, and build feature matrices for all three slices."""
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    train_df, calib_df, test_df = data.temporal_split(raw)

    cols = audit["online"]
    X_tr, y_tr, amt_tr, cats = build(train_df, cols)
    X_ca, y_ca, amt_ca, _ = build(calib_df, cols, cats)
    X_te, y_te, amt_te, _ = build(test_df, cols, cats)

    return {
        "features": cols,
        "audit": audit,
        "train": (X_tr, y_tr, amt_tr),
        "calib": (X_ca, y_ca, amt_ca),
        "test": (X_te, y_te, amt_te),
    }
