"""Tests for the parts where a silent bug would invalidate every headline number.

Run:  python -m pytest test_core.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import calibrate
import config
import data
import model


# --- temporal split --------------------------------------------------------

def _fake(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": rng.permutation(np.arange(n) * 10),  # deliberately unsorted
        "TransactionAmt": rng.uniform(10, 500, n),
        "isFraud": rng.binomial(1, 0.035, n),
        "ProductCD": rng.choice(list("WCHRS"), n),
    })


def test_split_is_strictly_ordered_in_time():
    tr, ca, te = data.temporal_split(_fake())
    assert tr["TransactionDT"].max() <= ca["TransactionDT"].min()
    assert ca["TransactionDT"].max() <= te["TransactionDT"].min()


def test_split_partitions_every_row_exactly_once():
    df = _fake()
    tr, ca, te = data.temporal_split(df)
    assert len(tr) + len(ca) + len(te) == len(df)
    ids = set(tr.TransactionID) | set(ca.TransactionID) | set(te.TransactionID)
    assert len(ids) == len(df)


def test_split_does_not_leak_rows_between_slices():
    tr, ca, te = data.temporal_split(_fake())
    assert not (set(tr.TransactionID) & set(te.TransactionID))
    assert not (set(ca.TransactionID) & set(te.TransactionID))


# --- feature audit ---------------------------------------------------------

def test_audit_excludes_D_and_V_but_keeps_lookalikes():
    df = pd.DataFrame(columns=[
        "TransactionID", "TransactionDT", "isFraud",
        "D1", "D15", "V1", "V339",          # excluded
        "DeviceType", "DeviceInfo", "dist1",  # start with D/d but must be kept
        "card1", "P_emaildomain",
    ])
    a = data.audit_leakage(df)
    assert "D1" in a["excluded_drift"] and "D15" in a["excluded_drift"]
    assert "V1" in a["excluded_latency"] and "V339" in a["excluded_latency"]
    for keep in ("DeviceType", "DeviceInfo", "dist1", "card1", "P_emaildomain"):
        assert keep in a["online"], f"{keep} was wrongly excluded"
    assert "isFraud" not in a["online"], "label must never be a feature"


# --- rupee cost model ------------------------------------------------------

def test_perfect_predictions_cost_nothing():
    y = np.array([0, 1, 0, 1])
    p = y.astype(float)
    amt = np.array([100.0, 200.0, 300.0, 400.0])
    assert model.expected_loss(p, y, amt, threshold=0.5) == 0.0


def test_blocking_nothing_costs_exactly_the_fraud():
    y = np.array([0, 1])
    p = np.array([0.0, 0.0])
    amt = np.array([1000.0, 2000.0])
    expected = 2000.0 + config.CHARGEBACK_FEE_INR
    assert model.expected_loss(p, y, amt, 0.5) == pytest.approx(expected)


def test_blocking_everything_costs_exactly_the_genuine():
    y = np.array([0, 1])
    p = np.array([1.0, 1.0])
    amt = np.array([1000.0, 2000.0])
    expected = 1000.0 * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    assert model.expected_loss(p, y, amt, 0.5) == pytest.approx(expected)


def test_loss_is_monotone_in_threshold_extremes():
    """Sanity: the sweep must actually explore different decisions."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.035, 5000)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 5000), 0, 1)
    amt = rng.uniform(100, 5000, 5000)
    block_all = model.expected_loss(p, y, amt, -1.0)
    block_none = model.expected_loss(p, y, amt, 2.0)
    assert block_all != block_none


# --- calibration -----------------------------------------------------------

def test_platt_preserves_ranking_exactly():
    """The whole reason we chose Platt over isotonic."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.001, 0.999, 4000)
    y = rng.binomial(1, raw)
    p = calibrate.fit_calibrator(raw, y, "platt")(raw)
    assert np.array_equal(np.argsort(raw), np.argsort(p))


def test_isotonic_creates_ties_platt_does_not():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.001, 0.999, 4000)
    y = rng.binomial(1, raw)
    n_iso = len(np.unique(calibrate.fit_calibrator(raw, y, "isotonic")(raw)))
    n_platt = len(np.unique(calibrate.fit_calibrator(raw, y, "platt")(raw)))
    assert n_iso < n_platt
    assert n_platt == len(np.unique(raw))


def test_calibrator_outputs_are_probabilities():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.001, 0.999, 2000)
    y = rng.binomial(1, raw)
    for kind in ("none", "platt", "isotonic"):
        p = calibrate.fit_calibrator(raw, y, kind)(raw)
        assert p.min() >= 0.0 and p.max() <= 1.0, kind
