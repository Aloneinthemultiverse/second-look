"""How much does the optimal threshold depend on the cost assumptions?

This is the headline result. A fraud paper reports one threshold. A merchant
needs a different one depending on their margin, their customer lifetime value,
and what a chargeback costs them. This script shows how far that number moves.

Uses the calibrated test scores saved by model.py -- no retraining.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import config
import pipeline

SCORES = config.ARTIFACTS / "cal_test_scores.npy"


def loss_at(p, y, amount, threshold, margin, ltv, cb_fee):
    """Rupee loss under an explicit cost model (params passed, not global)."""
    blocked = p >= threshold
    fp = blocked & (y == 0)
    fn = (~blocked) & (y == 1)
    return float((amount[fp] * margin + ltv).sum() + (amount[fn] + cb_fee).sum())


def optimal(p, y, amount, margin, ltv, cb_fee, grid):
    best_t, best_l = None, float("inf")
    for t in grid:
        l = loss_at(p, y, amount, t, margin, ltv, cb_fee)
        if l < best_l:
            best_t, best_l = float(t), l
    return best_t, best_l


def describe(p, y, threshold):
    pred = (p >= threshold).astype(int)
    prec, rec, _, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return prec, rec, float(pred.mean())


def main() -> None:
    p = np.load(SCORES)
    y, amount = pipeline.load_test_arrays()
    assert len(p) == len(y), "score/label length mismatch -- rerun model.py"

    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.9999, 400)))
    base = dict(margin=config.MERCHANT_MARGIN_RATE,
                ltv=config.CUSTOMER_LTV_INR,
                cb_fee=config.CHARGEBACK_FEE_INR)

    rows = []

    # --- one-at-a-time sensitivity -----------------------------------------
    d = config.SENSITIVITY_RANGE
    for name, key in (("margin", "margin"), ("LTV", "ltv"), ("chargeback fee", "cb_fee")):
        for mult, tag in ((1 - d, f"-{d:.0%}"), (1.0, "base"), (1 + d, f"+{d:.0%}")):
            params = dict(base)
            params[key] = base[key] * mult
            t, l = optimal(p, y, amount, **params, grid=grid)
            prec, rec, blocked = describe(p, y, t)
            rows.append({"scenario": f"{name} {tag}", "threshold": t, "loss": l,
                         "precision": prec, "recall": rec, "blocked_rate": blocked})

    # --- merchant archetypes -----------------------------------------------
    # The point of this block: the same model, the same scores, three very
    # different correct answers.
    archetypes = {
        "discount retail  (low LTV, thin margin)":
            dict(margin=0.12, ltv=400.0, cb_fee=1200.0),
        "mid D2C          (base assumptions)":
            dict(base),
        "subscription SaaS (high LTV, fat margin)":
            dict(margin=0.80, ltv=15000.0, cb_fee=1200.0),
        "high-ticket electronics (large amounts)":
            dict(margin=0.08, ltv=1000.0, cb_fee=3000.0),
    }
    arche_rows = []
    for name, params in archetypes.items():
        t, l = optimal(p, y, amount, **params, grid=grid)
        prec, rec, blocked = describe(p, y, t)
        arche_rows.append({"merchant": name, "threshold": t, "loss": l,
                           "precision": prec, "recall": rec, "blocked_rate": blocked})

    # --- report ------------------------------------------------------------
    w = 74
    print("=" * w)
    print("THRESHOLD SENSITIVITY TO COST ASSUMPTIONS")
    print("=" * w)
    print(f"{'scenario':<26}{'T*':>9}{'precision':>11}{'recall':>9}{'blocked':>10}{'loss (Rs)':>14}")
    print("-" * w)
    for r in rows:
        print(f"{r['scenario']:<26}{r['threshold']:>9.4f}{r['precision']:>11.3f}"
              f"{r['recall']:>9.3f}{r['blocked_rate']:>10.2%}{r['loss']:>14,.0f}")

    print()
    print("=" * w)
    print("SAME MODEL, SAME SCORES -- DIFFERENT CORRECT THRESHOLD PER MERCHANT")
    print("=" * w)
    print(f"{'merchant':<42}{'T*':>9}{'recall':>9}{'blocked':>10}")
    print("-" * w)
    for r in arche_rows:
        print(f"{r['merchant']:<42}{r['threshold']:>9.4f}{r['recall']:>9.3f}"
              f"{r['blocked_rate']:>10.2%}")
    print("-" * w)
    ts = [r["threshold"] for r in arche_rows]
    print(f"optimal threshold ranges {min(ts):.4f} -> {max(ts):.4f} "
          f"({max(ts)/max(min(ts), 1e-9):.1f}x) across merchant economics")
    print("=" * w)

    (config.ARTIFACTS / "sensitivity.json").write_text(
        json.dumps({"one_at_a_time": rows, "archetypes": arche_rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
