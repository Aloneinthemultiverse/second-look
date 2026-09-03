"""A fallback model for when the counting service is unavailable.

robustness.py found a quiet, expensive single point of failure: blanking C13
costs +Rs 18,209,609 (41% more loss) while flipping only 3.66% of decisions.
C1-C14 are address/phone count signals which in production come from one
counting service. If that service degrades, the primary model does not crash --
it keeps scoring, badly, and the volume of blocks barely moves. That is the worst
kind of outage: costly and silent.

The fix is a second model that never saw those features, plus a router.

  PRIMARY    all 77 online features. Used when the counting service answers.
  FALLBACK   77 minus C1-C14. Trained, calibrated and thresholded independently,
             so its probabilities are honest on its own feature set rather than
             being a degraded version of the primary's.

Each model has its own Platt calibrator, because a model trained without the C
family has a different score distribution and reusing the primary's calibrator
would silently mis-price every decision.

Three scenarios are measured:
  1. counting service healthy      -> primary
  2. counting service down, no fallback (today's behaviour)
  3. counting service down, fallback engaged
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import calibrate
import config
import features
import model

C_FAMILY = [f"C{i}" for i in range(1, 15)]


def per_instance_tau(amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return c_fp / (c_fp + c_fn)


def evaluate(p, y, amount):
    blocked = p >= per_instance_tau(amount)
    tp, fp, fn = blocked & (y == 1), blocked & (y == 0), (~blocked) & (y == 1)
    return {
        "blocked": int(blocked.sum()),
        "frauds_caught": int(tp.sum()),
        "recall": float(tp.sum() / max((y == 1).sum(), 1)),
        "precision": float(tp.sum() / max(blocked.sum(), 1)),
        "fpr": float(fp.sum() / max((y == 0).sum(), 1)),
        "rupee_loss": float((amount[fn] + config.CHARGEBACK_FEE_INR).sum()
                            + (amount[fp] * config.MERCHANT_MARGIN_RATE
                               + config.CUSTOMER_LTV_INR).sum()),
    }


def blank(df, cols):
    """Blank columns while preserving dtype (LightGBM rejects dtype changes)."""
    out = df.copy()
    for c in cols:
        if c not in out:
            continue
        if str(out[c].dtype) == "category":
            out[c] = pd.Categorical([None] * len(out),
                                    categories=out[c].cat.categories)
        else:
            out[c] = np.nan
    return out


def main() -> None:
    print("loading ...")
    s = features.load_splits()
    X_tr, y_tr, _ = s["train"]
    X_ca, y_ca, _ = s["calib"]
    X_te, y_te, amt_te = s["test"]

    present = [c for c in C_FAMILY if c in X_tr.columns]
    keep = [c for c in X_tr.columns if c not in present]
    print(f"counting-service features: {len(present)} ({', '.join(present)})")
    print(f"fallback feature set: {len(keep)} of {X_tr.shape[1]}")

    def fit(X_tr_, X_ca_):
        return lgb.train(model.PARAMS, lgb.Dataset(X_tr_, label=y_tr),
                         num_boost_round=model.NUM_ROUNDS,
                         valid_sets=[lgb.Dataset(X_ca_, label=y_ca)],
                         callbacks=[lgb.early_stopping(50, verbose=False)])

    print("training PRIMARY (all features) ...")
    prim = fit(X_tr, X_ca)
    cal_p = calibrate.fit_calibrator(prim.predict(X_ca), y_ca)

    print("training FALLBACK (no C-family) ...")
    fb = fit(X_tr[keep], X_ca[keep])
    # its own calibrator -- a different feature set has a different score
    # distribution, and reusing the primary's would mis-price every decision
    cal_f = calibrate.fit_calibrator(fb.predict(X_ca[keep]), y_ca)

    healthy = evaluate(cal_p(prim.predict(X_te)), y_te, amt_te)
    X_out = blank(X_te, present)
    degraded = evaluate(cal_p(prim.predict(X_out)), y_te, amt_te)
    fallback = evaluate(cal_f(fb.predict(X_te[keep])), y_te, amt_te)

    rows = {
        "1. counting service healthy (primary)": healthy,
        "2. service down, NO fallback (today)": degraded,
        "3. service down, fallback engaged": fallback,
    }

    w = 100
    print("\n" + "=" * w)
    print("FALLBACK FOR COUNTING-SERVICE OUTAGE")
    print("=" * w)
    print(f"{'scenario':<40}{'blocked':>10}{'caught':>9}{'recall':>9}"
          f"{'prec':>8}{'FPR':>8}{'rupee loss':>16}")
    print("-" * w)
    for k, v in rows.items():
        print(f"{k:<40}{v['blocked']:>10,}{v['frauds_caught']:>9,}"
              f"{v['recall']:>9.3f}{v['precision']:>8.3f}{v['fpr']:>8.2%}"
              f"{v['rupee_loss']:>16,.0f}")
    print("-" * w)

    h, d, f = (rows[k]["rupee_loss"] for k in rows)
    print(f"cost of an outage today        Rs {d - h:>+14,.0f}")
    print(f"cost of an outage WITH fallback Rs {f - h:>+14,.0f}")
    print(f"fallback saves                 Rs {d - f:>+14,.0f}  "
          f"({(d - f) / max(d - h, 1):.0%} of the outage cost recovered)")
    print()
    print("Routing rule: if any C-family feature is unavailable at inference,")
    print("score with the fallback model and its own calibrator and threshold.")
    print("Never score a partially-blank feature vector with the primary -- that")
    print("is the silent-failure path this exists to remove.")
    print("=" * w)

    (config.ARTIFACTS / "fallback.json").write_text(
        json.dumps({"c_family": present, "fallback_features": len(keep),
                    "scenarios": rows,
                    "outage_cost_today": d - h,
                    "outage_cost_with_fallback": f - h,
                    "recovered": d - f}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'fallback.json'}")


if __name__ == "__main__":
    main()
