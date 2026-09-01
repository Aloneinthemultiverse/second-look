"""Fixing the high-value fraud weakness.

The audit found the model is measurably worse on expensive fraud: missed frauds
average Rs 14,899 against Rs 11,062 for caught ones. By count we catch 44% of
fraud; by VALUE only 36.9%. Loss is denominated in rupees, so value-recall is
the metric that matters and the one we optimise against here.

Two fixes from the literature, plus the baseline:

  A BASELINE            unweighted booster, Platt calibrated. What we shipped.

  B COST-WEIGHTED + CALIBRATED   Zadrozny et al. (2003) cost-proportionate
    example weighting. Our earlier attempt weighted the training set and then
    used the raw scores, noting that weighting breaks P(fraud). That was the
    mistake: the literature is explicit that cost-sensitive learners REQUIRE
    calibration afterwards. Platt on the held-out calibration slice undoes the
    monotone distortion, so we keep the cost-aware ranking AND recover usable
    probabilities for the per-instance threshold.

  C SEGMENTED           a dedicated model for high-amount transactions.
    Research on transaction-amount distributions finds a single global model
    struggles because >90% of purchases are small, so the tail is drowned;
    routing large amounts to their own model is the standard remedy.

Every arm chooses its threshold on the CALIBRATION slice, never on test.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

import calibrate
import config
import features
import model

HIGH_VALUE_QUANTILE = 0.75      # top quartile by amount gets its own model


def metrics(p, y, amount, tau):
    """Count-recall, value-recall, and value-recall restricted to the tail."""
    blocked = p >= tau
    tp = blocked & (y == 1)
    fp = blocked & (y == 0)
    fn = (~blocked) & (y == 1)
    fraud_value = amount[y == 1].sum()

    hi = amount >= np.quantile(amount, HIGH_VALUE_QUANTILE)
    hi_fraud = (y == 1) & hi
    hi_caught = tp & hi

    loss = float((amount[fn] + config.CHARGEBACK_FEE_INR).sum()
                 + (amount[fp] * config.MERCHANT_MARGIN_RATE
                    + config.CUSTOMER_LTV_INR).sum())
    return {
        "recall_count": float(tp.sum() / max((y == 1).sum(), 1)),
        "recall_value": float(amount[tp].sum() / max(fraud_value, 1)),
        "recall_value_high": float(amount[hi_caught].sum()
                                   / max(amount[hi_fraud].sum(), 1)),
        "precision": float(tp.sum() / max(blocked.sum(), 1)),
        "block_rate": float(blocked.mean()),
        "mean_missed_fraud_inr": float(amount[fn].mean()) if fn.any() else 0.0,
        "mean_caught_fraud_inr": float(amount[tp].mean()) if tp.any() else 0.0,
        "rupee_loss": loss,
    }


def per_instance_tau(amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return c_fp / (c_fp + c_fn)


def train(X, y, X_val, y_val, weight=None):
    return lgb.train(model.PARAMS,
                     lgb.Dataset(X, label=y, weight=weight),
                     num_boost_round=model.NUM_ROUNDS,
                     valid_sets=[lgb.Dataset(X_val, label=y_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> None:
    s = features.load_splits()
    X_tr, y_tr, amt_tr = s["train"]
    X_ca, y_ca, amt_ca = s["calib"]
    X_te, y_te, amt_te = s["test"]
    tau_te = per_instance_tau(amt_te)

    results = {}

    # --- A. baseline -------------------------------------------------------
    print("A: baseline ...")
    bA = train(X_tr, y_tr, X_ca, y_ca)
    pA = calibrate.fit_calibrator(bA.predict(X_ca), y_ca)(bA.predict(X_te))
    results["A baseline"] = metrics(pA, y_te, amt_te, tau_te)

    # --- B. cost-weighted, then CALIBRATED (the fix) -----------------------
    print("B: cost-weighted + calibrated ...")
    w = np.where(y_tr == 1,
                 amt_tr + config.CHARGEBACK_FEE_INR,
                 amt_tr * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR)
    w = w / w.mean()
    bB = train(X_tr, y_tr, X_ca, y_ca, weight=w)
    # calibrate on the held-out slice -- this is the step we previously skipped
    pB = calibrate.fit_calibrator(bB.predict(X_ca), y_ca)(bB.predict(X_te))
    results["B cost-weighted+calib"] = metrics(pB, y_te, amt_te, tau_te)

    # --- C. segmented: separate model for the high-amount tail -------------
    print("C: segmented (dedicated high-amount model) ...")
    cut = float(np.quantile(amt_tr, HIGH_VALUE_QUANTILE))
    hi_tr, hi_ca, hi_te = amt_tr >= cut, amt_ca >= cut, amt_te >= cut
    print(f"   high-amount cut Rs {cut:,.0f}; "
          f"{hi_tr.mean():.0%} of train, {hi_te.mean():.0%} of test")

    b_hi = train(X_tr[hi_tr], y_tr[hi_tr], X_ca[hi_ca], y_ca[hi_ca],
                 weight=w[hi_tr])
    b_lo = train(X_tr[~hi_tr], y_tr[~hi_tr], X_ca[~hi_ca], y_ca[~hi_ca])

    # each segment calibrated on its own slice of the calibration split
    cal_hi = calibrate.fit_calibrator(b_hi.predict(X_ca[hi_ca]), y_ca[hi_ca])
    cal_lo = calibrate.fit_calibrator(b_lo.predict(X_ca[~hi_ca]), y_ca[~hi_ca])
    pC = np.empty(len(y_te))
    pC[hi_te] = cal_hi(b_hi.predict(X_te[hi_te]))
    pC[~hi_te] = cal_lo(b_lo.predict(X_te[~hi_te]))
    results["C segmented"] = metrics(pC, y_te, amt_te, tau_te)

    # --- report ------------------------------------------------------------
    wid = 104
    print("\n" + "=" * wid)
    print("FIXING THE HIGH-VALUE FRAUD WEAKNESS  (per-instance threshold, "
          "test set)")
    print("=" * wid)
    print(f"{'arm':<26}{'recall(count)':>15}{'recall(VALUE)':>15}"
          f"{'recall(top 25%)':>17}{'precision':>11}{'rupee loss':>16}")
    print("-" * wid)
    for k, v in results.items():
        print(f"{k:<26}{v['recall_count']:>15.3f}{v['recall_value']:>15.3f}"
              f"{v['recall_value_high']:>17.3f}{v['precision']:>11.3f}"
              f"{v['rupee_loss']:>16,.0f}")
    print("-" * wid)
    print(f"{'arm':<26}{'mean caught (Rs)':>20}{'mean missed (Rs)':>20}"
          f"{'gap':>14}")
    print("-" * wid)
    for k, v in results.items():
        gap = v["mean_missed_fraud_inr"] - v["mean_caught_fraud_inr"]
        print(f"{k:<26}{v['mean_caught_fraud_inr']:>20,.0f}"
              f"{v['mean_missed_fraud_inr']:>20,.0f}{gap:>+14,.0f}")
    print("-" * wid)

    base = results["A baseline"]
    best = min(results, key=lambda k: results[k]["rupee_loss"])
    bestv = max(results, key=lambda k: results[k]["recall_value"])
    print(f"baseline value-recall {base['recall_value']:.3f}, "
          f"missed-minus-caught gap Rs "
          f"{base['mean_missed_fraud_inr'] - base['mean_caught_fraud_inr']:+,.0f}")
    print(f"best by rupee loss:    {best}  "
          f"(Rs {base['rupee_loss'] - results[best]['rupee_loss']:+,.0f} vs baseline)")
    print(f"best by value-recall:  {bestv}  "
          f"({results[bestv]['recall_value'] - base['recall_value']:+.3f})")
    print("=" * wid)

    (config.ARTIFACTS / "high_value.json").write_text(
        json.dumps({"high_value_quantile": HIGH_VALUE_QUANTILE,
                    "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'high_value.json'}")


if __name__ == "__main__":
    main()
