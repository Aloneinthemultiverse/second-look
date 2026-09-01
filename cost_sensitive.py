"""Three ways to make cost-sensitive decisions, compared on rupees.

Grounded in the literature rather than invented:

  Elkan (2001), "The Foundations of Cost-Sensitive Learning"
    With calibrated probabilities the optimal decision threshold is
        tau* = C_FP / (C_FP + C_FN)
    and his recommendation is to learn a classifier normally, then compute
    optimal decisions from its probability estimates -- rather than rebalancing
    or reweighting the training set.

  Bahnsen et al., example/instance-dependent cost-sensitive learning
    Fraud is an INSTANCE-dependent cost problem: the cost of an error depends
    on the transaction amount, so a single global threshold cannot be optimal.
    The threshold should be computed per transaction.

Arms compared:

  A  GLOBAL      one threshold for everything, swept to minimise rupee loss.
                 This is what the project used until now.
  B  PER-INSTANCE  tau*(x) from Elkan's formula with per-transaction costs.
                 No sweeping, no fitting -- it is closed form.
  C  COST-WEIGHTED TRAINING  weight training rows by the cost of getting them
                 wrong, then use a global threshold. Note this breaks
                 calibration: the model no longer estimates P(fraud), so its
                 output cannot be fed to Elkan's formula. Included to test
                 whether the empirical alternative beats the principled one.

Reported with the standard savings metric from that literature:
    savings = (cost_without_model - cost_with_model) / cost_without_model
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import calibrate
import config
import features
import model


def per_instance_threshold(amount: np.ndarray) -> np.ndarray:
    """Elkan's tau* = C_FP / (C_FP + C_FN), evaluated per transaction."""
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return c_fp / (c_fp + c_fn)


def evaluate(blocked: np.ndarray, y: np.ndarray, amount: np.ndarray) -> dict:
    fp = blocked & (y == 0)
    fn = (~blocked) & (y == 1)
    loss = float((amount[fp] * config.MERCHANT_MARGIN_RATE
                  + config.CUSTOMER_LTV_INR).sum()
                 + (amount[fn] + config.CHARGEBACK_FEE_INR).sum())
    prec, rec, _, _ = precision_recall_fscore_support(
        y, blocked.astype(int), average="binary", zero_division=0)
    return {"rupee_loss": loss, "precision": float(prec), "recall": float(rec),
            "block_rate": float(blocked.mean())}


def main() -> None:
    print("loading ...")
    s = features.load_splits()
    X_tr, y_tr, amt_tr = s["train"]
    X_ca, y_ca, amt_ca = s["calib"]
    X_te, y_te, amt_te = s["test"]

    def fit(sample_weight=None):
        ds = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
        b = lgb.train(model.PARAMS, ds, num_boost_round=model.NUM_ROUNDS,
                      valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        return b

    # --- baseline model, calibrated ---------------------------------------
    print("training baseline ...")
    b0 = fit()
    p = calibrate.fit_calibrator(b0.predict(X_ca), y_ca)(b0.predict(X_te))

    do_nothing = evaluate(np.zeros(len(y_te), bool), y_te, amt_te)["rupee_loss"]

    results = {}

    # A0. global threshold SWEPT ON TEST -- optimistically biased.
    # Kept only to quantify how large that bias is. Selecting a threshold on
    # the same data you report it on is fitting to the test set.
    rows_te = model.sweep(p, y_te, amt_te, n=400)
    best_te = min(rows_te, key=lambda r: r["rupee_loss"])
    r = evaluate(p >= best_te["threshold"], y_te, amt_te)
    r["threshold"] = best_te["threshold"]
    results["A0 global (swept on TEST)"] = r

    # A. global threshold selected on the CALIBRATION slice, applied to test.
    # This is the honest version: no test information is used to choose it.
    p_ca = calibrate.fit_calibrator(b0.predict(X_ca), y_ca)(b0.predict(X_ca))

    rows_ca = model.sweep(p_ca, y_ca, amt_ca, n=400)
    best_ca = min(rows_ca, key=lambda r: r["rupee_loss"])
    r = evaluate(p >= best_ca["threshold"], y_te, amt_te)
    r["threshold"] = best_ca["threshold"]
    results["A global (chosen on calib)"] = r

    # B. per-instance threshold (Elkan / Bahnsen) --------------------------
    tau = per_instance_threshold(amt_te)
    r = evaluate(p >= tau, y_te, amt_te)
    r["threshold"] = f"{tau.min():.3f}-{tau.max():.3f} (per txn)"
    results["B per-instance tau*"] = r

    # C. cost-weighted training --------------------------------------------
    print("training cost-weighted ...")
    w = np.where(y_tr == 1,
                 amt_tr + config.CHARGEBACK_FEE_INR,
                 amt_tr * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR)
    w = w / w.mean()
    b1 = fit(sample_weight=w)
    p1 = b1.predict(X_te)  # NOT calibrated: weighting breaks P(fraud)
    # threshold chosen on the calibration slice, same rule as arm A -- an
    # earlier version swept it on test, which gave this arm the same unfair
    # advantage we are measuring in A0
    rows1 = model.sweep(b1.predict(X_ca), y_ca, amt_ca, n=400)
    best1 = min(rows1, key=lambda r_: r_["rupee_loss"])
    r = evaluate(p1 >= best1["threshold"], y_te, amt_te)
    r["threshold"] = best1["threshold"]
    results["C cost-weighted train"] = r

    # --- report ------------------------------------------------------------
    wid = 96
    print("\n" + "=" * wid)
    print("COST-SENSITIVE DECISION STRATEGIES  (temporally held-out test set)")
    print("=" * wid)
    print(f"{'strategy':<26}{'threshold':>24}{'precision':>11}{'recall':>9}"
          f"{'block':>9}{'loss (Rs)':>15}")
    print("-" * wid)
    print(f"{'do nothing':<26}{'-':>24}{'-':>11}{'-':>9}{'0.00%':>9}"
          f"{do_nothing:>15,.0f}")
    for k, v in results.items():
        t = v["threshold"]
        t = f"{t:.4f}" if isinstance(t, float) else t
        print(f"{k:<26}{t:>24}{v['precision']:>11.3f}{v['recall']:>9.3f}"
              f"{v['block_rate']:>9.2%}{v['rupee_loss']:>15,.0f}")
    print("-" * wid)
    print(f"{'strategy':<26}{'savings vs doing nothing':>30}"
          f"{'vs global threshold':>24}")
    print("-" * wid)
    base = results["A global (chosen on calib)"]["rupee_loss"]
    for k, v in results.items():
        sav = (do_nothing - v["rupee_loss"]) / do_nothing
        delta = base - v["rupee_loss"]
        print(f"{k:<26}{sav:>29.2%} {delta:>+23,.0f}")
    print("=" * wid)

    honest = {k: v for k, v in results.items() if not k.startswith("A0")}
    winner = min(honest, key=lambda k: honest[k]["rupee_loss"])
    bias = (results["A global (chosen on calib)"]["rupee_loss"]
            - results["A0 global (swept on TEST)"]["rupee_loss"])
    print(f"optimism from sweeping the threshold on test: Rs {bias:,.0f}")
    print(f"best honest strategy: {winner}")
    if winner.startswith("B"):
        print("The closed-form per-instance threshold beats a swept global one.")
        print("No fitting, no tuning -- it falls straight out of Elkan's formula")
        print("once the probabilities are calibrated.")
    print("=" * wid)

    (config.ARTIFACTS / "cost_sensitive.json").write_text(
        json.dumps({"do_nothing": do_nothing, "results": results}, indent=2,
                   default=str), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'cost_sensitive.json'}")


if __name__ == "__main__":
    main()
