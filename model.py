"""Train the detector, calibrate it, and find the rupee-optimal threshold.

Three stages, deliberately separated:

  1. DETECTOR   LightGBM -> a ranking score. Not a probability.
  2. CALIBRATOR Platt scaling fit on the temporal calibration slice
                -> an actual probability. Without this, multiplying the score
                by a rupee cost is arithmetically meaningless. Platt rather
                than isotonic: see calibrate.py for the measured reason.
  3. POLICY     threshold chosen to minimise expected rupee loss, compared
                against the F1-optimal threshold and against doing nothing.

The gap between the F1-optimal and rupee-optimal policies is the headline result.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

import calibrate
import config
import features

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": config.SEED,
}
NUM_ROUNDS = 600


# --- expected rupee loss ---------------------------------------------------

def expected_loss(p: np.ndarray, y: np.ndarray, amount: np.ndarray,
                  threshold: float) -> float:
    """Total rupees lost by a block-above-threshold policy on this slice.

    Blocking a genuine transaction costs margin + LTV.
    Allowing a fraudulent one costs the amount + chargeback fee.
    Correct decisions cost nothing.
    """
    blocked = p >= threshold
    false_positive = blocked & (y == 0)
    false_negative = (~blocked) & (y == 1)

    fp_cost = (amount[false_positive] * config.MERCHANT_MARGIN_RATE
               + config.CUSTOMER_LTV_INR).sum()
    fn_cost = (amount[false_negative] + config.CHARGEBACK_FEE_INR).sum()
    return float(fp_cost + fn_cost)


def sweep(p: np.ndarray, y: np.ndarray, amount: np.ndarray, n: int = 200):
    """Evaluate rupee loss and F1 across candidate thresholds."""
    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.9999, n)))
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)
        rows.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "rupee_loss": expected_loss(p, y, amount, t),
        })
    return rows


def main() -> None:
    print("loading and splitting ...")
    s = features.load_splits()
    X_tr, y_tr, _ = s["train"]
    X_ca, y_ca, amt_ca = s["calib"]
    X_te, y_te, amt_te = s["test"]
    print(f"  {len(s['features'])} online features")

    # 1. detector
    print("training detector ...")
    booster = lgb.train(
        PARAMS,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=NUM_ROUNDS,
        valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    raw_ca = booster.predict(X_ca)
    raw_te = booster.predict(X_te)

    # 2. calibrator -- fit on the calibration slice only, never on test
    print(f"calibrating ({calibrate.DEFAULT_CALIBRATOR} on the calibration slice) ...")
    cal = calibrate.fit_calibrator(raw_ca, y_ca)
    cal_te = cal(raw_te)

    # reliability measured on TEST -- fitting and validating on the same slice
    # would prove nothing
    ece_raw = calibrate.ece(y_te, raw_te)
    ece_cal = calibrate.ece(y_te, cal_te)

    # 3. policy comparison
    #
    # THRESHOLDS ARE SELECTED ON THE CALIBRATION SLICE, NEVER ON TEST.
    # An earlier version of this file swept on the test set and reported the
    # result on that same test set -- fitting to the evaluation data. That was
    # worth Rs 713,146 of savings that were never real (README finding #2). The
    # biased figure is still computed below, but only to quantify the bias, and
    # it is never the headline.
    print("sweeping thresholds on the CALIBRATION slice ...")
    cal_ca = cal(raw_ca)
    rows_ca = sweep(cal_ca, y_ca, amt_ca)
    best_f1_ca = max(rows_ca, key=lambda r: r["f1"])
    best_rupee_ca = min(rows_ca, key=lambda r: r["rupee_loss"])

    # honest: thresholds chosen off-test, every metric then measured ON TEST
    def on_test(sel):
        t = sel["threshold"]
        pred = (cal_te >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="binary", zero_division=0)
        return {"threshold": t, "precision": float(prec), "recall": float(rec),
                "f1": float(f1),
                "rupee_loss": expected_loss(cal_te, y_te, amt_te, t)}

    best_f1 = on_test(best_f1_ca)
    best_rupee = on_test(best_rupee_ca)

    # kept only to measure the optimism of the bug we removed
    rows_te = sweep(cal_te, y_te, amt_te)
    biased = min(rows_te, key=lambda r: r["rupee_loss"])

    do_nothing = expected_loss(cal_te, y_te, amt_te, threshold=2.0)  # never block

    report = {
        "n_features": len(s["features"]),
        "test_rows": int(len(y_te)),
        "test_fraud_rate": float(y_te.mean()),
        "pr_auc_raw": float(average_precision_score(y_te, raw_te)),
        "pr_auc_calibrated": float(average_precision_score(y_te, cal_te)),
        "calibration_error_raw": ece_raw,
        "calibration_error_calibrated": ece_cal,
        "policy_do_nothing": {"rupee_loss": do_nothing},
        "policy_f1_optimal": best_f1,
        "policy_rupee_optimal": best_rupee,
        "saving_vs_f1_optimal": best_f1["rupee_loss"] - best_rupee["rupee_loss"],
        "threshold_selected_on": "calibration slice (never test)",
        "policy_rupee_optimal_BIASED_swept_on_test": biased,
        "optimism_from_sweeping_on_test": best_rupee["rupee_loss"] - biased["rupee_loss"],
    }
    (config.ARTIFACTS / "results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    booster.save_model(str(config.ARTIFACTS / "detector.txt"))
    np.save(config.ARTIFACTS / "cal_test_scores.npy", cal_te)

    # --- report ---
    print("\n" + "=" * 66)
    print("RESULTS  (temporally held-out test set)")
    print("=" * 66)
    print(f"rows {len(y_te):,}   fraud {y_te.mean():.4%}   features {len(s['features'])}")
    print(f"PR-AUC                     {report['pr_auc_raw']:.4f}")
    print(f"calibration error  raw     {ece_raw:.4f}")
    print(f"calibration error  after   {ece_cal:.4f}")
    print("-" * 66)
    print(f"{'policy':<22}{'threshold':>11}{'precision':>11}{'recall':>9}{'rupee loss':>14}")
    print(f"{'do nothing':<22}{'-':>11}{'-':>11}{'-':>9}{do_nothing:>14,.0f}")
    for name, r in (("F1-optimal", best_f1), ("rupee-optimal", best_rupee)):
        print(f"{name:<22}{r['threshold']:>11.4f}{r['precision']:>11.3f}"
              f"{r['recall']:>9.3f}{r['rupee_loss']:>14,.0f}")
    print("-" * 66)
    print(f"thresholds selected on the calibration slice, never on test")
    print(f"optimism if we HAD swept on test: "
          f"Rs {report['optimism_from_sweeping_on_test']:,.0f} "
          f"(the bug removed in README finding #2)")
    print("=" * 66)


if __name__ == "__main__":
    main()
