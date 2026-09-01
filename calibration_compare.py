"""Does calibration cost us ranking? Measure it instead of assuming.

Trains once, then evaluates three calibrators on the same held-out test scores:

  none      raw LightGBM output
  platt     strictly monotonic sigmoid -> ranking provably preserved
  isotonic  step function -> better probability fit, but creates ties

Reports PR-AUC (a ranking metric) and expected calibration error (a probability
metric) for each, so the trade is visible rather than assumed.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import calibrate
import config
import pipeline
from sensitivity import optimal



def main() -> None:
    print("training once ...")
    r = pipeline.train_and_score()
    raw_ca, y_ca = r["raw_calib"], r["y_calib"]
    raw_te, y_te, amount = r["raw_scores"], r["y"], r["amount"]

    results = {}
    for kind in ("none", "platt", "isotonic"):
        p = calibrate.fit_calibrator(raw_ca, y_ca, kind)(raw_te)
        grid = np.unique(np.quantile(p, np.linspace(0.50, 0.9999, 300)))
        t, loss = optimal(p, y_te, amount,
                          config.MERCHANT_MARGIN_RATE,
                          config.CUSTOMER_LTV_INR,
                          config.CHARGEBACK_FEE_INR, grid)
        results[kind] = {
            "pr_auc": float(average_precision_score(y_te, p)),
            "roc_auc": float(roc_auc_score(y_te, p)),
            "ece": calibrate.ece(y_te, p),
            "n_unique_scores": int(len(np.unique(p))),
            "threshold": t,
            "rupee_loss": loss,
        }

    w = 80
    print("\n" + "=" * w)
    print("CALIBRATOR COMPARISON  (temporally held-out test set)")
    print("=" * w)
    print(f"{'calibrator':<12}{'PR-AUC':>9}{'ROC-AUC':>10}{'ECE':>9}"
          f"{'unique scores':>15}{'rupee loss':>16}")
    print("-" * w)
    for k, v in results.items():
        print(f"{k:<12}{v['pr_auc']:>9.4f}{v['roc_auc']:>10.4f}{v['ece']:>9.4f}"
              f"{v['n_unique_scores']:>15,}{v['rupee_loss']:>16,.0f}")
    print("-" * w)

    none_, platt, iso = results["none"], results["platt"], results["isotonic"]
    print(f"isotonic vs raw:  PR-AUC {iso['pr_auc'] - none_['pr_auc']:+.4f}   "
          f"ECE {iso['ece'] - none_['ece']:+.4f}")
    print(f"platt    vs raw:  PR-AUC {platt['pr_auc'] - none_['pr_auc']:+.4f}   "
          f"ECE {platt['ece'] - none_['ece']:+.4f}")
    print()
    print("Isotonic collapses the score space into steps -- see 'unique scores'.")
    print("Ties inside a step are unordered, so ranking metrics fall. Platt is")
    print("strictly monotonic, so PR-AUC and ROC-AUC are unchanged by construction.")
    print("=" * w)

    (config.ARTIFACTS / "calibration_compare.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'calibration_compare.json'}")


if __name__ == "__main__":
    main()
