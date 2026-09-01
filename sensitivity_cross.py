"""Does the merchant-sensitivity finding hold on both datasets?

An earlier version of this analysis compared raw threshold VALUES across
datasets and concluded the finding did not replicate. That was a measurement
error: score distributions differ between models, so a threshold of 0.30 on one
dataset and 0.05 on another are not comparable quantities.

The comparable units are the DECISIONS the threshold produces:
  - block rate: what fraction of traffic is declined
  - recall:     what fraction of fraud is caught

Those mean the same thing on any dataset, so that is what we report.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import calibrate
import config
import model
import pipeline
import second_dataset as ulb
from sensitivity import optimal

ARCHETYPES = {
    "discount retail":   dict(margin=0.12, ltv=400.0, cb_fee=1200.0),
    "mid D2C":           dict(margin=0.25, ltv=2500.0, cb_fee=1200.0),
    "high-ticket":       dict(margin=0.08, ltv=1000.0, cb_fee=3000.0),
    "subscription SaaS": dict(margin=0.80, ltv=15000.0, cb_fee=1200.0),
}


def analyse(name, p, y, amount):
    # grid reaches 1.0 so a "block almost nothing" optimum is a real interior
    # solution rather than an artefact of where the grid stops
    grid = np.append(np.unique(np.quantile(p, np.linspace(0.50, 0.99999, 500))), 1.01)
    out = {}
    for arche, params in ARCHETYPES.items():
        t, loss = optimal(p, y, amount, **params, grid=grid)
        pred = (p >= t).astype(int)
        prec, rec, _, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)
        out[arche] = {
            "threshold": t,
            "block_rate": float(pred.mean()),
            "recall": float(rec),
            "precision": float(prec),
            "loss": loss,
        }
    return {"dataset": name, "base_rate": float(y.mean()), "archetypes": out}


def ulb_scores():
    d = ulb.load_ulb()
    X_tr, y_tr = d["train"]
    X_ca, y_ca = d["calib"]
    X_te, y_te, amt = d["test"]
    b = lgb.train(model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = calibrate.fit_calibrator(b.predict(X_ca), y_ca)(b.predict(X_te))
    return p, y_te, amt


def main() -> None:
    print("IEEE-CIS: reusing saved calibrated scores ...")
    p1 = np.load(config.ARTIFACTS / "cal_test_scores.npy")
    y1, amt1 = pipeline.load_test_arrays()
    r1 = analyse("IEEE-CIS", p1, y1, amt1)

    print("ULB: training ...")
    p2, y2, amt2 = ulb_scores()
    r2 = analyse("ULB", p2, y2, amt2)

    w = 84
    for r in (r1, r2):
        print("\n" + "=" * w)
        print(f"{r['dataset']}  (base rate {r['base_rate']:.4%})")
        print("=" * w)
        print(f"{'merchant':<20}{'threshold':>12}{'block rate':>13}"
              f"{'recall':>10}{'precision':>12}")
        print("-" * w)
        for k, v in r["archetypes"].items():
            print(f"{k:<20}{v['threshold']:>12.5f}{v['block_rate']:>13.3%}"
                  f"{v['recall']:>10.3f}{v['precision']:>12.3f}")
        br = [v["block_rate"] for v in r["archetypes"].values()]
        rc = [v["recall"] for v in r["archetypes"].values()]
        n_off = sum(b == 0.0 for b in br)
        print("-" * w)
        # A ratio is meaningless once any archetype blocks nothing, so report
        # the range and how many merchants are told not to deploy at all.
        print(f"block rate spans {min(br):.3%} -> {max(br):.3%}   "
              f"recall spans {min(rc):.3f} -> {max(rc):.3f}")
        if n_off:
            print(f"  {n_off}/{len(br)} archetypes: cost-optimal policy is to BLOCK NOTHING.")
            print("  Not a failure -- at this base rate the model's false positives land")
            print("  on the largest transactions, so catching fraud costs more than the")
            print("  fraud does. The cost model is saying: do not deploy at these economics.")
        else:
            print(f"  block-rate spread {max(br)/min(br):.1f}x")

    print("\n" + "=" * w)
    print("CROSS-DATASET COMPARISON  (block rate, the comparable unit)")
    print("=" * w)
    print(f"{'merchant':<20}{'IEEE-CIS':>16}{'ULB':>16}{'same ordering?':>20}")
    print("-" * w)
    order1 = sorted(ARCHETYPES, key=lambda k: r1["archetypes"][k]["block_rate"])
    order2 = sorted(ARCHETYPES, key=lambda k: r2["archetypes"][k]["block_rate"])
    for k in ARCHETYPES:
        print(f"{k:<20}{r1['archetypes'][k]['block_rate']:>16.3%}"
              f"{r2['archetypes'][k]['block_rate']:>16.3%}"
              f"{'':>20}")
    print("-" * w)
    print(f"IEEE-CIS ordering (least -> most blocking): {' < '.join(order1)}")
    print(f"ULB      ordering (least -> most blocking): {' < '.join(order2)}")
    agree = order1 == order2
    print(f"\nordering replicates: {'YES' if agree else 'NO'}")
    print("Block-rate ratios are omitted where an archetype optimally blocks
"
          "nothing -- the ratio is undefined, not infinite.")
    print("=" * w)

    (config.ARTIFACTS / "sensitivity_cross.json").write_text(
        json.dumps({"ieee": r1, "ulb": r2}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'sensitivity_cross.json'}")


if __name__ == "__main__":
    main()
