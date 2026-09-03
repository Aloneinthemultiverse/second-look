"""Produce the headline number.

This exists because it did not. The figure quoted everywhere -- 1,776 of 4,064
frauds, 43.7% -- was originally produced by an inline command in a terminal and
never committed, so nobody cloning the repo could reproduce the one number the
submission leads with. That is the same failure as retracting a bug in prose
while leaving it in the code.

Canonical configuration, and the reason for each choice:

  ENSEMBLE at full tree budget   LightGBM + XGBoost + RandomForest, simple
                                 averaging. Verified +Rs 1,178,231 across three
                                 seeds in verify_fusion.py.
  PLATT calibration              fit on the temporal calibration slice. Required
                                 before any cost arithmetic; preserves ranking,
                                 unlike isotonic.
  PER-INSTANCE threshold         tau*(x) = C_FP/(C_FP + C_FN), closed form from
                                 costs. Nothing is fitted on the test set, so
                                 there is no threshold to leak.

Run with --seeds to report the headline across three seeds rather than one.
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import calibrate
import config
import features
from leaderboard import LGB_P, XGB_P, evaluate

ROUNDS = 600
SEEDS = [42, 7, 2024]


def numeric(df):
    return df.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c).fillna(-999)


def run(seed: int, s) -> dict:
    X_tr, y_tr, _ = s["train"]
    X_ca, y_ca, _ = s["calib"]
    X_te, y_te, amt_te = s["test"]

    m = lgb.train(dict(LGB_P, seed=seed, bagging_seed=seed,
                       feature_fraction_seed=seed),
                  lgb.Dataset(X_tr, label=y_tr), num_boost_round=ROUNDS)
    xm = xgb.train(dict(XGB_P, seed=seed),
                   xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True),
                   num_boost_round=ROUNDS)
    rf = RandomForestClassifier(n_estimators=ROUNDS, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=seed).fit(numeric(X_tr), y_tr)

    d_ca = xgb.DMatrix(X_ca, enable_categorical=True)
    d_te = xgb.DMatrix(X_te, enable_categorical=True)
    ca = np.mean([m.predict(X_ca), xm.predict(d_ca),
                  rf.predict_proba(numeric(X_ca))[:, 1]], axis=0)
    te = np.mean([m.predict(X_te), xm.predict(d_te),
                  rf.predict_proba(numeric(X_te))[:, 1]], axis=0)
    p = calibrate.fit_calibrator(ca, y_ca)(te)
    return evaluate(p, y_te, amt_te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", action="store_true",
                    help="report across 3 seeds instead of 1")
    args = ap.parse_args()

    print("loading ...")
    s = features.load_splits()
    seeds = SEEDS if args.seeds else [config.SEED]

    runs = []
    for sd in seeds:
        print(f"training ensemble at {ROUNDS} trees, seed {sd} ...")
        runs.append(run(sd, s))

    r = runs[0]
    w = 62
    print("\n" + "=" * w)
    print("CANONICAL RESULT  ensemble, per-instance tau*, nothing fit on test")
    print("=" * w)
    rows = [("frauds caught", f"{r['frauds_caught']:,} of {r['frauds_total']:,}"),
            ("recall (count)", f"{r['recall_count']:.1%}"),
            ("recall (value)", f"{r['recall_value']:.1%}"),
            ("precision", f"{r['precision']:.3f}"),
            ("false-positive rate", f"{r['fpr']:.2%}"),
            ("ROC-AUC", f"{r['roc_auc']:.4f}"),
            ("PR-AUC", f"{r['pr_auc']:.4f}"),
            ("rupee loss", f"Rs {r['rupee_loss']:,.0f}")]
    for k, v in rows:
        print(f"{k:<24}{v:>38}")

    if len(runs) > 1:
        print("-" * w)
        print("across seeds " + ", ".join(str(x) for x in seeds) + ":")
        for key, label, fmt in (("recall_count", "recall", ".1%"),
                                ("pr_auc", "PR-AUC", ".4f"),
                                ("rupee_loss", "rupee loss", ",.0f")):
            v = [x[key] for x in runs]
            print(f"  {label:<14}min {min(v):{fmt}}   max {max(v):{fmt}}   "
                  f"sd {st.stdev(v):{fmt}}")
    else:
        print("-" * w)
        print("single seed. run with --seeds for the spread.")
    print("=" * w)

    (config.ARTIFACTS / "canonical.json").write_text(
        json.dumps(runs[0] if len(runs) == 1 else
                   {"seeds": seeds, "runs": runs}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'canonical.json'}")


if __name__ == "__main__":
    main()
