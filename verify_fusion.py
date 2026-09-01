"""Is the fusion gain real, or is it one lucky seed?

fusion.py reported Rs 887,879 saved from a single run. Earlier in this project a
single-run finding of Rs 964,857 turned out to have a standard deviation larger
than its mean and was cut. The same standard applies here.

Retrains all three model families across seeds and reports the spread of the
gain. If the sd approaches the mean, the fusion result gets cut too.
"""
from __future__ import annotations

import json
import statistics as stats

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import calibrate
import config
import data
import features
import model
from fusion import per_instance_tau, rupee_loss, to_numeric

SEEDS = [42, 7, 2024]


def run_seed(seed, X_tr, y_tr, X_ca, y_ca, X_te, y_te, amt_te,
             N_tr, N_ca, N_te):
    params = dict(model.PARAMS, seed=seed, bagging_seed=seed,
                  feature_fraction_seed=seed)
    m = lgb.train(params, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    ca = {"lgb": m.predict(X_ca)}
    te = {"lgb": m.predict(X_te)}

    xm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric="aucpr", early_stopping_rounds=50,
                           random_state=seed, n_jobs=-1,
                           enable_categorical=True, tree_method="hist")
    xm.fit(X_tr, y_tr, eval_set=[(X_ca, y_ca)], verbose=False)
    ca["xgb"], te["xgb"] = xm.predict_proba(X_ca)[:, 1], xm.predict_proba(X_te)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=seed)
    rf.fit(N_tr, y_tr)
    ca["rf"], te["rf"] = rf.predict_proba(N_ca)[:, 1], rf.predict_proba(N_te)[:, 1]

    def evaluate(p_ca, p_te):
        p = calibrate.fit_calibrator(p_ca, y_ca)(p_te)
        return {"roc_auc": float(roc_auc_score(y_te, p)),
                "pr_auc": float(average_precision_score(y_te, p)),
                "rupee_loss": rupee_loss(p, y_te, amt_te)}

    base = evaluate(ca["lgb"], te["lgb"])
    rf_only = evaluate(ca["rf"], te["rf"])
    fus = evaluate(np.mean([ca[k] for k in ca], axis=0),
                   np.mean([te[k] for k in te], axis=0))
    return {"seed": seed, "lightgbm": base, "randomforest": rf_only,
            "fusion": fus,
            "gain_fusion": base["rupee_loss"] - fus["rupee_loss"],
            "gain_rf": base["rupee_loss"] - rf_only["rupee_loss"],
            "auc_gain": fus["roc_auc"] - base["roc_auc"]}


def main() -> None:
    print("loading ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    tr_df, ca_df, te_df = data.temporal_split(raw)
    cols = audit["online"]
    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te_df, cols, cats)
    N_tr, N_ca, N_te = to_numeric(X_tr), to_numeric(X_ca), to_numeric(X_te)

    runs = []
    for s in SEEDS:
        print(f"seed {s} ...")
        r = run_seed(s, X_tr, y_tr, X_ca, y_ca, X_te, y_te, amt_te,
                     N_tr, N_ca, N_te)
        runs.append(r)
        print(f"  fusion gain Rs {r['gain_fusion']:>10,.0f}   "
              f"RF-only gain Rs {r['gain_rf']:>10,.0f}   "
              f"AUC {r['auc_gain']:+.4f}")

    w = 84
    print("\n" + "=" * w)
    print("IS THE FUSION GAIN REAL?  (3 seeds, same temporal split)")
    print("=" * w)
    print(f"{'quantity':<28}{'min':>15}{'max':>15}{'mean':>15}{'sd':>11}")
    print("-" * w)
    for label, key, fmt in (("fusion gain (Rs)", "gain_fusion", ",.0f"),
                            ("RandomForest gain (Rs)", "gain_rf", ",.0f"),
                            ("fusion ROC-AUC gain", "auc_gain", ".4f")):
        v = [r[key] for r in runs]
        sd = stats.stdev(v)
        print(f"{label:<28}{min(v):>15{fmt}}{max(v):>15{fmt}}"
              f"{stats.mean(v):>15{fmt}}{sd:>11{fmt}}")
    print("-" * w)

    g = [r["gain_fusion"] for r in runs]
    mean, sd = stats.mean(g), stats.stdev(g)
    positive = sum(x > 0 for x in g)
    print(f"fusion gain positive in {positive}/{len(g)} seeds")
    if sd < abs(mean) / 2 and positive == len(g):
        verdict = ("ROBUST: gain is positive in every seed and the spread is well "
                   "under the mean. The fusion result stands.")
    elif positive == len(g):
        verdict = ("DIRECTIONALLY ROBUST: positive in every seed but noisy. Report "
                   "the range, not a point estimate.")
    else:
        verdict = ("NOT ROBUST: sign flips across seeds. Cut the claim, as we cut "
                   "the F1 finding.")
    print(verdict)
    print("=" * w)

    (config.ARTIFACTS / "verify_fusion.json").write_text(
        json.dumps({"seeds": runs, "mean_gain": mean, "sd_gain": sd,
                    "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'verify_fusion.json'}")


if __name__ == "__main__":
    main()
