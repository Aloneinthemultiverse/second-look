"""Does the anomaly-layer gain survive across seeds?

anomaly_layer.py reported Rs 334,171 from blending 10% IsolationForest into the
supervised fusion, on a single seed. Two single-seed findings in this project
have already been retracted (the F1 gap, the RandomForest paradox). This applies
the same test before the number is allowed into the README.

Also tracks VALUE-recall, because the anomaly layer was added to fix the
high-value gap and on one seed it made that gap worse. If that is consistent
across seeds, the layer is doing something different from what it was added for.
"""
from __future__ import annotations

import json
import statistics as stats

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score

import calibrate
import config
import data
import features
import model
from anomaly_layer import value_metrics
from fusion import to_numeric

SEEDS = [42, 7, 2024]
BLEND = 0.10


def run_seed(seed, X_tr, y_tr, X_ca, y_ca, X_te, y_te, amt_te, N_tr, N_ca, N_te):
    params = dict(model.PARAMS, seed=seed, bagging_seed=seed,
                  feature_fraction_seed=seed)
    m = lgb.train(params, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    ca, te = {"lgb": m.predict(X_ca)}, {"lgb": m.predict(X_te)}

    xm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric="aucpr", early_stopping_rounds=50,
                           random_state=seed, n_jobs=-1,
                           enable_categorical=True, tree_method="hist")
    xm.fit(X_tr, y_tr, eval_set=[(X_ca, y_ca)], verbose=False)
    ca["xgb"], te["xgb"] = xm.predict_proba(X_ca)[:, 1], xm.predict_proba(X_te)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=seed).fit(N_tr, y_tr)
    ca["rf"], te["rf"] = rf.predict_proba(N_ca)[:, 1], rf.predict_proba(N_te)[:, 1]

    iso = IsolationForest(n_estimators=300, max_samples=100_000,
                          contamination="auto", random_state=seed,
                          n_jobs=-1).fit(N_tr[y_tr == 0])
    a_ca, a_te = -iso.score_samples(N_ca), -iso.score_samples(N_te)
    lo, hi = a_ca.min(), a_ca.max()
    n_ca = np.clip((a_ca - lo) / (hi - lo + 1e-12), 0, 1)
    n_te = np.clip((a_te - lo) / (hi - lo + 1e-12), 0, 1)

    sup_ca = np.mean([ca[k] for k in ("lgb", "xgb", "rf")], axis=0)
    sup_te = np.mean([te[k] for k in ("lgb", "xgb", "rf")], axis=0)

    def ev(p_ca, p_te):
        return value_metrics(calibrate.fit_calibrator(p_ca, y_ca)(p_te),
                             y_te, amt_te)

    fus = ev(sup_ca, sup_te)
    ano = ev((1 - BLEND) * sup_ca + BLEND * n_ca,
             (1 - BLEND) * sup_te + BLEND * n_te)
    return {"seed": seed, "iso_auc": float(roc_auc_score(y_te, a_te)),
            "fusion": fus, "fusion_anomaly": ano,
            "gain": fus["rupee_loss"] - ano["rupee_loss"],
            "value_recall_delta": ano["recall_value"] - fus["recall_value"],
            "fpr_delta": ano["fpr"] - fus["fpr"]}


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
        print(f"  gain Rs {r['gain']:>10,.0f}   value-recall "
              f"{r['value_recall_delta']:+.4f}   FPR {r['fpr_delta']:+.4%}   "
              f"iso AUC {r['iso_auc']:.4f}")

    w = 86
    print("\n" + "=" * w)
    print(f"IS THE ANOMALY-LAYER GAIN REAL?  ({int(BLEND*100)}% blend, "
          f"{len(SEEDS)} seeds)")
    print("=" * w)
    print(f"{'quantity':<30}{'min':>15}{'max':>15}{'mean':>15}{'sd':>11}")
    print("-" * w)
    for label, key, fmt in (("rupee gain", "gain", ",.0f"),
                            ("value-recall change", "value_recall_delta", ".4f"),
                            ("FPR change", "fpr_delta", ".4%"),
                            ("isolation forest AUC", "iso_auc", ".4f")):
        v = [r[key] for r in runs]
        cells = [format(x, fmt) for x in (min(v), max(v), stats.mean(v))]
        cells.append(format(stats.stdev(v), fmt))
        print(f"{label:<30}" + "".join(f"{c:>15}" for c in cells[:3])
              + f"{cells[3]:>11}")
    print("-" * w)

    g = [r["gain"] for r in runs]
    vr = [r["value_recall_delta"] for r in runs]
    mean, sd = stats.mean(g), stats.stdev(g)
    pos = sum(x > 0 for x in g)
    print(f"rupee gain positive in {pos}/{len(g)} seeds")
    print(f"value-recall worse in {sum(x < 0 for x in vr)}/{len(vr)} seeds")

    if pos == len(g) and sd < abs(mean) / 2:
        verdict = ("ROBUST: adopt the anomaly layer. Note it improves rupees via "
                   "lower FPR, NOT via catching more expensive fraud.")
    elif pos == len(g):
        verdict = ("DIRECTIONALLY POSITIVE but noisy -- report a range, not a "
                   "point estimate.")
    else:
        verdict = ("NOT ROBUST: sign flips across seeds. Cut it, as we cut the "
                   "F1 gap and the RandomForest paradox.")
    print(verdict)
    print("=" * w)

    (config.ARTIFACTS / "verify_anomaly.json").write_text(
        json.dumps({"blend": BLEND, "seeds": runs, "mean_gain": mean,
                    "sd_gain": sd, "verdict": verdict}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'verify_anomaly.json'}")


if __name__ == "__main__":
    main()
