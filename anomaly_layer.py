"""Does an unsupervised outlier layer catch the expensive fraud we miss?

Motivation. By count we catch 44% of fraud; by VALUE only 39%. Loss is
denominated in rupees, so the value gap is what matters.

PayPal's published approach is explicitly multilayered -- "velocity rules,
unsupervised outlier detection, and supervised classification" -- while this
project has supervised classification only. The argument for the missing layer:
expensive fraud is rare, so it is under-represented in supervised training,
whereas an outlier detector does not need labels and can flag patterns the
classifier never learned. Research on transaction-amount distributions makes the
same point: >90% of purchases are small, so the high-value tail gets drowned.

Test: add an IsolationForest anomaly score alongside the supervised models and
measure VALUE-recall, not count-recall.

Honest framing up front: an anomaly score is not a fraud score. Outliers include
legitimate large purchases. The question is whether it carries information the
supervised models lack, and whether that survives being turned into rupees.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import calibrate
import config
import data
import features
import model
from fusion import per_instance_tau, to_numeric

HIGH_Q = 0.75


def value_metrics(p, y, amount):
    blocked = p >= per_instance_tau(amount)
    tp, fp, fn = blocked & (y == 1), blocked & (y == 0), (~blocked) & (y == 1)
    hi = amount >= np.quantile(amount, HIGH_Q)
    loss = float((amount[fp] * config.MERCHANT_MARGIN_RATE
                  + config.CUSTOMER_LTV_INR).sum()
                 + (amount[fn] + config.CHARGEBACK_FEE_INR).sum())
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "recall_count": float(tp.sum() / max((y == 1).sum(), 1)),
        "recall_value": float(amount[tp].sum() / max(amount[y == 1].sum(), 1)),
        "recall_value_high": float(amount[tp & hi].sum()
                                   / max(amount[(y == 1) & hi].sum(), 1)),
        "fpr": float(fp.sum() / max((y == 0).sum(), 1)),
        "rupee_loss": loss,
    }


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

    ca, te = {}, {}
    print("supervised: lightgbm ...")
    m = lgb.train(model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    ca["lgb"], te["lgb"] = m.predict(X_ca), m.predict(X_te)

    print("supervised: xgboost ...")
    xm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric="aucpr", early_stopping_rounds=50,
                           random_state=config.SEED, n_jobs=-1,
                           enable_categorical=True, tree_method="hist")
    xm.fit(X_tr, y_tr, eval_set=[(X_ca, y_ca)], verbose=False)
    ca["xgb"], te["xgb"] = xm.predict_proba(X_ca)[:, 1], xm.predict_proba(X_te)[:, 1]

    print("supervised: randomforest ...")
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=config.SEED).fit(N_tr, y_tr)
    ca["rf"], te["rf"] = rf.predict_proba(N_ca)[:, 1], rf.predict_proba(N_te)[:, 1]

    print("unsupervised: isolation forest (trained on GENUINE rows only) ...")
    # Fit on non-fraud only, so "anomalous" means "unlike normal traffic".
    iso = IsolationForest(n_estimators=300, max_samples=100_000,
                          contamination="auto", random_state=config.SEED,
                          n_jobs=-1).fit(N_tr[y_tr == 0])
    # score_samples: lower = more anomalous. Flip so higher = more suspicious.
    ca["iso"] = -iso.score_samples(N_ca)
    te["iso"] = -iso.score_samples(N_te)

    iso_auc = roc_auc_score(y_te, te["iso"])
    print(f"   isolation forest alone: ROC-AUC {iso_auc:.4f} "
          f"({'informative' if iso_auc > 0.55 else 'barely better than chance'})")

    def ev(name, p_ca, p_te):
        return name, value_metrics(calibrate.fit_calibrator(p_ca, y_ca)(p_te),
                                   y_te, amt_te)

    sup_ca = np.mean([ca[k] for k in ("lgb", "xgb", "rf")], axis=0)
    sup_te = np.mean([te[k] for k in ("lgb", "xgb", "rf")], axis=0)

    # normalise the anomaly score onto [0,1] before blending
    def norm(v, ref):
        lo, hi = ref.min(), ref.max()
        return np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)

    iso_ca_n, iso_te_n = norm(ca["iso"], ca["iso"]), norm(te["iso"], ca["iso"])

    rows = [
        ev("supervised only (LGB)", ca["lgb"], te["lgb"]),
        ev("supervised fusion (3)", sup_ca, sup_te),
        ev("isolation forest only", ca["iso"], te["iso"]),
        ev("fusion + 10% anomaly", 0.9 * sup_ca + 0.1 * iso_ca_n,
           0.9 * sup_te + 0.1 * iso_te_n),
        ev("fusion + 25% anomaly", 0.75 * sup_ca + 0.25 * iso_ca_n,
           0.75 * sup_te + 0.25 * iso_te_n),
    ]

    w = 108
    print("\n" + "=" * w)
    print("DOES AN UNSUPERVISED OUTLIER LAYER CATCH THE EXPENSIVE FRAUD? "
          "(PayPal-style multilayer)")
    print("=" * w)
    print(f"{'configuration':<26}{'ROC-AUC':>10}{'recall(count)':>15}"
          f"{'recall(VALUE)':>15}{'recall(top25%)':>16}{'FPR':>8}{'rupee loss':>16}")
    print("-" * w)
    out = {}
    for name, r in rows:
        out[name] = r
        print(f"{name:<26}{r['roc_auc']:>10.4f}{r['recall_count']:>15.3f}"
              f"{r['recall_value']:>15.3f}{r['recall_value_high']:>16.3f}"
              f"{r['fpr']:>8.2%}{r['rupee_loss']:>16,.0f}")
    print("-" * w)

    base = out["supervised fusion (3)"]
    best_v = max(out, key=lambda k: out[k]["recall_value"])
    best_r = min(out, key=lambda k: out[k]["rupee_loss"])
    print(f"best VALUE-recall: {best_v}  ({out[best_v]['recall_value']:.3f})")
    print(f"best rupee loss:   {best_r}  (Rs {out[best_r]['rupee_loss']:,.0f})")
    print(f"\nPayPal reports FPR below 5%; ours is "
          f"{base['fpr']:.2%} -- we are far more conservative already.")
    if not best_v.startswith("fusion +") and not best_r.startswith("fusion +"):
        print("The anomaly layer does NOT help here. Reported as a negative result.")
    print("=" * w)

    (config.ARTIFACTS / "anomaly_layer.json").write_text(
        json.dumps({"isolation_forest_auc": float(iso_auc), "results": out},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'anomaly_layer.json'}")


if __name__ == "__main__":
    main()
