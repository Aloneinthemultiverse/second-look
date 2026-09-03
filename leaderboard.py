"""One table, every configuration, identical conditions.

Results in this project accumulated across scripts that used different
algorithms, hyperparameters and round budgets. That made cross-script
comparison unsound -- federated.py and pooling_value.py disagreed on whether
centralisation helps, and the round counts differed, so neither could be
trusted against the other.

This runs every configuration under one controlled setup:

  same features (77 online)        same temporal split
  same calibration (Platt)         same per-instance threshold tau*(x)
  same total tree budget           same seed

so the only thing that varies is the configuration itself.

Reports how much fraud each actually detects, by count and by value, with the
rupee consequence. This is the table to quote.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import calibrate
import config
import data
import features

ROUNDS = 300                # identical tree budget for every configuration
MIN_CLIENT_ROWS = 2_000

LGB_P = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 64,
         "min_data_in_leaf": 100, "feature_fraction": 0.8,
         "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
         "seed": config.SEED}
XGB_P = {"objective": "binary:logistic", "eval_metric": "aucpr",
         "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
         "colsample_bytree": 0.8, "tree_method": "hist",
         "enable_categorical": True, "seed": config.SEED}


def evaluate(p, y, amount):
    """Detection and cost at the per-instance cost-optimal threshold."""
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    blocked = p >= c_fp / (c_fp + c_fn)
    tp, fp, fn = blocked & (y == 1), blocked & (y == 0), (~blocked) & (y == 1)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "frauds_caught": int(tp.sum()),
        "frauds_total": int((y == 1).sum()),
        "recall_count": float(tp.sum() / max((y == 1).sum(), 1)),
        "recall_value": float(amount[tp].sum() / max(amount[y == 1].sum(), 1)),
        "precision": float(tp.sum() / max(blocked.sum(), 1)),
        "fpr": float(fp.sum() / max((y == 0).sum(), 1)),
        "rupee_loss": float((amount[fn] + config.CHARGEBACK_FEE_INR).sum()
                            + (amount[fp] * config.MERCHANT_MARGIN_RATE
                               + config.CUSTOMER_LTV_INR).sum()),
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
    N_tr = X_tr.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c).fillna(-999)
    N_ca = X_ca.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c).fillna(-999)
    N_te = X_te.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c).fillna(-999)

    s_tr, s_ca, s_te = (d["ProductCD"].to_numpy() for d in (tr_df, ca_df, te_df))
    clients = [c for c in sorted(set(s_tr))
               if (s_tr == c).sum() >= MIN_CLIENT_ROWS
               and (s_te == c).sum() >= 200 and y_te[s_te == c].sum() >= 10]

    d_ca_all = xgb.DMatrix(X_ca, enable_categorical=True)
    d_te_all = xgb.DMatrix(X_te, enable_categorical=True)
    cal = calibrate.fit_calibrator
    R = {}

    print(f"single LightGBM ({ROUNDS} rounds) ...")
    m = lgb.train(LGB_P, lgb.Dataset(X_tr, label=y_tr), num_boost_round=ROUNDS)
    lgb_ca, lgb_te = m.predict(X_ca), m.predict(X_te)
    R["single LightGBM"] = evaluate(cal(lgb_ca, y_ca)(lgb_te), y_te, amt_te)

    print(f"single XGBoost ({ROUNDS} rounds) ...")
    xm = xgb.train(XGB_P, xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True),
                   num_boost_round=ROUNDS)
    xgb_ca, xgb_te = xm.predict(d_ca_all), xm.predict(d_te_all)
    R["single XGBoost"] = evaluate(cal(xgb_ca, y_ca)(xgb_te), y_te, amt_te)

    print("RandomForest ...")
    rf = RandomForestClassifier(n_estimators=ROUNDS, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=config.SEED).fit(N_tr, y_tr)
    rf_ca, rf_te = rf.predict_proba(N_ca)[:, 1], rf.predict_proba(N_te)[:, 1]
    R["RandomForest"] = evaluate(cal(rf_ca, y_ca)(rf_te), y_te, amt_te)

    print("ENSEMBLE (LGB + XGB + RF) ...")
    ens_ca = np.mean([lgb_ca, xgb_ca, rf_ca], axis=0)
    ens_te = np.mean([lgb_te, xgb_te, rf_te], axis=0)
    R["ENSEMBLE (3 models)"] = evaluate(cal(ens_ca, y_ca)(ens_te), y_te, amt_te)

    # federated arms -- same per-client budget, so totals match centralised
    per_client = ROUNDS // len(clients)
    print(f"federated: {len(clients)} clients x {per_client} rounds ...")
    boosters, local_pred = {}, np.full(len(y_te), np.nan)
    for c in clients:
        m_tr, m_ca, m_te = s_tr == c, s_ca == c, s_te == c
        b = xgb.train(XGB_P, xgb.DMatrix(X_tr[m_tr], label=y_tr[m_tr],
                                         enable_categorical=True),
                      num_boost_round=per_client)
        boosters[c] = b
        d_ca_c = xgb.DMatrix(X_ca[m_ca], enable_categorical=True)
        d_te_c = xgb.DMatrix(X_te[m_te], enable_categorical=True)
        local_pred[m_te] = cal(b.predict(d_ca_c), y_ca[m_ca])(b.predict(d_te_c))
    R["local only (no sharing)"] = evaluate(local_pred, y_te, amt_te)

    bag_ca = np.mean([boosters[c].predict(d_ca_all) for c in clients], axis=0)
    bag_te = np.mean([boosters[c].predict(d_te_all) for c in clients], axis=0)
    R["FedXGBBagging"] = evaluate(cal(bag_ca, y_ca)(bag_te), y_te, amt_te)

    print("centralised XGBoost (same total budget) ...")
    # already computed as single XGBoost with ROUNDS -- reuse for a fair row
    R["centralised (pooled)"] = R["single XGBoost"]

    w = 112
    print("\n" + "=" * w)
    print(f"LEADERBOARD -- identical conditions: 77 features, temporal split, "
          f"Platt, tau*(x), {ROUNDS} trees")
    print("=" * w)
    print(f"{'configuration':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'frauds caught':>15}"
          f"{'recall':>9}{'val-recall':>12}{'prec':>8}{'FPR':>8}{'rupee loss':>16}")
    print("-" * w)
    for k, v in R.items():
        print(f"{k:<26}{v['roc_auc']:>9.4f}{v['pr_auc']:>9.4f}"
              f"{v['frauds_caught']:>8,}/{v['frauds_total']:<6,}"
              f"{v['recall_count']:>9.3f}{v['recall_value']:>12.3f}"
              f"{v['precision']:>8.3f}{v['fpr']:>8.2%}{v['rupee_loss']:>16,.0f}")
    print("-" * w)

    best = min(R, key=lambda k: R[k]["rupee_loss"])
    base = R["single LightGBM"]
    print(f"best by rupee loss: {best}")
    print(f"  vs single LightGBM: Rs {base['rupee_loss'] - R[best]['rupee_loss']:+,.0f}")
    print(f"  frauds caught: {R[best]['frauds_caught']:,} of "
          f"{R[best]['frauds_total']:,} ({R[best]['recall_count']:.1%} by count, "
          f"{R[best]['recall_value']:.1%} by value)")
    fed = R["FedXGBBagging"]["rupee_loss"] - R["local only (no sharing)"]["rupee_loss"]
    print(f"\nfederation vs no collaboration: Rs {-fed:+,.0f} "
          f"({'helps' if fed < 0 else 'HURTS'})")
    print("SINGLE SEED. Ensemble gain is separately verified in verify_fusion.py.")
    print("=" * w)

    (config.ARTIFACTS / "leaderboard.json").write_text(
        json.dumps({"rounds": ROUNDS, "clients": clients, "results": R},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'leaderboard.json'}")


if __name__ == "__main__":
    main()
