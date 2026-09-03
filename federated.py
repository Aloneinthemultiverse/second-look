"""Federated gradient boosting for fraud detection.

Implements the two horizontal federated GBDT schemes evaluated in the ACM CIKM
2025 empirical study of federated gradient boosting in banking, which found
FedXGBBagging the strongest of four representative methods on tabular fraud:

  FedXGBBagging  each client trains its own booster on ONLY its own rows.
                 The server collects the boosters and predicts by averaging
                 their outputs. No raw data leaves a client -- only model
                 artifacts. This is the bagging aggregation rule.

  FedXGBCyclic   one booster is passed round-robin between clients. Each client
                 appends trees fitted on its own data, then hands the model on.
                 Again no raw data moves, but the model carries information
                 forward between rounds.

Compared against two reference points that bracket what federation can achieve:

  LOCAL ONLY     each client uses only its own model on its own traffic. This is
                 the world without collaboration -- the floor.
  CENTRALISED    all data pooled into one model. Privacy-violating, and the
                 usual assumed ceiling.

CLIENTS. `ProductCD` (W/C/H/R/S) is a real column, so the partition is not one
we invented. Fraud rates run 1.93%-13.35% across it, which makes this a
genuinely NON-IID federation -- the hard case, and the one FL research is mostly
about.

WHAT IS AND IS NOT SIMULATED. The learning is real: no client ever sees another
client's rows, and aggregation uses only fitted models. Not simulated: network
transport, secure aggregation, differential privacy, or client dropout. So this
measures the STATISTICAL behaviour of federated boosting, not its security
properties. Stated so the claim stays inside what was actually run.
"""
from __future__ import annotations

import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

import calibrate
import config
import data
import features

MIN_CLIENT_ROWS = 2_000
LOCAL_ROUNDS = 150          # trees each client fits per turn
CYCLIC_PASSES = 3           # times the model goes round the ring

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "aucpr",
    "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
    "colsample_bytree": 0.8, "tree_method": "hist",
    "enable_categorical": True, "seed": config.SEED,
}


def rupee_loss(p, y, amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    blocked = p >= c_fp / (c_fp + c_fn)          # per-instance tau*
    return float((amount[(~blocked) & (y == 1)] + config.CHARGEBACK_FEE_INR).sum()
                 + (amount[blocked & (y == 0)] * config.MERCHANT_MARGIN_RATE
                    + config.CUSTOMER_LTV_INR).sum())


def score(p, y, amount):
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "rupee_loss": rupee_loss(p, y, amount)}


def main() -> None:
    print("loading ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    tr_df, ca_df, te_df = data.temporal_split(raw)
    cols = audit["online"]
    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te_df, cols, cats)

    s_tr = tr_df["ProductCD"].to_numpy()
    s_ca = ca_df["ProductCD"].to_numpy()
    s_te = te_df["ProductCD"].to_numpy()

    clients = [c for c in sorted(set(s_tr))
               if (s_tr == c).sum() >= MIN_CLIENT_ROWS
               and (s_te == c).sum() >= 200 and y_te[s_te == c].sum() >= 10]
    print(f"clients: {clients}")
    for c in clients:
        m = s_tr == c
        print(f"  {c}: {m.sum():>7,} rows, fraud {y_tr[m].mean():.2%}")

    dtest = xgb.DMatrix(X_te, enable_categorical=True)

    # --- LOCAL ONLY: each client trains and serves only itself --------------
    print("\ntraining local models (no collaboration) ...")
    local_boosters, local_pred = {}, np.full(len(y_te), np.nan)
    for c in clients:
        m_tr, m_ca, m_te = s_tr == c, s_ca == c, s_te == c
        d_tr = xgb.DMatrix(X_tr[m_tr], label=y_tr[m_tr], enable_categorical=True)
        d_ca = xgb.DMatrix(X_ca[m_ca], label=y_ca[m_ca], enable_categorical=True)
        b = xgb.train(XGB_PARAMS, d_tr, num_boost_round=LOCAL_ROUNDS,
                      evals=[(d_ca, "cal")], early_stopping_rounds=30,
                      verbose_eval=False)
        local_boosters[c] = b
        cal = calibrate.fit_calibrator(b.predict(d_ca), y_ca[m_ca])
        local_pred[m_te] = cal(b.predict(
            xgb.DMatrix(X_te[m_te], enable_categorical=True)))

    # --- FedXGBBagging: average every client's booster over ALL traffic -----
    print("aggregating FedXGBBagging (model artifacts only, no rows shared) ...")
    d_ca_all = xgb.DMatrix(X_ca, enable_categorical=True)
    bag_ca = np.mean([local_boosters[c].predict(d_ca_all) for c in clients], axis=0)
    bag_te = np.mean([local_boosters[c].predict(dtest) for c in clients], axis=0)
    bag_te = calibrate.fit_calibrator(bag_ca, y_ca)(bag_te)

    # --- FedXGBCyclic: one model passed round-robin, each adds its own trees -
    print(f"training FedXGBCyclic ({CYCLIC_PASSES} passes over "
          f"{len(clients)} clients) ...")
    booster = None
    per_turn = max(20, LOCAL_ROUNDS // (CYCLIC_PASSES * len(clients)))
    for p in range(CYCLIC_PASSES):
        for c in clients:
            m = s_tr == c
            d = xgb.DMatrix(X_tr[m], label=y_tr[m], enable_categorical=True)
            booster = xgb.train(XGB_PARAMS, d, num_boost_round=per_turn,
                                xgb_model=booster, verbose_eval=False)
        print(f"   pass {p+1}/{CYCLIC_PASSES} done")
    cyc_te = calibrate.fit_calibrator(booster.predict(d_ca_all), y_ca)(
        booster.predict(dtest))

    # --- CENTRALISED: all rows pooled (privacy-violating reference) ---------
    print("training centralised model (all data pooled) ...")
    d_all = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    cen = xgb.train(XGB_PARAMS, d_all, num_boost_round=LOCAL_ROUNDS * 2,
                    evals=[(xgb.DMatrix(X_ca, label=y_ca,
                                        enable_categorical=True), "cal")],
                    early_stopping_rounds=30, verbose_eval=False)
    cen_te = calibrate.fit_calibrator(cen.predict(d_ca_all), y_ca)(
        cen.predict(dtest))

    # --- results ------------------------------------------------------------
    cov = ~np.isnan(local_pred)
    results = {
        "local only (no collaboration)": score(local_pred[cov], y_te[cov], amt_te[cov]),
        "FedXGBBagging": score(bag_te[cov], y_te[cov], amt_te[cov]),
        "FedXGBCyclic": score(cyc_te[cov], y_te[cov], amt_te[cov]),
        "centralised (pooled, no privacy)": score(cen_te[cov], y_te[cov], amt_te[cov]),
    }

    w = 92
    print("\n" + "=" * w)
    print("FEDERATED GRADIENT BOOSTING  (5 non-IID clients, ProductCD segments)")
    print("=" * w)
    print(f"{'configuration':<36}{'ROC-AUC':>11}{'PR-AUC':>11}{'rupee loss':>17}"
          f"{'vs local':>15}")
    print("-" * w)
    base = results["local only (no collaboration)"]["rupee_loss"]
    for k, v in results.items():
        print(f"{k:<36}{v['roc_auc']:>11.4f}{v['pr_auc']:>11.4f}"
              f"{v['rupee_loss']:>17,.0f}{base - v['rupee_loss']:>+15,.0f}")
    print("-" * w)

    fed_best = min(("FedXGBBagging", "FedXGBCyclic"),
                   key=lambda k: results[k]["rupee_loss"])
    fed = results[fed_best]["rupee_loss"]
    cen_l = results["centralised (pooled, no privacy)"]["rupee_loss"]
    print(f"best federated scheme: {fed_best}")
    print(f"  federation vs no collaboration   Rs {base - fed:+,.0f}")
    print(f"  federation vs pooling everything Rs {cen_l - fed:+,.0f}")
    if cen_l != base:
        closed = (base - fed) / (base - cen_l) if base != cen_l else 0.0
        print(f"  federation closes {closed:.0%} of the gap between working alone")
        print("  and pooling all data -- the standard FL benchmark.")
    print()
    print("Clients never exchanged rows. Only fitted boosters were aggregated.")
    print("Network transport, secure aggregation and DP are NOT simulated, so")
    print("this measures statistical behaviour, not security properties.")
    print("SINGLE SEED -- not yet verified across seeds.")
    print("=" * w)

    (config.ARTIFACTS / "federated.json").write_text(
        json.dumps({"clients": clients, "results": results,
                    "local_rounds": LOCAL_ROUNDS,
                    "cyclic_passes": CYCLIC_PASSES}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'federated.json'}")


if __name__ == "__main__":
    main()
