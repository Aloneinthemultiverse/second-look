"""Can calibration fix federated aggregation? And does a federated ensemble help?

federated.py found FedXGBBagging catastrophically worse than not collaborating
(AUC 0.8838 -> 0.7891). The diagnosis was specific: clients have base rates from
1.93% to 13.35%, so a client trained on 13% fraud emits systematically larger
raw scores than one trained on 1.9%. Averaging those raw numbers is
arithmetically incoherent -- they are not on the same scale.

If that diagnosis is right, the fix follows from it: calibrate EACH CLIENT's
model locally before aggregating, so every vote is an actual probability on a
common scale, then average. Each client fits its calibrator on its own
calibration slice; still no raw data leaves any client.

Four arms, all with the same tree budget:

  A  raw bagging            average raw scores            (what failed)
  B  calibrated bagging     calibrate per client, then average   (the fix)
  C  federated ensemble     each client trains LGB+XGB+RF, calibrates,
                            then all are averaged        (more capacity per client)
  D  centralised ensemble   pooled data, 3 models        (privacy-violating ceiling)

Reference point: local-only, each client serving just its own traffic.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import calibrate
import config
import data
import features
from leaderboard import LGB_P, XGB_P, evaluate

ROUNDS_TOTAL = 300
MIN_CLIENT_ROWS = 2_000


def numeric(df):
    return df.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c).fillna(-999)


def main() -> None:
    print("loading ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    tr_df, ca_df, te_df = data.temporal_split(raw)
    cols = audit["online"]
    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te_df, cols, cats)
    N_ca, N_te = numeric(X_ca), numeric(X_te)

    s_tr, s_ca, s_te = (d["ProductCD"].to_numpy() for d in (tr_df, ca_df, te_df))
    clients = [c for c in sorted(set(s_tr))
               if (s_tr == c).sum() >= MIN_CLIENT_ROWS
               and (s_te == c).sum() >= 200 and y_te[s_te == c].sum() >= 10]
    per_client = ROUNDS_TOTAL // len(clients)
    print(f"{len(clients)} clients x {per_client} trees each")

    d_ca_all = xgb.DMatrix(X_ca, enable_categorical=True)
    d_te_all = xgb.DMatrix(X_te, enable_categorical=True)

    raw_ca, raw_te = [], []          # arm A ingredients
    cal_ca, cal_te = [], []          # arm B ingredients
    ens_ca, ens_te = [], []          # arm C ingredients
    local_pred = np.full(len(y_te), np.nan)

    for c in clients:
        m_tr, m_ca, m_te = s_tr == c, s_ca == c, s_te == c

        # --- each client's XGBoost, trained only on its own rows ------------
        b = xgb.train(XGB_P, xgb.DMatrix(X_tr[m_tr], label=y_tr[m_tr],
                                         enable_categorical=True),
                      num_boost_round=per_client)
        r_ca, r_te = b.predict(d_ca_all), b.predict(d_te_all)
        raw_ca.append(r_ca); raw_te.append(r_te)

        # local calibrator, fitted on this client's own calibration slice only
        f = calibrate.fit_calibrator(b.predict(xgb.DMatrix(X_ca[m_ca],
                                     enable_categorical=True)), y_ca[m_ca])
        cal_ca.append(f(r_ca)); cal_te.append(f(r_te))
        local_pred[m_te] = f(b.predict(xgb.DMatrix(X_te[m_te],
                                                   enable_categorical=True)))

        # --- same client, now training a 3-model ensemble locally -----------
        lm = lgb.train(LGB_P, lgb.Dataset(X_tr[m_tr], label=y_tr[m_tr]),
                       num_boost_round=per_client)
        rf = RandomForestClassifier(n_estimators=per_client, min_samples_leaf=5,
                                    max_features="sqrt", n_jobs=-1,
                                    random_state=config.SEED
                                    ).fit(numeric(X_tr[m_tr]), y_tr[m_tr])
        e_ca = np.mean([r_ca, lm.predict(X_ca), rf.predict_proba(N_ca)[:, 1]], axis=0)
        e_te = np.mean([r_te, lm.predict(X_te), rf.predict_proba(N_te)[:, 1]], axis=0)
        # calibrate this client's ensemble on its own slice
        fe = calibrate.fit_calibrator(e_ca[m_ca], y_ca[m_ca])
        ens_ca.append(fe(e_ca)); ens_te.append(fe(e_te))
        print(f"  client {c} done ({m_tr.sum():,} rows, "
              f"fraud {y_tr[m_tr].mean():.2%})")

    cal = calibrate.fit_calibrator
    R = {"local only (no sharing)": evaluate(local_pred, y_te, amt_te)}

    a_te = cal(np.mean(raw_ca, axis=0), y_ca)(np.mean(raw_te, axis=0))
    R["A raw bagging"] = evaluate(a_te, y_te, amt_te)

    b_te = cal(np.mean(cal_ca, axis=0), y_ca)(np.mean(cal_te, axis=0))
    R["B calibrated bagging"] = evaluate(b_te, y_te, amt_te)

    c_te = cal(np.mean(ens_ca, axis=0), y_ca)(np.mean(ens_te, axis=0))
    R["C federated ensemble"] = evaluate(c_te, y_te, amt_te)

    print("centralised ensemble (pooled) ...")
    lm = lgb.train(LGB_P, lgb.Dataset(X_tr, label=y_tr), num_boost_round=ROUNDS_TOTAL)
    xm = xgb.train(XGB_P, xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True),
                   num_boost_round=ROUNDS_TOTAL)
    rf = RandomForestClassifier(n_estimators=ROUNDS_TOTAL, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=config.SEED).fit(numeric(X_tr), y_tr)
    p_ca = np.mean([lm.predict(X_ca), xm.predict(d_ca_all),
                    rf.predict_proba(N_ca)[:, 1]], axis=0)
    p_te = np.mean([lm.predict(X_te), xm.predict(d_te_all),
                    rf.predict_proba(N_te)[:, 1]], axis=0)
    R["D centralised ensemble"] = evaluate(cal(p_ca, y_ca)(p_te), y_te, amt_te)

    w = 104
    print("\n" + "=" * w)
    print("DOES CALIBRATION FIX FEDERATED AGGREGATION?")
    print("=" * w)
    print(f"{'arm':<28}{'ROC-AUC':>10}{'PR-AUC':>10}{'frauds caught':>16}"
          f"{'recall':>9}{'FPR':>8}{'rupee loss':>16}")
    print("-" * w)
    for k, v in R.items():
        print(f"{k:<28}{v['roc_auc']:>10.4f}{v['pr_auc']:>10.4f}"
              f"{v['frauds_caught']:>9,}/{v['frauds_total']:<6,}"
              f"{v['recall_count']:>9.3f}{v['fpr']:>8.2%}{v['rupee_loss']:>16,.0f}")
    print("-" * w)

    loc = R["local only (no sharing)"]["rupee_loss"]
    cen = R["D centralised ensemble"]["rupee_loss"]
    for arm in ("A raw bagging", "B calibrated bagging", "C federated ensemble"):
        gain = loc - R[arm]["rupee_loss"]
        closed = gain / (loc - cen) if loc != cen else 0.0
        print(f"{arm:<28} vs local {gain:>+14,.0f}   closes {closed:>6.0%} of the "
              f"gap to pooled data")
    print()
    print(f"calibration effect: raw {R['A raw bagging']['roc_auc']:.4f} -> "
          f"calibrated {R['B calibrated bagging']['roc_auc']:.4f} AUC "
          f"({R['B calibrated bagging']['roc_auc'] - R['A raw bagging']['roc_auc']:+.4f})")
    print("SINGLE SEED.")
    print("=" * w)

    (config.ARTIFACTS / "federated_ensemble.json").write_text(
        json.dumps({"clients": clients, "rounds_total": ROUNDS_TOTAL,
                    "results": R}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'federated_ensemble.json'}")


if __name__ == "__main__":
    main()
