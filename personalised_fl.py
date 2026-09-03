"""Personalised federated learning with gated routing.

federated.py showed plain averaging is worse than not collaborating. The
diagnosis (finding #8) is that averaging dilutes COMPETENCE: each client votes on
traffic it never trained on, and the ignorant majority outvotes the one model
that knows the segment.

That diagnosis makes a prediction: if we stop treating all clients as equally
qualified on every transaction, the loss should be recoverable. Three ways to do
that, all still without moving raw rows between clients:

  A HARD GATING          route each transaction to its own client's model.
                         Pure personalisation, zero collaboration.

  B PERSONALISED FL      build a global model by federated aggregation, then let
                         each client CONTINUE BOOSTING it on its own data, and
                         route each transaction to its own client's fine-tuned
                         model. This is the standard personalised-FL recipe:
                         shared initialisation, local adaptation.

  C SOFT GATE (stacking) learn, per segment, how much to trust EACH client's
                         model, using only the calibration slice. If client W's
                         model is useless on segment C the gate should drive its
                         weight to zero -- and where a foreign client genuinely
                         helps, the gate can borrow from it.

Reference: uniform averaging (the thing that failed) and the centralised
ensemble (privacy-violating ceiling).

PRIVACY NOTE, stated honestly. A and B share only model artifacts, like plain
FedAvg. C additionally requires the server to see each client's PREDICTIONS on a
shared validation slice plus its labels, to fit the gate. That is a weaker
guarantee than A or B -- common in federated stacking and distillation, but it
is not the same privacy story, and it should not be presented as if it were.
"""
from __future__ import annotations

import json

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

import calibrate
import config
import data
import features
from leaderboard import XGB_P, evaluate

ROUNDS_TOTAL = 300
MIN_CLIENT_ROWS = 2_000


def main() -> None:
    print("loading ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    tr_df, ca_df, te_df = data.temporal_split(raw)
    cols = audit["online"]
    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te_df, cols, cats)

    s_tr, s_ca, s_te = (d["ProductCD"].to_numpy() for d in (tr_df, ca_df, te_df))
    clients = [c for c in sorted(set(s_tr))
               if (s_tr == c).sum() >= MIN_CLIENT_ROWS
               and (s_te == c).sum() >= 200 and y_te[s_te == c].sum() >= 10]
    per_client = ROUNDS_TOTAL // len(clients)
    print(f"{len(clients)} clients x {per_client} trees")

    d_ca_all = xgb.DMatrix(X_ca, enable_categorical=True)
    d_te_all = xgb.DMatrix(X_te, enable_categorical=True)
    D = lambda X, y=None: xgb.DMatrix(X, label=y, enable_categorical=True)

    # --- round 1: every client trains locally ------------------------------
    print("round 1: local training ...")
    local = {}
    for c in clients:
        m = s_tr == c
        local[c] = xgb.train(XGB_P, D(X_tr[m], y_tr[m]), num_boost_round=per_client)

    # matrices of every client's opinion on every row
    P_ca = np.column_stack([local[c].predict(d_ca_all) for c in clients])
    P_te = np.column_stack([local[c].predict(d_te_all) for c in clients])

    cal = calibrate.fit_calibrator
    R = {}

    # uniform averaging -- the failure we are trying to beat
    R["uniform averaging (failed)"] = evaluate(
        cal(P_ca.mean(1), y_ca)(P_te.mean(1)), y_te, amt_te)

    # --- A. hard gating -----------------------------------------------------
    print("A: hard gating (own client only) ...")
    hard = np.full(len(y_te), np.nan)
    for i, c in enumerate(clients):
        m_ca, m_te = s_ca == c, s_te == c
        f = cal(P_ca[m_ca, i], y_ca[m_ca])
        hard[m_te] = f(P_te[m_te, i])
    R["A hard gating"] = evaluate(hard, y_te, amt_te)

    # --- B. personalised FL: global init, then local fine-tuning -----------
    print("B: personalised FL (global init + local fine-tune) ...")
    # federated global model: sequential passes so it sees every client's data
    glob = None
    for c in clients:
        m = s_tr == c
        glob = xgb.train(XGB_P, D(X_tr[m], y_tr[m]),
                         num_boost_round=max(10, per_client // 2),
                         xgb_model=glob)
    pers = np.full(len(y_te), np.nan)
    for c in clients:
        m_tr, m_ca, m_te = s_tr == c, s_ca == c, s_te == c
        b = xgb.train(XGB_P, D(X_tr[m_tr], y_tr[m_tr]),
                      num_boost_round=per_client, xgb_model=glob)
        f = cal(b.predict(D(X_ca[m_ca])), y_ca[m_ca])
        pers[m_te] = f(b.predict(D(X_te[m_te])))
    R["B personalised FL"] = evaluate(pers, y_te, amt_te)

    # --- C. soft gate: per-segment learned weights over all clients --------
    print("C: soft gate (per-segment stacking on calibration slice) ...")
    soft = np.full(len(y_te), np.nan)
    gate_weights = {}
    for c in clients:
        m_ca, m_te = s_ca == c, s_te == c
        if y_ca[m_ca].sum() < 10:            # not enough positives to fit a gate
            i = clients.index(c)
            soft[m_te] = cal(P_ca[m_ca, i], y_ca[m_ca])(P_te[m_te, i])
            gate_weights[c] = None
            continue
        g = LogisticRegression(max_iter=2000, C=1.0)
        g.fit(P_ca[m_ca], y_ca[m_ca])
        soft[m_te] = g.predict_proba(P_te[m_te])[:, 1]
        gate_weights[c] = {cl: round(float(w), 3)
                           for cl, w in zip(clients, g.coef_[0])}
    R["C soft gate (stacking)"] = evaluate(soft, y_te, amt_te)

    # --- D. centralised ensemble ceiling -----------------------------------
    print("D: centralised (pooled) ...")
    cen = xgb.train(XGB_P, D(X_tr, y_tr), num_boost_round=ROUNDS_TOTAL)
    R["D centralised (pooled)"] = evaluate(
        cal(cen.predict(d_ca_all), y_ca)(cen.predict(d_te_all)), y_te, amt_te)

    w = 104
    print("\n" + "=" * w)
    print("PERSONALISED FL WITH GATED ROUTING")
    print("=" * w)
    print(f"{'arm':<30}{'ROC-AUC':>10}{'PR-AUC':>10}{'frauds caught':>16}"
          f"{'recall':>9}{'FPR':>8}{'rupee loss':>16}")
    print("-" * w)
    for k, v in R.items():
        print(f"{k:<30}{v['roc_auc']:>10.4f}{v['pr_auc']:>10.4f}"
              f"{v['frauds_caught']:>9,}/{v['frauds_total']:<6,}"
              f"{v['recall_count']:>9.3f}{v['fpr']:>8.2%}{v['rupee_loss']:>16,.0f}")
    print("-" * w)

    unif = R["uniform averaging (failed)"]["rupee_loss"]
    hard_l = R["A hard gating"]["rupee_loss"]
    cen_l = R["D centralised (pooled)"]["rupee_loss"]
    for arm in ("A hard gating", "B personalised FL", "C soft gate (stacking)"):
        v = R[arm]["rupee_loss"]
        closed = (hard_l - v) / (hard_l - cen_l) if hard_l != cen_l else 0.0
        print(f"{arm:<30} vs uniform {unif - v:>+14,.0f}   "
              f"vs hard gating {hard_l - v:>+14,.0f}   "
              f"closes {closed:>6.0%} of remaining gap to pooled")
    print()
    print("gate weights learned per segment (how much each client is trusted):")
    for c, gw in gate_weights.items():
        print(f"  segment {c}: {gw}")
    print("\nSINGLE SEED. Arm C shares validation predictions + labels with the")
    print("server, a weaker privacy guarantee than arms A and B.")
    print("=" * w)

    (config.ARTIFACTS / "personalised_fl.json").write_text(
        json.dumps({"clients": clients, "results": R,
                    "gate_weights": gate_weights}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'personalised_fl.json'}")


if __name__ == "__main__":
    main()
