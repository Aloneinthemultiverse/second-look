"""Testing the 2026 combinatorial-fusion result on our setup.

arXiv:2606.10393 (June 2026) reports ROC-AUC 0.9405 on IEEE-CIS by diversity-
weighted score fusion of Random Forest, XGBoost and LightGBM. We reach 0.8870
with a single LightGBM. Two explanations are possible and they have very
different implications:

  (a) the gap is the FEATURE RESTRICTION -- they use all 394 columns, we use the
      77 computable inside a checkout latency budget; or
  (b) the gap is the SINGLE-MODEL choice, in which case fusion should recover it
      even under our restriction.

This tests (b) directly: same 77 online features, same temporal split, same
calibration, three model families fused three ways.

The question that matters more than AUC: DO THE GAINS TRANSLATE INTO RUPEES?
A fusion that lifts AUC but not rupee loss would be consistent with this
project's whole argument -- that ranking metrics and operating-point economics
are different things.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import calibrate
import config
import data
import features
import model

SECONDS_PER_DAY = 86_400


def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Ordinal codes for RandomForest, which cannot take pandas categoricals."""
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) == "category":
            out[c] = out[c].cat.codes.astype(float)
    return out.fillna(-999.0)


def per_instance_tau(amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return c_fp / (c_fp + c_fn)


def rupee_loss(p, y, amount):
    blocked = p >= per_instance_tau(amount)
    fp, fn = blocked & (y == 0), (~blocked) & (y == 1)
    return float((amount[fp] * config.MERCHANT_MARGIN_RATE
                  + config.CUSTOMER_LTV_INR).sum()
                 + (amount[fn] + config.CHARGEBACK_FEE_INR).sum())


def card_precision_at_k(p, y, card, day, k=10):
    df = pd.DataFrame({"score": p, "fraud": y, "card1": card, "day": day})
    hits = tot = 0
    for _, d in df.groupby("day"):
        g = d.groupby("card1").agg(score=("score", "max"), fraud=("fraud", "max"))
        top = g.nlargest(min(k, len(g)), "score")
        hits += int(top["fraud"].sum()); tot += len(top)
    return hits / tot if tot else 0.0


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

    card = te_df["card1"].to_numpy()
    day = (te_df["TransactionDT"].to_numpy() // SECONDS_PER_DAY).astype(int)
    day -= day.min()

    raw_ca, raw_te, timings = {}, {}, {}

    t0 = time.perf_counter()
    print("training LightGBM ...")
    m = lgb.train(model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    raw_ca["lightgbm"], raw_te["lightgbm"] = m.predict(X_ca), m.predict(X_te)
    timings["lightgbm"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("training XGBoost ...")
    xm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric="aucpr", early_stopping_rounds=50,
                           random_state=config.SEED, n_jobs=-1,
                           enable_categorical=True, tree_method="hist")
    xm.fit(X_tr, y_tr, eval_set=[(X_ca, y_ca)], verbose=False)
    raw_ca["xgboost"] = xm.predict_proba(X_ca)[:, 1]
    raw_te["xgboost"] = xm.predict_proba(X_te)[:, 1]
    timings["xgboost"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("training RandomForest ...")
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=config.SEED)
    rf.fit(N_tr, y_tr)
    raw_ca["randomforest"] = rf.predict_proba(N_ca)[:, 1]
    raw_te["randomforest"] = rf.predict_proba(N_te)[:, 1]
    timings["randomforest"] = time.perf_counter() - t0

    # --- fusions ----------------------------------------------------------
    names = list(raw_te)
    aucs = {n: roc_auc_score(y_te, raw_te[n]) for n in names}

    def rank_norm(v):
        return rankdata(v) / len(v)

    R_ca = {n: rank_norm(raw_ca[n]) for n in names}
    R_te = {n: rank_norm(raw_te[n]) for n in names}

    fused_ca, fused_te = {}, {}
    fused_ca["average score"] = np.mean([raw_ca[n] for n in names], axis=0)
    fused_te["average score"] = np.mean([raw_te[n] for n in names], axis=0)
    fused_ca["average rank"] = np.mean([R_ca[n] for n in names], axis=0)
    fused_te["average rank"] = np.mean([R_te[n] for n in names], axis=0)

    # diversity-weighted: weight by validation performance, combine on ranks
    perf = np.array([average_precision_score(y_ca, raw_ca[n]) for n in names])
    wts = perf / perf.sum()
    fused_ca["diversity-weighted"] = np.average([R_ca[n] for n in names],
                                                axis=0, weights=wts)
    fused_te["diversity-weighted"] = np.average([R_te[n] for n in names],
                                                axis=0, weights=wts)

    # --- evaluate ---------------------------------------------------------
    rows = []
    for n in names:
        cal = calibrate.fit_calibrator(raw_ca[n], y_ca)(raw_te[n])
        rows.append(("single: " + n, cal))
    for n in fused_te:
        cal = calibrate.fit_calibrator(fused_ca[n], y_ca)(fused_te[n])
        rows.append(("fusion: " + n, cal))

    w = 100
    print("\n" + "=" * w)
    print("COMBINATORIAL FUSION (arXiv:2606.10393) ON OUR 77 ONLINE FEATURES")
    print("=" * w)
    print(f"{'model':<30}{'ROC-AUC':>10}{'PR-AUC':>10}{'cardP@10':>11}"
          f"{'rupee loss':>16}{'vs LightGBM':>16}")
    print("-" * w)
    out = {}
    base_loss = None
    for label, p in rows:
        r = {"roc_auc": float(roc_auc_score(y_te, p)),
             "pr_auc": float(average_precision_score(y_te, p)),
             "card_p10": float(card_precision_at_k(p, y_te, card, day)),
             "rupee_loss": rupee_loss(p, y_te, amt_te)}
        out[label] = r
        if label == "single: lightgbm":
            base_loss = r["rupee_loss"]
        rows_delta = "" if base_loss is None else f"{base_loss - r['rupee_loss']:+,.0f}"
        print(f"{label:<30}{r['roc_auc']:>10.4f}{r['pr_auc']:>10.4f}"
              f"{r['card_p10']:>11.3f}{r['rupee_loss']:>16,.0f}{rows_delta:>16}")
    print("-" * w)
    print(f"training time: " + "  ".join(f"{k} {v:.0f}s" for k, v in timings.items()))

    best_auc = max(out, key=lambda k: out[k]["roc_auc"])
    best_rup = min(out, key=lambda k: out[k]["rupee_loss"])
    print(f"\nbest ROC-AUC:    {best_auc}  ({out[best_auc]['roc_auc']:.4f})")
    print(f"best rupee loss: {best_rup}  (Rs {out[best_rup]['rupee_loss']:,.0f})")
    print(f"published SOTA on this dataset: 0.9405 ROC-AUC (all 394 features)")
    print(f"our best here:                  {out[best_auc]['roc_auc']:.4f} "
          f"(77 online features, temporal split)")
    if best_auc != best_rup:
        print("\nNOTE: the model that ranks best is NOT the model that loses least")
        print("money. That is this project's central claim, reproduced on an")
        print("ensemble comparison.")
    print("=" * w)

    (config.ARTIFACTS / "fusion.json").write_text(
        json.dumps({"results": out, "timings": timings,
                    "weights": dict(zip(names, wts.tolist()))}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'fusion.json'}")


if __name__ == "__main__":
    main()
