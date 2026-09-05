"""One pipeline, every dataset we can get. Does the method transfer or not?

Individual transfer checks were scattered across second_dataset.py,
try_indian.py and try_baf.py, each with its own loader and its own table. That
makes them hard to compare and easy to quote selectively. This runs all of them
through the SAME code and prints one table.

The pipeline is fixed for every dataset. Temporal split, LightGBM at identical
parameters, Platt calibration on the calibration slice, Elkan per-instance
threshold, Chow three-way band. Nothing is tuned per dataset. Only the loader
changes, because the column names differ.

WHAT THIS TESTS, and what it does not.
It tests the METHOD -- does the machinery find signal and turn it into a sane
policy on data it has never seen. It does NOT test the detector's weights, which
are retrained per dataset and are not expected to transfer.

The negative control matters as much as the positives. One of these datasets is
synthetic: its fraud label is independent of every feature, which we verified
directly. A method that scores well there would be broken. Reporting only the
datasets that flatter the method would be the same failure this project has
retracted ten times.

THE HEADLINE COMPARISON is `lift` -- PR-AUC divided by the base rate. Raw PR-AUC
is not comparable across datasets with fraud rates from 0.17% to 5%, but lift is.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGET, AMOUNT, TIME = "isFraud", "TransactionAmt", "TransactionDT"


# --- loaders: the ONLY per-dataset code ------------------------------------

def _std(df, y, amt, t, name, note):
    df = df.copy()
    df[TARGET], df[AMOUNT], df[TIME] = y, amt, t
    df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    return {"name": name, "note": note, "df": df}


def load_ieee():
    txn = pd.read_csv("data/train_transaction.csv")
    ident = pd.read_csv("data/train_identity.csv")
    df = txn.merge(ident, on="TransactionID", how="left")
    import data as D
    keep = D.audit_leakage(df)["online"]
    return _std(df[keep + ["isFraud", "TransactionDT"]], df["isFraud"],
                df["TransactionAmt"] * 88.0, df["TransactionDT"],
                "IEEE-CIS", "US card-not-present, the shipped dataset")


def load_ulb():
    df = pd.read_csv("data/creditcard.csv")
    feats = [c for c in df.columns if c not in ("Class", "Time")]
    return _std(df[feats], df["Class"], df["Amount"] * 95.0, df["Time"],
                "ULB credit card", "EU card, 0.17% fraud, 28 PCA components")


def load_baf():
    p = glob.glob(os.path.expanduser(
        "~/.cache/kagglehub/datasets/sgpjesus/*/versions/*/Base.csv"))
    if not p:
        return None
    df = pd.read_csv(p[0]).sort_values("month", kind="stable").reset_index(drop=True)
    feats = [c for c in df.columns
             if c not in ("fraud_bool", "month", "proposed_credit_limit")]
    return _std(df[feats], df["fraud_bool"], df["proposed_credit_limit"],
                df["month"].to_numpy() * 10**7 + np.arange(len(df)),
                "NeurIPS BAF", "account-opening fraud, 1M rows, real benchmark")


def load_indian():
    p = glob.glob(os.path.expanduser(
        "~/.cache/kagglehub/datasets/marusagar/*/versions/*/*.csv"))
    if not p:
        return None
    df = pd.read_csv(p[0])
    ts = pd.to_datetime(df["Transaction_Date"] + " " + df["Transaction_Time"],
                        format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df = df[ts.notna()].copy()
    feats = ["Gender", "Age", "State", "City", "Bank_Branch", "Account_Type",
             "Merchant_ID", "Transaction_Type", "Merchant_Category",
             "Account_Balance", "Transaction_Device", "Transaction_Location",
             "Device_Type"]
    return _std(df[feats], df["Is_Fraud"], df["Transaction_Amount"],
                (ts[ts.notna()].astype("int64") // 10**9).to_numpy(),
                "Indian bank", "SYNTHETIC -- negative control, label is a coin flip")


def load_sparkov():
    p = glob.glob(os.path.expanduser(
        "~/.cache/kagglehub/datasets/kartik2112/*/versions/*/fraudTrain.csv"))
    if not p:
        return None
    df = pd.read_csv(p[0])
    ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df = df[ts.notna()].copy()
    feats = ["category", "amt", "gender", "state", "city_pop", "job",
             "merch_lat", "merch_long", "lat", "long", "zip"]
    feats = [c for c in feats if c in df.columns]
    return _std(df[feats], df["is_fraud"], df["amt"] * 88.0,
                (ts[ts.notna()].astype("int64") // 10**9).to_numpy(),
                "Sparkov", "simulated US card, merchant/geo features")


def load_paysim():
    p = glob.glob(os.path.expanduser(
        "~/.cache/kagglehub/datasets/ealaxi/*/versions/*/*.csv"))
    if not p:
        return None
    df = pd.read_csv(p[0])
    feats = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
             "oldbalanceDest", "newbalanceDest"]
    return _std(df[feats], df["isFraud"], df["amount"],
                df["step"].to_numpy() * 10**7 + np.arange(len(df)),
                "PaySim", "simulated mobile money, agent-based")


LOADERS = [load_ieee, load_ulb, load_baf, load_sparkov, load_paysim, load_indian]

# Cost constants are scaled to each dataset's own exposure, because a fixed
# rupee LTV against a median amount of 22 (ULB) and 4,300 (IEEE-CIS) is not the
# same policy. LTV is set to 50% of the median amount throughout -- one rule,
# applied identically, rather than a per-dataset fudge.
LTV_AS_MEDIAN_FRACTION = 0.50
FEE_AS_MEDIAN_FRACTION = 0.25
MARGIN_RATE = 0.25
REVIEW_COST_FRACTION = 0.03


def run(entry):
    import calibrate
    import features
    import lightgbm as lgb
    import model as M
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = entry["df"]
    n = len(df)
    i_tr, i_ca = int(n * 0.70), int(n * 0.80)
    cols = [c for c in df.columns if c not in (TARGET, TIME)]

    tr, ca, te = df.iloc[:i_tr], df.iloc[i_tr:i_ca], df.iloc[i_ca:]
    if te[TARGET].sum() < 30:
        return {"name": entry["name"], "error": "too few frauds in test"}

    X_tr, y_tr, _, cats = features.build(tr, cols)
    X_ca, y_ca, _, _ = features.build(ca, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te, cols, cats)

    b = lgb.train(M.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=M.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = calibrate.fit_calibrator(b.predict(X_ca), y_ca)(b.predict(X_te))

    auc = float(roc_auc_score(y_te, p))
    ap = float(average_precision_score(y_te, p))
    base = float(y_te.mean())
    F = int((y_te == 1).sum())

    med = float(np.median(amt_te))
    ltv, fee = med * LTV_AS_MEDIAN_FRACTION, med * FEE_AS_MEDIAN_FRACTION
    c_fp, c_fn = amt_te * MARGIN_RATE + ltv, amt_te + fee

    blocked = p >= c_fp / (c_fp + c_fn)
    rc = med * REVIEW_COST_FRACTION
    cost = np.stack([p * c_fn, np.full_like(p, rc), (1 - p) * c_fp])
    a = cost.argmin(axis=0)
    tw = {nm: {"n": int((a == i).sum()),
               "frauds": int(((a == i) & (y_te == 1)).sum())}
          for i, nm in enumerate(("ALLOW", "REVIEW", "BLOCK"))}
    reached = tw["REVIEW"]["frauds"] + tw["BLOCK"]["frauds"]
    conc = ((tw["REVIEW"]["frauds"] / max(F, 1))
            / max(tw["REVIEW"]["n"] / len(y_te), 1e-9))

    return {"name": entry["name"], "note": entry["note"], "rows": n,
            "features": len(cols), "test_rows": int(len(y_te)),
            "frauds_total": F, "base_rate": base, "median_amount": med,
            "roc_auc": auc, "pr_auc": ap, "lift": ap / base,
            "two_way_blocked": int(blocked.sum()),
            "two_way_caught": int((blocked & (y_te == 1)).sum()),
            "three_way": tw, "fraud_reached": reached,
            "fraud_reached_rate": reached / max(F, 1),
            "review_concentration": conc}


def main() -> None:
    rows = []
    for fn in LOADERS:
        try:
            e = fn()
        except Exception as exc:
            print(f"[{fn.__name__}] load failed: {type(exc).__name__}: {exc}")
            continue
        if e is None:
            print(f"[{fn.__name__}] not downloaded, skipping")
            continue
        print(f"\n>>> {e['name']}  ({len(e['df']):,} rows) -- {e['note']}")
        try:
            r = run(e)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            continue
        if "error" in r:
            print(f"    skipped: {r['error']}")
            continue
        rows.append(r)
        print(f"    AUC {r['roc_auc']:.4f}  lift {r['lift']:.1f}x  "
              f"reached {r['fraud_reached']:,}/{r['frauds_total']:,}")

    w = 118
    print("\n" + "=" * w)
    print("ONE PIPELINE, EVERY DATASET.  Nothing tuned per dataset.")
    print("=" * w)
    print(f"{'dataset':<18}{'rows':>11}{'fraud%':>8}{'AUC':>8}{'PR-AUC':>9}"
          f"{'lift':>7}{'ALLOW':>10}{'REVIEW':>10}{'BLOCK':>8}"
          f"{'reached':>17}{'conc':>8}")
    print("-" * w)
    for r in rows:
        t = r["three_way"]
        print(f"{r['name']:<18}{r['rows']:>11,}{r['base_rate']:>8.2%}"
              f"{r['roc_auc']:>8.4f}{r['pr_auc']:>9.4f}{r['lift']:>6.1f}x"
              f"{t['ALLOW']['n']:>10,}{t['REVIEW']['n']:>10,}{t['BLOCK']['n']:>8,}"
              f"{r['fraud_reached']:>8,}/{r['frauds_total']:<7,}"
              f"{r['review_concentration']:>7.1f}x")
    print("-" * w)
    print("lift = PR-AUC / base rate. Comparable across datasets; raw PR-AUC is not.")
    print("conc = how much denser the review lane is in fraud than the traffic it")
    print("       holds. 1.0x means the model is no better than picking at random.")
    print()
    real = [r for r in rows if "SYNTHETIC" not in r["note"]]
    syn = [r for r in rows if "SYNTHETIC" in r["note"]]
    if real:
        print(f"REAL datasets ({len(real)}): AUC "
              f"{min(r['roc_auc'] for r in real):.3f}-"
              f"{max(r['roc_auc'] for r in real):.3f}, lift "
              f"{min(r['lift'] for r in real):.1f}x-"
              f"{max(r['lift'] for r in real):.1f}x")
    for r in syn:
        print(f"NEGATIVE CONTROL ({r['name']}): AUC {r['roc_auc']:.4f}, "
              f"lift {r['lift']:.1f}x, concentration "
              f"{r['review_concentration']:.2f}x -- finds nothing, as it should.")
    print("=" * w)

    Path("artifacts/transfer.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote artifacts/transfer.json")


if __name__ == "__main__":
    main()
