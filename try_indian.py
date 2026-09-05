"""Point the shipped pipeline at an Indian bank dataset, following ADAPTING.md.

This is a test of a CLAIM. The README says the decision layer transfers to other
data with four required columns and three config changes. Claims in this repo get
measured, not asserted, so this runs the actual thing end to end.

DATASET: marusagar/bank-transaction-fraud-detection (Kaggle)
  200,000 Indian bank transactions, 24 columns, 5.04% fraud, all amounts in INR.

WHAT HAS TO CHANGE, and it is exactly what ADAPTING.md says:
  1. rename 4 columns to the names the pipeline expects
  2. USD_TO_INR = 1.0, because these amounts are ALREADY rupees. Leaving it at
     88 would inflate every amount 88x and every cost figure with it.
  3. the cost constants stay as-is here only because we have no better estimate
     for this merchant; on real deployment they would be replaced.

WHAT WE DROP, and why:
  Customer_Name, Customer_Contact, Customer_Email  -- personal data, and
      near-unique per row, so they are identifiers rather than features.
  Customer_ID, Transaction_Description             -- same, high-cardinality ids.
  Transaction_Currency                             -- constant (INR), no signal.

READ THE RESULT CAREFULLY. The column distributions in this dataset are close to
perfectly uniform -- gender 50/50, five transaction types at 20% each, six
merchant categories at 16.7% each. That is the signature of SYNTHETIC data. If a
model finds no signal here, the honest conclusion is that there is no signal to
find, not that the pipeline failed. A pipeline that reports AUC ~0.5 on noise is
behaving correctly; one that reports 0.9 would be the thing to worry about.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = (Path.home() / ".cache" / "kagglehub" / "datasets" / "marusagar"
       / "bank-transaction-fraud-detection" / "versions" / "1"
       / "Bank_Transaction_Fraud_Detection.csv")

DROP = ["Customer_Name", "Customer_Contact", "Customer_Email", "Customer_ID",
        "Transaction_Description", "Transaction_Currency"]
RENAME = {"Is_Fraud": "isFraud", "Transaction_Amount": "TransactionAmt",
          "Transaction_ID": "TransactionID"}


def prepare() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    print(f"loaded {len(df):,} rows, {len(df.columns)} columns")

    # TransactionDT: the pipeline only needs an increasing number. Build one
    # from date + time; the ordering is what the temporal split relies on.
    ts = pd.to_datetime(df["Transaction_Date"] + " " + df["Transaction_Time"],
                        format="%d-%m-%Y %H:%M:%S", errors="coerce")
    bad = int(ts.isna().sum())
    if bad:
        print(f"  {bad:,} unparseable timestamps dropped")
        df, ts = df[ts.notna()].copy(), ts[ts.notna()]
    df["TransactionDT"] = (ts.astype("int64") // 10**9).to_numpy()

    df = df.drop(columns=DROP + ["Transaction_Date", "Transaction_Time"])
    df = df.rename(columns=RENAME)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    print(f"  fraud rate {df['isFraud'].mean():.2%}  "
          f"({int(df['isFraud'].sum()):,} positives)")
    print(f"  {len(df.columns) - 4} feature columns after dropping PII and ids")
    print(f"  amount range Rs {df['TransactionAmt'].min():,.0f} - "
          f"Rs {df['TransactionAmt'].max():,.0f}")
    return df


def main() -> None:
    import config
    config.USD_TO_INR = 1.0          # ADAPTING.md step 2: already rupees

    import calibrate
    import data
    import features
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = prepare()

    # --- the pipeline's own machinery, unmodified ---------------------------
    audit = data.audit_leakage(df)
    tr, ca, te = data.temporal_split(df)
    print(f"\ntemporal split  train {len(tr):,}  calib {len(ca):,}  "
          f"test {len(te):,}")
    print(f"  fraud rate     train {tr['isFraud'].mean():.2%}  "
          f"calib {ca['isFraud'].mean():.2%}  test {te['isFraud'].mean():.2%}")

    cols = audit["online"]
    print(f"  {len(cols)} columns classified online: {', '.join(cols)}")

    X_tr, y_tr, _, cats = features.build(tr, cols)
    X_ca, y_ca, _, _ = features.build(ca, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te, cols, cats)

    print("\ntraining ...")
    import model as M
    booster = lgb.train(M.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                        num_boost_round=M.NUM_ROUNDS,
                        valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    p = calibrate.fit_calibrator(booster.predict(X_ca), y_ca)(
        booster.predict(X_te))

    auc = float(roc_auc_score(y_te, p))
    ap = float(average_precision_score(y_te, p))

    # --- the decision layer, unchanged from the shipped system --------------
    c_fp = amt_te * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amt_te + config.CHARGEBACK_FEE_INR
    blocked = p >= c_fp / (c_fp + c_fn)
    tp = int((blocked & (y_te == 1)).sum())
    F = int((y_te == 1).sum())

    w = 78
    print("\n" + "=" * w)
    print("SHIPPED PIPELINE, INDIAN BANK DATA, NOTHING BUT CONFIG CHANGED")
    print("=" * w)
    print(f"{'test rows':<34}{len(y_te):>44,}")
    print(f"{'frauds in test':<34}{F:>44,}")
    print(f"{'ROC-AUC':<34}{auc:>44.4f}")
    print(f"{'PR-AUC':<34}{ap:>44.4f}")
    print(f"{'PR-AUC of random guessing':<34}{y_te.mean():>44.4f}")
    print(f"{'frauds blocked':<34}{tp:>37,} / {F:,}")
    print(f"{'block rate':<34}{blocked.mean():>44.2%}")
    print("-" * w)

    # --- three-way ALLOW / REVIEW / BLOCK, same Chow rule as the shipped run -
    from chow_band import chow_actions
    print("TWO-WAY (Elkan tau*)")
    for nm, m in (("ALLOW", ~blocked), ("BLOCK", blocked)):
        f = int((m & (y_te == 1)).sum())
        print(f"  {nm:<8}{int(m.sum()):>10,} txns{m.mean():>9.2%}"
              f"   frauds {f:>6,}{f / max(F, 1):>8.1%}")
    three = {}
    for rc in (25.0, 150.0, 400.0):
        a, _ = chow_actions(p, amt_te, rc)
        print(f"\nTHREE-WAY (Chow, review cost Rs {rc:,.0f})")
        row = {}
        for i, nm in enumerate(("ALLOW", "REVIEW", "BLOCK")):
            m = a == i
            f = int((m & (y_te == 1)).sum())
            row[nm] = {"n": int(m.sum()), "frauds": f}
            print(f"  {nm:<8}{int(m.sum()):>10,} txns{m.mean():>9.2%}"
                  f"   frauds {f:>6,}{f / max(F, 1):>8.1%}")
        reached = row["REVIEW"]["frauds"] + row["BLOCK"]["frauds"]
        row["fraud_reached"] = reached
        three[str(rc)] = row
        print(f"  {'-> reached':<8}{'':>10} {'':>13}   frauds {reached:>6,}"
              f"{reached / max(F, 1):>8.1%}")
    print("-" * w)

    lift = ap / y_te.mean()
    if auc < 0.55:
        verdict = (
            "NO SIGNAL. The pipeline ran end to end on a new dataset with only\n"
            "the documented config changes -- that part of the claim holds. But\n"
            "this data has nothing to learn: ROC-AUC at chance, PR-AUC at the\n"
            "base rate. The near-uniform column distributions said as much\n"
            "before training. A pipeline reporting 0.5 on synthetic noise is\n"
            "behaving correctly; one reporting 0.9 would mean it was leaking.")
    elif lift < 2:
        verdict = ("WEAK SIGNAL. It runs, and it finds a little -- but not enough\n"
                   "to justify a blocking policy on this data.")
    else:
        verdict = (f"REAL SIGNAL. PR-AUC is {lift:.1f}x the base rate. The method\n"
                   "transfers to Indian bank data, not just IEEE-CIS.")
    print(verdict)
    print("=" * w)

    Path("artifacts/try_indian.json").write_text(json.dumps({
        "source": "marusagar/bank-transaction-fraud-detection",
        "rows": int(len(df)), "test_rows": int(len(y_te)),
        "fraud_rate": float(df["isFraud"].mean()),
        "online_columns": cols, "roc_auc": auc, "pr_auc": ap,
        "base_rate": float(y_te.mean()), "lift_over_base": float(lift),
        "frauds_blocked": tp, "frauds_total": F,
        "block_rate": float(blocked.mean()), "three_way": three,
        "config_changes": ["USD_TO_INR = 1.0", "4 columns renamed",
                           "6 PII/id columns dropped"],
        "verdict": verdict.replace("\n", " "),
    }, indent=2), encoding="utf-8")
    print("\nwrote artifacts/try_indian.json")


if __name__ == "__main__":
    main()
