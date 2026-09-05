"""Can the best Indian dataset be cleaned into something usable?

Five Indian datasets were tested and all five failed: three were noise
(AUC 0.50-0.55) and two leaked (0.9999 from a target derived from its own
features, 0.9754 from a deterministic rule). This takes the most promising of
them and asks whether removing the leak leaves anything real underneath.

DATASET: jatinkhandelwal112/indian-financial-fraud-dataset
  250,000 transactions across 24,902 customers -- about 10 each. That relational
  structure is why this one is worth salvaging: the other Indian datasets
  generated every row independently, so no customer history existed at all.

THE LEAK, and how it was found. Fraud rate by Card_Status:

    Active   3.49%
    Blocked  100.00%
    Expired  100.00%
    Lost     100.00%

Every non-Active card is fraud with probability exactly 1.0. A model scoring
0.9754 on this is not detecting fraud, it is reading Card_Status. `Fraud_Reason`
is worse -- it is populated only for frauds, so it is the label in disguise.

THE CLEANUP:
  1. drop Fraud_Reason           -- present only on positives
  2. drop Card_Status            -- deterministic label rule
  3. keep only Active cards      -- removes the rule's effect entirely rather
                                    than trusting that dropping the column is
                                    enough; correlated proxies could survive
  4. add customer-history features, built STRICTLY from earlier transactions

Step 4 is the point. A real fraud system's strongest features are behavioural:
how fast is this customer transacting, is this amount unusual for them, is this
a new device. None of that exists as a column -- it has to be derived, and it
has to be derived without looking forward. Every feature below uses only rows
that precede the current one for that customer.

WHAT COUNTS AS SUCCESS. Not a high AUC -- we already know a high AUC here means
a leak. Success is a MODEST lift that survives removing the deterministic rule,
which would mean the generator simulated something behaviourally coherent.
AUC near 0.5 after cleaning means the remaining fraud is unlearnable noise and
the 0.9754 was the rule and nothing else.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = os.path.expanduser(
    "~/.cache/kagglehub/datasets/jatinkhandelwal112/"
    "indian-financial-fraud-dataset/versions/1")
LEAK = ["Fraud_Reason", "Card_Status", "Transaction_Status", "Customer_ID_card"]


def load():
    t = pd.read_csv(f"{BASE}/Transaction_Data_250k.csv")
    cust = pd.read_csv(f"{BASE}/Cusmtomer_data.csv")
    card = pd.read_csv(f"{BASE}/Cards_Data.csv")
    merch = pd.read_csv(f"{BASE}/merchant_table.csv")

    df = (t.merge(cust, on="Customer_ID", how="left")
           .merge(card, on="Card_ID", how="left", suffixes=("", "_card"))
           .merge(merch, on="Merchant_ID", how="left", suffixes=("", "_m")))

    ts = pd.to_datetime(df["Transaction_Date"].astype(str) + " "
                        + df["Transaction_Time"].astype(str), errors="coerce")
    df = df[ts.notna()].copy()
    df["_t"] = ts[ts.notna()]
    return df.sort_values("_t", kind="stable").reset_index(drop=True)


def behavioural(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer history features. Strictly backward-looking.

    shift(1) inside each customer group is what makes this safe: row i sees the
    aggregate of rows 0..i-1 for that customer and nothing else. Without the
    shift, a customer's own current transaction leaks into its own feature and
    the whole exercise becomes another version of the bug we are removing.
    """
    g = df.groupby("Customer_ID", sort=False)
    df["cust_txn_seq"] = g.cumcount()
    prev_mean = g["Transaction_Amount"].apply(
        lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    prev_max = g["Transaction_Amount"].apply(
        lambda s: s.shift(1).expanding().max()).reset_index(level=0, drop=True)
    df["amt_vs_cust_mean"] = df["Transaction_Amount"] / prev_mean.replace(0, np.nan)
    df["amt_vs_cust_max"] = df["Transaction_Amount"] / prev_max.replace(0, np.nan)
    df["secs_since_last"] = g["_t"].diff().dt.total_seconds()
    df["amt_over_limit"] = df["Transaction_Amount"] / df["Credit_Limit"].replace(0, np.nan)

    # device/merchant novelty for this customer, again backward only
    for col, name in (("Device_Type", "new_device"),
                      ("Merchant_ID", "new_merchant")):
        seen = g[col].apply(lambda s: s.shift(1).expanding().apply(
            lambda x: 0, raw=False)) if False else None
        df[name] = (~df.duplicated(subset=["Customer_ID", col], keep="first")).astype(int)
    return df


def run(df, cols, tag, i=None):
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    y = df["Fraud_Flag"].to_numpy()
    i = i or int(len(X) * 0.8)
    if y[i:].sum() < 25:
        print(f"  {tag}: too few test positives")
        return None
    m = lgb.train({"objective": "binary", "learning_rate": 0.05,
                   "num_leaves": 64, "min_data_in_leaf": 100,
                   "verbosity": -1, "seed": 42},
                  lgb.Dataset(X[:i], label=y[:i]), num_boost_round=400)
    p = m.predict(X[i:])
    base = float(y[i:].mean())
    auc = float(roc_auc_score(y[i:], p))
    ap = float(average_precision_score(y[i:], p))
    print(f"  {tag:<42} AUC {auc:.4f}  PR-AUC {ap:.4f}  "
          f"base {base:.4f}  lift {ap / base:>5.1f}x")
    return {"tag": tag, "rows": int(len(X)), "features": int(X.shape[1]),
            "test_positives": int(y[i:].sum()), "roc_auc": auc,
            "pr_auc": ap, "base_rate": base, "lift": ap / base,
            "top_features": list(pd.Series(
                m.feature_importance("gain"), index=X.columns)
                .sort_values(ascending=False).head(6).index)}


def main() -> None:
    df = load()
    print(f"loaded {len(df):,} transactions, "
          f"{df['Customer_ID'].nunique():,} customers "
          f"({len(df) / df['Customer_ID'].nunique():.1f} each)")
    print(f"fraud rate {df['Fraud_Flag'].mean():.3%}\n")

    drop = ["Fraud_Flag", "_t", "Transaction_Date", "Transaction_Time",
            "Transaction_ID", "Customer_Name", "Merchant_Name",
            "Customer_ID", "Card_ID", "Merchant_ID"]
    rows = []

    print("STAGE 1 -- as published, leaks included")
    cols = [c for c in df.columns if c not in drop]
    rows.append(run(df, cols, "everything (Fraud_Reason + Card_Status)"))

    print("\nSTAGE 2 -- drop the two leaking columns")
    cols2 = [c for c in cols if c not in LEAK]
    rows.append(run(df, cols2, "leaks dropped (4), all cards"))

    print("\nSTAGE 3 -- Active cards only, so the rule cannot act through proxies")
    act = df[df["Card_Status"] == "Active"].reset_index(drop=True)
    print(f"  {len(act):,} rows kept, fraud rate now {act['Fraud_Flag'].mean():.3%}")
    rows.append(run(act, cols2, "Active only, all 4 leaks dropped"))

    print("\nSTAGE 4 -- add backward-looking customer-history features")
    act = behavioural(act)
    extra = ["cust_txn_seq", "amt_vs_cust_mean", "amt_vs_cust_max",
             "secs_since_last", "amt_over_limit", "new_device", "new_merchant"]
    rows.append(run(act, cols2 + extra, "Active + behavioural history"))

    rows = [r for r in rows if r]
    w = 92
    print("\n" + "=" * w)
    print("CAN THE INDIAN DATASET BE CLEANED INTO SOMETHING USABLE?")
    print("=" * w)
    print(f"{'stage':<44}{'AUC':>9}{'PR-AUC':>10}{'lift':>9}{'rows':>12}")
    print("-" * w)
    for r in rows:
        print(f"{r['tag']:<44}{r['roc_auc']:>9.4f}{r['pr_auc']:>10.4f}"
              f"{r['lift']:>8.1f}x{r['rows']:>12,}")
    print("-" * w)
    print("for reference:  IEEE-CIS 0.8870 / 14.8x    NeurIPS BAF 0.8892 / 13.3x")
    print("                pure noise 0.5000 / 1.0x")
    print()

    final = rows[-1]
    if final["roc_auc"] > 0.75:
        v = ("USABLE. Real behavioural signal survives removing the rule, so the\n"
             "generator simulated coherent customer behaviour. This can serve as\n"
             "an Indian-context demonstration set, with the caveat that it is\n"
             "simulated and its ceiling is the generator's own rules.")
    elif final["roc_auc"] > 0.60:
        v = ("PARTIALLY USABLE. Something survives, well below the real datasets.\n"
             "Reportable as a demonstration, not as evidence of performance.")
    else:
        v = ("NOT SALVAGEABLE. Once the deterministic rule is removed there is\n"
             "nothing underneath. The 0.9754 was the rule and nothing else, and\n"
             "no cleaning recovers signal that was never generated.")
    print(v)
    print("=" * w)

    Path("artifacts/clean_indian.json").write_text(json.dumps(
        {"source": "jatinkhandelwal112/indian-financial-fraud-dataset",
         "leaks_removed": LEAK, "stages": rows, "verdict": v.replace("\n", " ")},
        indent=2), encoding="utf-8")
    print("\nwrote artifacts/clean_indian.json")


if __name__ == "__main__":
    main()
