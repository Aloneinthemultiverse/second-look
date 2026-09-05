"""Every Indian dataset, cleaned the same way, through the same pipeline.

Four public Indian transaction datasets were found. Each is loaded, stripped of
leakage, trained on a temporal split, calibrated, and passed through the same
cost-optimal decision layer as the shipped system. One table at the end.

WHY THE CLEANING STEP EXISTS. Two of these score near-perfectly as published,
and both times the cause was a column that does not exist at decision time:

  Fraud_Reason        populated only on frauds -- the label wearing a hat
  Card_Status         Blocked/Expired/Lost were 100% fraud, deterministic
  Transaction_Status  Declined 59.6% fraud vs Successful 1.4% -- you only know
                      a payment was declined AFTER something declined it

That last one is the interesting trap. It looks like an ordinary feature. It is
an outcome. Any dataset with a status, reason, or disposition column should be
assumed guilty until checked, and this script checks: it reports each dataset
BEFORE and AFTER cleaning so the size of the leak is visible rather than
quietly corrected.

WHAT TO EXPECT. Three of these four are noise (AUC ~0.50) and cleaning cannot
help -- there is nothing under the leak because nothing was ever generated.
Only one survives, and it survives at AUC ~0.85, which is a believable number
rather than a suspicious one.
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

import calibrate

CACHE = os.path.expanduser("~/.cache/kagglehub/datasets")

# Columns that are outcomes, identifiers or personal data -- never features.
# Matched case-insensitively as substrings.
# Named explicitly. An earlier version matched the substring "status", which
# also caught Marital_Status and Merchant_Status -- ordinary features, not
# outcomes. Guessing at leaks by substring removes real signal, so each one is
# justified by a measured fraud-rate difference instead.
LEAK_EXACT = {"transaction_status", "fraud_reason", "card_status",
              "kyc_status", "payment_status", "order_status", "dispute_status"}
ID_PATTERNS = ["_id", "id_", "customer_name", "merchant_name", "_email",
               "_contact", "description"]


def find(pattern):
    hits = glob.glob(f"{CACHE}/{pattern}")
    return hits[0] if hits else None


def load_marusagar():
    p = find("marusagar/*/versions/*/*.csv")
    if not p:
        return None
    d = pd.read_csv(p)
    ts = pd.to_datetime(d["Transaction_Date"] + " " + d["Transaction_Time"],
                        format="%d-%m-%Y %H:%M:%S", errors="coerce")
    d = d[ts.notna()].copy()
    d["_t"], d["_y"] = ts[ts.notna()], d["Is_Fraud"]
    d["_amt"] = d["Transaction_Amount"]
    return "marusagar/bank-transaction-fraud", d, ["Transaction_Date",
                                                   "Transaction_Time", "Is_Fraud"]


def load_upi():
    p = find("skullagos5246/*/versions/*/*.csv")
    if not p:
        return None
    d = pd.read_csv(p)
    ts = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d[ts.notna()].copy()
    d["_t"], d["_y"] = ts[ts.notna()], d["fraud_flag"]
    d["_amt"] = d["amount (INR)"]
    return "skullagos/upi-transactions-2024", d, ["timestamp", "fraud_flag"]


def load_belbino():
    p = find("belbino/*/versions/*/*.csv")
    if not p:
        return None
    d = pd.read_csv(p)
    ts = pd.to_datetime(d["transaction_date"].astype(str) + " "
                        + d["transaction_time"].astype(str),
                        errors="coerce", dayfirst=True)
    d = d[ts.notna()].copy()
    d["_t"], d["_y"] = ts[ts.notna()], d["is_fraud"]
    d["_amt"] = d["transaction_amount"]
    return "belbino/indian-banking-2019-2024", d, ["transaction_date",
                                                   "transaction_time", "is_fraud"]


def load_jatin():
    b = find("jatinkhandelwal112/*/versions/*")
    if not b:
        return None
    b = os.path.dirname(glob.glob(f"{b}/*.csv")[0]) if os.path.isdir(b) else b
    t = pd.read_csv(f"{b}/Transaction_Data_250k.csv")
    d = (t.merge(pd.read_csv(f"{b}/Cusmtomer_data.csv"), on="Customer_ID", how="left")
          .merge(pd.read_csv(f"{b}/Cards_Data.csv"), on="Card_ID", how="left",
                 suffixes=("", "_card"))
          .merge(pd.read_csv(f"{b}/merchant_table.csv"), on="Merchant_ID",
                 how="left", suffixes=("", "_m")))
    ts = pd.to_datetime(d["Transaction_Date"].astype(str) + " "
                        + d["Transaction_Time"].astype(str), errors="coerce")
    d = d[ts.notna()].copy()
    d["_t"], d["_y"] = ts[ts.notna()], d["Fraud_Flag"]
    d["_amt"] = d["Transaction_Amount"]
    # Blocked/Expired/Lost cards are labelled fraud with probability 1.0.
    # Dropping the column is not enough; drop the rows the rule fired on.
    if "Card_Status" in d.columns:
        d = d[d["Card_Status"] == "Active"]
    return "jatinkhandelwal/indian-financial-fraud", d, ["Transaction_Date",
                                                         "Transaction_Time",
                                                         "Fraud_Flag"]


LOADERS = [load_marusagar, load_upi, load_belbino, load_jatin]


def split_cols(d, always_drop):
    """Separate honest features from leaks and identifiers."""
    base = [c for c in d.columns
            if c not in set(always_drop) | {"_t", "_y", "_amt"}]
    leaks = [c for c in base if c.lower() in LEAK_EXACT]
    # High cardinality only condemns a column if it is NOT numeric. An amount
    # is near-unique in a large file and is the most important feature there
    # is; an earlier version deleted it as an identifier and cost 0.24 AUC.
    ids = [c for c in base
           if c not in leaks
           and (any(k in c.lower() for k in ID_PATTERNS)
                or (d[c].dtype == object and d[c].nunique() > 0.5 * len(d)))]
    return [c for c in base if c not in leaks + ids], leaks, ids


def fit_predict(d, cols):
    X = d[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    y = d["_y"].to_numpy().astype(int)
    n = len(d)
    i_tr, i_ca = int(n * 0.70), int(n * 0.80)
    if y[i_ca:].sum() < 25:
        return None
    b = lgb.train({"objective": "binary", "learning_rate": 0.05,
                   "num_leaves": 64, "min_data_in_leaf": 100,
                   "verbosity": -1, "seed": 42},
                  lgb.Dataset(X[:i_tr], label=y[:i_tr]), num_boost_round=400)
    p = calibrate.fit_calibrator(b.predict(X[i_tr:i_ca]),
                                 y[i_tr:i_ca])(b.predict(X[i_ca:]))
    return p, y[i_ca:], d["_amt"].to_numpy(float)[i_ca:]


def decide(p, y, amt):
    med = float(np.median(amt))
    ltv, fee, rc = med * 0.5, med * 0.25, med * 0.05
    c_fp, c_fn = amt * 0.25 + ltv, amt + fee
    blocked = p >= c_fp / (c_fp + c_fn)
    a = np.stack([p * c_fn, np.full_like(p, rc), (1 - p) * c_fp]).argmin(axis=0)
    F = int((y == 1).sum())
    tw = {nm: {"n": int((a == i).sum()),
               "frauds": int(((a == i) & (y == 1)).sum())}
          for i, nm in enumerate(("ALLOW", "REVIEW", "BLOCK"))}
    reached = tw["REVIEW"]["frauds"] + tw["BLOCK"]["frauds"]
    conc = ((tw["REVIEW"]["frauds"] / max(F, 1))
            / max(tw["REVIEW"]["n"] / len(y), 1e-9))
    return {"frauds_total": F, "median_amount": med,
            "two_way_blocked": int(blocked.sum()),
            "two_way_caught": int((blocked & (y == 1)).sum()),
            "three_way": tw, "reached": reached,
            "reached_rate": reached / max(F, 1), "review_concentration": conc}


def main() -> None:
    out = []
    for fn in LOADERS:
        try:
            loaded = fn()
        except Exception as exc:
            print(f"[{fn.__name__}] failed: {type(exc).__name__}: {exc}")
            continue
        if loaded is None:
            print(f"[{fn.__name__}] not downloaded, skipping")
            continue
        name, d, always = loaded
        d = d.sort_values("_t", kind="stable").reset_index(drop=True)
        good, leaks, ids = split_cols(d, always)
        print(f"\n>>> {name}")
        print(f"    {len(d):,} rows, fraud {d['_y'].mean():.2%}")
        print(f"    features kept {len(good)} | leaks removed {leaks} "
              f"| ids removed {len(ids)}")

        row = {"name": name, "rows": int(len(d)),
               "fraud_rate": float(d["_y"].mean()),
               "leaks_removed": leaks, "features": len(good)}

        if leaks:
            r = fit_predict(d, good + leaks)
            if r:
                row["auc_with_leaks"] = float(roc_auc_score(r[1], r[0]))
                print(f"    WITH leaks:  AUC {row['auc_with_leaks']:.4f}  <- inflated")

        r = fit_predict(d, good)
        if not r:
            print("    too few test frauds, skipped")
            continue
        p, y, amt = r
        row["auc"] = float(roc_auc_score(y, p))
        row["pr_auc"] = float(average_precision_score(y, p))
        row["base_rate"] = float(y.mean())
        row["lift"] = row["pr_auc"] / row["base_rate"]
        row.update(decide(p, y, amt))
        print(f"    CLEANED:     AUC {row['auc']:.4f}  lift {row['lift']:.1f}x  "
              f"caught {row['reached']:,}/{row['frauds_total']:,}")
        out.append(row)

    w = 112
    print("\n" + "=" * w)
    print("EVERY INDIAN DATASET, CLEANED, THROUGH THE SAME PIPELINE")
    print("=" * w)
    print(f"{'dataset':<38}{'AUC':>8}{'lift':>7}{'ALLOW':>9}{'REVIEW':>9}"
          f"{'BLOCK':>8}{'caught':>16}{'conc':>8}")
    print("-" * w)
    tot_c = tot_f = 0
    for r in out:
        t = r["three_way"]
        tot_c += r["reached"]
        tot_f += r["frauds_total"]
        print(f"{r['name']:<38}{r['auc']:>8.4f}{r['lift']:>6.1f}x"
              f"{t['ALLOW']['n']:>9,}{t['REVIEW']['n']:>9,}{t['BLOCK']['n']:>8,}"
              f"{r['reached']:>7,}/{r['frauds_total']:<8,}"
              f"{r['review_concentration']:>7.1f}x")
    print("-" * w)
    print(f"{'TOTAL across Indian datasets':<38}{'':>21}{'':>17}"
          f"{tot_c:>7,}/{tot_f:<8,}{tot_c / max(tot_f, 1):>7.1%}")
    print("-" * w)
    print("conc = how much denser the review lane is in fraud than the traffic it")
    print("holds. 1.0x means no better than picking at random -- the model found")
    print("nothing, and the frauds in the review lane are there by volume alone.")
    print()
    real = [r for r in out if r["auc"] > 0.65]
    print(f"{len(real)} of {len(out)} Indian datasets carry usable signal.")
    for r in out:
        if r["auc"] <= 0.65:
            print(f"  {r['name']}: AUC {r['auc']:.4f} -- noise, not salvageable")
    print("=" * w)

    Path("artifacts/indian_all.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote artifacts/indian_all.json")


if __name__ == "__main__":
    main()
