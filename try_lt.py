"""Real Indian financial data, at last -- L&T vehicle loan defaults.

Eight datasets have been tested. The three Indian ones were all generated: AUC
0.49, 0.53, and 0.85 only after removing three leaks. On that basis this project
was about to claim "there is no real Indian dataset". That claim is too broad,
and this is the check before saying it out loud.

DATASET: mamtadhaker/lt-vehicle-loan-default-prediction
  L&T Financial Services' actual loan book, released for a hackathon. It is
  unmistakably real: Aadhar_flag, PAN_flag, VoterID_flag, PERFORM_CNS.SCORE
  (the CIBIL bureau score), branch ids, pincodes, state ids. Nobody generates
  a column called PERFORM_CNS.SCORE.DESCRIPTION.

WHAT IS DIFFERENT ABOUT IT. This is LOAN DEFAULT, not transaction fraud. The
event is a borrower failing to repay, not a stolen card. That matters for how
the result may be quoted: it does not show the detector transfers to Indian
FRAUD, because no such public dataset exists. What it does show is whether the
decision layer works on a real Indian loss event, which is a weaker but honest
claim -- and loss_types.py already argues the layer is loss-type agnostic, so
this is that argument meeting real data instead of assumed costs.

COSTS. A default costs the outstanding exposure; refusing a good borrower costs
the margin on a loan you did not write plus the customer. `disbursed_amount` is
the exposure and is already in rupees, so USD_TO_INR does not apply.

TEMPORAL SPLIT on DisbursalDate. The competition shipped its own train/test
split, but the test file has no labels, so train.csv is split by date here --
earlier disbursals predict later ones, which is the real deployment order.
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

SRC = glob.glob(os.path.expanduser(
    "~/.cache/kagglehub/datasets/mamtadhaker/*/versions/*/train.csv"))[0]

# Identifiers. branch/supplier/state/manufacturer are legitimately predictive
# (some branches really do lend worse), so they stay; only row-level keys go.
DROP_IDS = ["UniqueID", "Current_pincode_ID", "Employee_code_ID"]


def duration_to_months(s):
    """'1yrs 11mon' -> 23. The file ships tenure as a formatted string."""
    x = s.str.extract(r"(?P<y>\d+)yrs\s*(?P<m>\d+)mon")
    return x["y"].astype(float) * 12 + x["m"].astype(float)


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"loaded {len(df):,} loans, {len(df.columns)} columns")

    dis = pd.to_datetime(df["DisbursalDate"], errors="coerce", dayfirst=True)
    df = df[dis.notna()].copy()
    df["_t"] = dis[dis.notna()]
    df = df.sort_values("_t", kind="stable").reset_index(drop=True)

    dob = pd.to_datetime(df["Date.of.Birth"], errors="coerce", dayfirst=True)
    df["age_years"] = (df["_t"] - dob).dt.days / 365.25
    df.loc[(df["age_years"] < 18) | (df["age_years"] > 90), "age_years"] = np.nan
    for c in ("AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH"):
        df[c] = duration_to_months(df[c].astype(str))

    y = df["loan_default"].to_numpy().astype(int)
    amt = df["disbursed_amount"].to_numpy(float)
    print(f"default rate {y.mean():.2%} ({y.sum():,} defaults)")
    print(f"disbursed amount median Rs {np.median(amt):,.0f}, "
          f"max Rs {amt.max():,.0f}")
    print(f"disbursals span {df['_t'].min():%Y-%m-%d} to {df['_t'].max():%Y-%m-%d}")

    cols = [c for c in df.columns
            if c not in DROP_IDS + ["loan_default", "_t", "Date.of.Birth",
                                    "DisbursalDate"]]
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    print(f"{len(cols)} features")

    n = len(df)
    i_tr, i_ca = int(n * 0.70), int(n * 0.80)
    print(f"\ntemporal split  train {i_tr:,}  calib {i_ca - i_tr:,}  "
          f"test {n - i_ca:,}")
    print(f"  default rate  train {y[:i_tr].mean():.2%}  "
          f"calib {y[i_tr:i_ca].mean():.2%}  test {y[i_ca:].mean():.2%}")

    b = lgb.train({"objective": "binary", "learning_rate": 0.05,
                   "num_leaves": 64, "min_data_in_leaf": 100,
                   "feature_fraction": 0.8, "bagging_fraction": 0.8,
                   "bagging_freq": 1, "verbosity": -1, "seed": 42},
                  lgb.Dataset(X[:i_tr], label=y[:i_tr]),
                  num_boost_round=600,
                  valid_sets=[lgb.Dataset(X[i_tr:i_ca], label=y[i_tr:i_ca])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = calibrate.fit_calibrator(b.predict(X[i_tr:i_ca]),
                                 y[i_tr:i_ca])(b.predict(X[i_ca:]))
    yt, at = y[i_ca:], amt[i_ca:]
    F = int(yt.sum())
    auc = float(roc_auc_score(yt, p))
    ap = float(average_precision_score(yt, p))
    base = float(yt.mean())

    # Same cost rule as every other dataset in the transfer study.
    med = float(np.median(at))
    ltv, fee, rc = med * 0.5, med * 0.25, med * 0.05
    c_fp, c_fn = at * 0.25 + ltv, at + fee
    blocked = p >= c_fp / (c_fp + c_fn)
    a = np.stack([p * c_fn, np.full_like(p, rc), (1 - p) * c_fp]).argmin(axis=0)
    tw = {nm: {"n": int((a == i).sum()),
               "frauds": int(((a == i) & (yt == 1)).sum())}
          for i, nm in enumerate(("ALLOW", "REVIEW", "BLOCK"))}
    reached = tw["REVIEW"]["frauds"] + tw["BLOCK"]["frauds"]
    conc = ((tw["REVIEW"]["frauds"] / max(F, 1))
            / max(tw["REVIEW"]["n"] / len(yt), 1e-9))

    w = 82
    print("\n" + "=" * w)
    print("L&T VEHICLE LOAN DEFAULT -- REAL INDIAN FINANCIAL DATA")
    print("=" * w)
    print(f"{'test loans':<34}{len(yt):>48,}")
    print(f"{'defaults in test':<34}{F:>48,}")
    print(f"{'ROC-AUC':<34}{auc:>48.4f}")
    print(f"{'PR-AUC':<34}{ap:>48.4f}")
    print(f"{'base rate':<34}{base:>48.4f}")
    print(f"{'lift over base':<34}{ap / base:>47.1f}x")
    print("-" * w)
    print(f"{'TWO-WAY  refuse':<34}{int(blocked.sum()):>26,} loans   "
          f"caught {int((blocked & (yt == 1)).sum()):>6,}")
    for nm in ("ALLOW", "REVIEW", "BLOCK"):
        r = tw[nm]
        print(f"  {nm:<32}{r['n']:>26,} loans   "
              f"defaults {r['frauds']:>5,} ({r['frauds'] / max(F, 1):>5.1%})")
    print(f"{'  reached (review+block)':<34}{reached:>33,} / {F:,} "
          f"({reached / max(F, 1):.1%})")
    print(f"{'  review concentration':<34}{conc:>47.1f}x")
    print("-" * w)
    print("Reference from the transfer study:")
    print("  IEEE-CIS 0.8870 / 2.6x     NeurIPS BAF 0.8892 / 3.9x")
    print("  Indian generated datasets: 0.4896, 0.5252, 0.8513 (after 3 leaks)")
    print("  pure noise 0.5000 / 1.0x")
    print()
    if auc > 0.75:
        v = ("STRONG. Real Indian financial data, real signal, no cleaning "
             "needed.")
    elif auc > 0.62:
        v = ("MODEST BUT REAL. Loan default is intrinsically harder to predict "
             "than card\nfraud -- the published benchmarks for this dataset sit "
             "around 0.65 AUC, so this\nis in the expected range rather than "
             "disappointing.")
    else:
        v = "WEAK. Below the published benchmarks for this dataset; investigate."
    print(v)
    print()
    print("SCOPE, stated carefully: this is LOAN DEFAULT, not transaction fraud.")
    print("It does NOT show the fraud detector transfers to Indian fraud. It "
          "shows the")
    print("decision layer works on a real Indian loss event -- which narrows the "
          "earlier")
    print("claim from 'no real Indian data' to 'no real Indian TRANSACTION-FRAUD "
          "data'.")
    print("=" * w)

    Path("artifacts/try_lt.json").write_text(json.dumps({
        "source": "mamtadhaker/lt-vehicle-loan-default-prediction",
        "loss_type": "loan default (not transaction fraud)",
        "real_data": True, "rows": int(n), "test_rows": int(len(yt)),
        "defaults_total": F, "roc_auc": auc, "pr_auc": ap, "base_rate": base,
        "lift": ap / base, "features": cols,
        "two_way_blocked": int(blocked.sum()),
        "two_way_caught": int((blocked & (yt == 1)).sum()),
        "three_way": tw, "reached": reached,
        "reached_rate": reached / max(F, 1),
        "review_concentration": conc, "verdict": v.replace("\n", " "),
    }, indent=2), encoding="utf-8")
    print("\nwrote artifacts/try_lt.json")


if __name__ == "__main__":
    main()
