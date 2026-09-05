"""Control experiment: the shipped pipeline on data that HAS signal.

try_indian.py found nothing and blocked nothing. Two explanations fit that:

  (a) the Indian dataset is synthetic -- its fraud label is a 5% coin flip
      independent of every feature, which we verified directly, or
  (b) our architecture is broken and would find nothing anywhere.

Only a dataset with known signal separates them. If we find signal here, (a)
holds. If we find nothing here too, (b) holds and the whole submission is in
trouble. This is the falsification test.

DATASET: Bank Account Fraud (NeurIPS 2022), Jesus et al., Feedzai.
  1,000,000 rows, 32 features, 1.103% fraud, 8 months.
  Published as an academic benchmark precisely because it carries real,
  documented predictive signal (reported AUC ~0.87 in the paper).

IMPORTANT DIFFERENCE, stated because it changes how to read the block counts.
This is ACCOUNT-OPENING fraud, not transaction fraud. There is no transaction
amount. The nearest thing to an exposure is `proposed_credit_limit`, which is
what we map onto TransactionAmt -- the money at risk if a fraudulent
application is approved.

That means our rupee constants (tuned for IEEE-CIS ticket sizes) are the wrong
scale here, and a low block count could be a COST-SCALING artefact rather than a
detection failure. So this reports the two separately:

  RANKING QUALITY (ROC-AUC, PR-AUC)  -- does the architecture find signal?
                                        Independent of any cost assumption.
  DECISION OUTPUT (allow/review/block) -- what the policy does at these costs,
                                        reported at the default constants AND
                                        at constants rescaled to this dataset.

The first question is the one that decides whether our architecture works.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

TRAIN_MONTHS, CALIB_MONTHS, TEST_MONTHS = range(0, 5), [5], [6, 7]


def load():
    p = glob.glob(os.path.expanduser(
        "~/.cache/kagglehub/datasets/sgpjesus/*/versions/*/Base.csv"))[0]
    df = pd.read_csv(p)
    print(f"loaded {len(df):,} rows, {len(df.columns)} columns")

    # month is the only time column and it is coarse (8 values). Row order
    # within a month is the file's own order; combine them into a strictly
    # increasing key so the temporal split has a total order to work with.
    df = df.sort_values("month", kind="stable").reset_index(drop=True)
    df["TransactionDT"] = df["month"].to_numpy() * 10**7 + np.arange(len(df))
    df["isFraud"] = df["fraud_bool"]
    df["TransactionAmt"] = df["proposed_credit_limit"]
    df["TransactionID"] = np.arange(len(df))
    print(f"  fraud rate {df['isFraud'].mean():.3%} "
          f"({int(df['isFraud'].sum()):,} positives)")
    print(f"  exposure (proposed_credit_limit) median "
          f"{df['TransactionAmt'].median():,.0f}, "
          f"max {df['TransactionAmt'].max():,.0f}")
    return df


def three_way(p, amt, y, fp_rate, fp_fixed, fn_fixed, review_cost):
    c_fp = amt * fp_rate + fp_fixed
    c_fn = amt + fn_fixed
    cost = np.stack([p * c_fn, np.full_like(p, review_cost), (1 - p) * c_fp])
    a = cost.argmin(axis=0)
    out = {}
    for i, nm in enumerate(("ALLOW", "REVIEW", "BLOCK")):
        m = a == i
        out[nm] = {"n": int(m.sum()), "frauds": int((m & (y == 1)).sum())}
    return out


def main() -> None:
    import calibrate
    import config
    import data
    import features
    import lightgbm as lgb
    import model as M
    from sklearn.metrics import average_precision_score, roc_auc_score

    config.USD_TO_INR = 1.0
    df = load()

    audit = data.audit_leakage(df)
    cols = [c for c in audit["online"]
            if c not in ("fraud_bool", "month", "proposed_credit_limit")]
    print(f"  {len(cols)} feature columns")

    m = df["month"].to_numpy()
    tr, ca, te = df[np.isin(m, list(TRAIN_MONTHS))], df[m == 5], df[np.isin(m, TEST_MONTHS)]
    print(f"\ntemporal split by MONTH  train {len(tr):,} (months 0-4)  "
          f"calib {len(ca):,} (month 5)  test {len(te):,} (months 6-7)")
    print(f"  fraud rate  train {tr['isFraud'].mean():.3%}  "
          f"calib {ca['isFraud'].mean():.3%}  test {te['isFraud'].mean():.3%}")

    X_tr, y_tr, _, cats = features.build(tr, cols)
    X_ca, y_ca, _, _ = features.build(ca, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te, cols, cats)

    print("\ntraining ...")
    b = lgb.train(M.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=M.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = calibrate.fit_calibrator(b.predict(X_ca), y_ca)(b.predict(X_te))

    auc, ap = float(roc_auc_score(y_te, p)), float(average_precision_score(y_te, p))
    base = float(y_te.mean())
    F = int((y_te == 1).sum())

    w = 84
    print("\n" + "=" * w)
    print("CONTROL: SAME PIPELINE, DATA THAT IS KNOWN TO HAVE SIGNAL")
    print("=" * w)
    print("PART 1 -- RANKING QUALITY.  Does the architecture find signal?")
    print("This part depends on NO cost assumption.")
    print("-" * w)
    print(f"{'test rows':<40}{len(y_te):>44,}")
    print(f"{'frauds in test':<40}{F:>44,}")
    print(f"{'ROC-AUC':<40}{auc:>44.4f}")
    print(f"{'PR-AUC':<40}{ap:>44.4f}")
    print(f"{'PR-AUC of random guessing':<40}{base:>44.4f}")
    print(f"{'lift over base rate':<40}{ap / base:>43.1f}x")
    print(f"{'  (Indian dataset, for contrast)':<40}{'AUC 0.5067, lift 1.0x':>44}")
    print("-" * w)

    verdict = ("ARCHITECTURE WORKS. It finds real signal when real signal exists,"
               "\nso the Indian result was the dataset, not us."
               if auc > 0.75 else
               "PROBLEM. Signal exists in this data and we did not find it."
               if auc < 0.65 else
               "PARTIAL. Some signal found, weaker than the benchmark reports.")
    print(verdict)

    print("\n" + "-" * w)
    print("PART 2 -- DECISION OUTPUT.  What does the policy actually do?")
    print("Read with care: this is ACCOUNT-OPENING fraud, so 'amount' is the")
    print("proposed credit limit and our rupee constants are the wrong scale.")
    print("-" * w)

    med = float(np.median(amt_te))
    scaled_ltv = med * 0.25          # keep LTV proportional to typical exposure
    settings = [
        ("default constants (IEEE-CIS scale)", config.MERCHANT_MARGIN_RATE,
         config.CUSTOMER_LTV_INR, config.CHARGEBACK_FEE_INR),
        (f"rescaled to this dataset (LTV={scaled_ltv:,.0f})",
         config.MERCHANT_MARGIN_RATE, scaled_ltv, med * 0.1),
    ]
    out = {}
    for label, fpr, fpf, fnf in settings:
        c_fp, c_fn = amt_te * fpr + fpf, amt_te + fnf
        blocked = p >= c_fp / (c_fp + c_fn)
        tp = int((blocked & (y_te == 1)).sum())
        print(f"\n{label}")
        print(f"  TWO-WAY   BLOCK {int(blocked.sum()):>8,} txns "
              f"({blocked.mean():>6.2%})   frauds caught {tp:>6,} / {F:,}"
              f" ({tp / max(F, 1):.1%})")
        tw = three_way(p, amt_te, y_te, fpr, fpf, fnf, 150.0)
        for nm in ("ALLOW", "REVIEW", "BLOCK"):
            r = tw[nm]
            print(f"  THREE-WAY {nm:<7}{r['n']:>8,} txns "
                  f"({r['n'] / len(y_te):>6.2%})   frauds {r['frauds']:>6,}"
                  f" ({r['frauds'] / max(F, 1):>5.1%})")
        conc = ((tw["REVIEW"]["frauds"] / max(F, 1))
                / max(tw["REVIEW"]["n"] / len(y_te), 1e-9))
        print(f"  review-lane fraud concentration: {conc:.2f}x "
              f"(1.0 = no better than chance)")
        out[label] = {"two_way_blocked": int(blocked.sum()),
                      "two_way_frauds_caught": tp, "three_way": tw,
                      "review_concentration": conc}
    print("=" * w)

    with open("artifacts/try_baf.json", "w", encoding="utf-8") as f:
        json.dump({"source": "sgpjesus/bank-account-fraud-dataset-neurips-2022",
                   "variant": "Base", "test_rows": int(len(y_te)),
                   "frauds_total": F, "roc_auc": auc, "pr_auc": ap,
                   "base_rate": base, "lift": ap / base,
                   "features": cols, "verdict": verdict.replace("\n", " "),
                   "policies": out}, f, indent=2)
    print("\nwrote artifacts/try_baf.json")


if __name__ == "__main__":
    main()
