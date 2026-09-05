"""Pre-flight leakage screen. Run this BEFORE training on your own data.

    python check_data.py                 # checks the CSV in config.TRANSACTION_CSV
    python check_data.py path/to/my.csv  # or any file

WHY THIS EXISTS. The pipeline is plug-and-play: rename four columns, adjust three
constants, and it runs. That is also the danger. `data.audit_leakage()` only knows
IEEE-CIS's own column prefixes, so on ANY other dataset it classifies every column
as usable -- including columns that do not exist at decision time.

This was not hypothetical. Testing nine datasets turned up:

    Transaction_Status   "Declined" 59.6% fraud vs "Successful" 1.4%
                         -- you only know a payment was declined AFTER something
                         declined it. Top feature in the model. AUC 0.9754.
    Card_Status          Blocked/Expired/Lost were 100.00% fraud. Deterministic.
    Fraud_Reason         populated only on frauds. The label wearing a hat.
    Approved_Flag        a risk grade the bank computed FROM the other columns.
                         Predicting it from them scored AUC 0.9999.

Every one of those looks like an ordinary column. Every one produces a model that
is excellent offline and worthless in production, because at the moment you have
to decide, the column is empty.

The screen below is deliberately noisy: it would rather flag an honest feature
than miss a leak. Read each finding and decide. A flag is a question, not a verdict.

WHAT IT CANNOT DO. It cannot know your business. A column that is legitimately
available at decision time in your system may be an outcome in someone else's.
The final call is yours, and the question to ask each flagged column is always
the same: **at the instant I must decide, do I actually have this value?**
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET_CANDIDATES = ["isFraud", "is_fraud", "Is_Fraud", "fraud_bool",
                     "Fraud_Flag", "fraud_flag", "Class", "class", "loan_default"]
AMOUNT_CANDIDATES = ["TransactionAmt", "Transaction_Amount", "amount", "Amount",
                     "transaction_amount", "disbursed_amount"]
TIME_CANDIDATES = ["TransactionDT", "Transaction_Date", "timestamp", "Time",
                   "transaction_date", "step", "month"]

SINGLE_COL_AUC_ALARM = 0.75    # one column alone should not predict this well
SINGLE_COL_AUC_CERTAIN = 0.99  # at this level it is not a feature, it is the label
DETERMINISTIC_MIN_ROWS = 30    # ignore tiny categories; 3 rows at 100% is noise


def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def single_col_auc(s: pd.Series, y: np.ndarray) -> float | None:
    """AUC of one column alone. Categoricals are mapped to their fraud rate."""
    try:
        if s.dtype == object or str(s.dtype) == "category":
            if s.nunique() > 200:
                return None
            rate = pd.Series(y, index=s.index).groupby(s, observed=True).mean()
            v = s.map(rate).astype(float)
        else:
            v = pd.to_numeric(s, errors="coerce").astype(float)
        v = v.fillna(v.median() if v.notna().any() else 0.0)
        if v.nunique() < 2:
            return None
        return float(max(roc_auc_score(y, v), 1 - roc_auc_score(y, v)))
    except Exception:
        return None


def main() -> None:
    import config
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.TRANSACTION_CSV
    if not path.exists():
        print(f"no such file: {path}")
        raise SystemExit(1)

    df = pd.read_csv(path)
    print(f"\n{path.name}  --  {len(df):,} rows, {len(df.columns)} columns\n")

    tgt = find_col(df, TARGET_CANDIDATES)
    amt = find_col(df, AMOUNT_CANDIDATES)
    tim = find_col(df, TIME_CANDIDATES)

    w = 78
    print("=" * w)
    print("REQUIRED COLUMNS")
    print("=" * w)
    for label, col, names in (("label", tgt, TARGET_CANDIDATES),
                              ("amount", amt, AMOUNT_CANDIDATES),
                              ("time", tim, TIME_CANDIDATES)):
        if col:
            print(f"  OK    {label:<8} -> {col}")
        else:
            print(f"  MISS  {label:<8} -> none of {names[:4]}...")
    if not tgt:
        print("\nCannot continue without a label column. Rename yours to 'isFraud'.")
        raise SystemExit(1)

    y = df[tgt].to_numpy()
    if set(pd.unique(y)) - {0, 1}:
        print(f"\nlabel '{tgt}' is not 0/1 -- values {sorted(set(pd.unique(y)))[:6]}")
        raise SystemExit(1)
    rate = float(y.mean())

    findings = []

    print("\n" + "=" * w)
    print("BASE RATE")
    print("=" * w)
    print(f"  {rate:.4%} positive ({int(y.sum()):,} of {len(y):,})")
    if abs(rate - 0.5) < 0.005:
        findings.append(("CRITICAL", tgt,
                         f"base rate is {rate:.4%} -- this dataset has been "
                         "rebalanced. Precision, PR-AUC and every cost figure "
                         "computed on it will be meaningless."))
        print("  ^ REBALANCED. See findings.")
    elif rate > 0.25:
        print(f"  note: {rate:.1%} is high for fraud. Check this is really a fraud "
              "label\n        and not a default, dispute or approval outcome.")

    print("\n" + "=" * w)
    print("COLUMN SCREEN")
    print("=" * w)
    skip = {tgt}
    checked = 0
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        checked += 1

        # 1. null-pattern leak: populated on one class, empty on the other
        n1 = s[y == 1].notna().mean()
        n0 = s[y == 0].notna().mean()
        if abs(n1 - n0) > 0.5:
            findings.append(("CRITICAL", c,
                             f"populated for {n1:.0%} of positives but {n0:.0%} of "
                             "negatives. A column that only exists when the answer "
                             "is known IS the answer."))
            continue

        # 2. deterministic category levels
        if (s.dtype == object or str(s.dtype) == "category") and s.nunique() <= 60:
            g = pd.DataFrame({"v": s, "y": y}).groupby("v", observed=True)["y"] \
                  .agg(["mean", "size"])
            g = g[g["size"] >= DETERMINISTIC_MIN_ROWS]
            # ONLY all-positive levels. This screen twice flagged honest columns
            # before that restriction: first every sparse email domain with zero
            # frauds, then verizon.net at 0% across 620 rows. Neither is a leak.
            # A leak is a column you do not HAVE when you decide -- an email
            # domain is present at checkout, and a domain that never defrauds
            # anyone is a strong feature, which is the thing we want the model to
            # find. An all-POSITIVE level is different in kind: "if X then fraud,
            # always" is what a labelling rule looks like from the inside, and
            # Card_Status had three of them at exactly 100.00%.
            hard = g[g["mean"] >= 0.999]
            if len(hard) and len(hard) < len(g):
                ex = ", ".join(f"{i}={r['mean']:.1%} (n={int(r['size']):,})"
                               for i, r in hard.head(3).iterrows())
                findings.append(("CRITICAL", c,
                                 "category levels that are 100% positive: "
                                 f"{ex}. \"If X then fraud, always\" is a "
                                 "labelling rule, not a feature."))
                continue

        # 3. one column that predicts far too well on its own
        a = single_col_auc(s, y)
        if a is not None and a >= SINGLE_COL_AUC_CERTAIN:
            # No honest feature separates fraud this cleanly on its own. At this
            # level the column is not predicting the label, it is reporting it.
            findings.append(("CRITICAL", c,
                             f"predicts the label alone at AUC {a:.4f}. Nothing "
                             "available before the decision separates fraud this "
                             "cleanly -- this column is reporting the outcome, "
                             "not predicting it."))
            continue
        if a is not None and a >= SINGLE_COL_AUC_ALARM:
            findings.append(("SUSPECT", c,
                             f"predicts the label alone at AUC {a:.4f}. Ask whether "
                             "this value exists at the moment you must decide."))
            continue

        # 4. identifiers
        if (s.dtype == object or str(s.dtype) == "category") \
                and s.nunique() > 0.5 * len(s):
            findings.append(("DROP", c,
                             f"{s.nunique():,} unique values in {len(s):,} rows -- "
                             "an identifier. Drop it; a tree will memorise it."))

    print(f"  screened {checked} columns")

    # 5. per-entity independence -- did anyone actually simulate customers?
    idc = [c for c in df.columns if "customer" in c.lower() and "id" in c.lower()]
    if idc:
        u = df[idc[0]].nunique()
        if u == len(df):
            findings.append(("SUSPECT", idc[0],
                             f"every row has a different {idc[0]} ({u:,} of "
                             f"{len(df):,}). No customer transacts twice, so no "
                             "behavioural history exists. Typical of generated data."))

    print("\n" + "=" * w)
    print(f"FINDINGS  ({len(findings)})")
    print("=" * w)
    if not findings:
        print("  Nothing flagged. That is not a guarantee -- this screen cannot")
        print("  know which of your columns are filled in after the fact.")
    order = {"CRITICAL": 0, "SUSPECT": 1, "DROP": 2}
    for sev, col, msg in sorted(findings, key=lambda f: order[f[0]]):
        print(f"\n  [{sev}] {col}")
        for line in _wrap(msg, w - 10):
            print(f"      {line}")

    crit = sum(1 for f in findings if f[0] == "CRITICAL")
    print("\n" + "=" * w)
    if crit:
        print(f"DO NOT TRAIN YET. {crit} critical finding(s).")
        print("Remove those columns, or the rows the rule fires on, and re-run this.")
        print("A model trained on them will look excellent and be worthless live.")
    elif findings:
        print("Review the flags above, then train. Each one is a question:")
        print("at the instant you must decide, do you actually have that value?")
    else:
        print("Clear to train. Set your costs in config.py first -- every rupee")
        print("figure this repo produces is downstream of them.")
    print("=" * w)
    raise SystemExit(2 if crit else 0)


def _wrap(text, width):
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()
