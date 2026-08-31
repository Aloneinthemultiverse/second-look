"""Data loading, temporal splitting, and the leakage audit.

The single most important thing in this file is that the split is TEMPORAL.
Public work on IEEE-CIS very often splits randomly, which trains the model on
transactions that occurred after the ones it is evaluated on. That leaks the
future, inflates every score, and produces a model that cannot run in
production. We split on TransactionDT and never shuffle.
"""
from __future__ import annotations

import pandas as pd

import config


# Features excluded from live inference. Two reasons, tracked separately so we
# can report the cost of each restriction.
#
# TEMPORAL DRIFT: the D_* columns are timedeltas (days since the card was first
#   seen, days since the prior transaction). They ARE computable at checkout, so
#   this is NOT leakage in the chargeback sense -- an earlier version of this
#   file claimed that and it was wrong. The real problem is that they are
#   day-offsets which grow with TransactionDT, so under a temporal split a model
#   can learn "large D => later period". That is time leakage which does not
#   survive in production. Excluded here; the alternative is to normalise them
#   as (D - day_index). Deliberate choice, not an oversight.
#
# LATENCY: computable in principle, but not inside a <50ms budget at checkout
#   (long-window aggregates over the full card/device history).
DRIFT_PREFIXES = ("D",)
LATENCY_PREFIXES = ("V",)  # 339 anonymised Vesta aggregates -- offline features


def load_raw() -> pd.DataFrame:
    """Merge the transaction and identity tables on TransactionID."""
    if not config.TRANSACTION_CSV.exists():
        raise FileNotFoundError(
            f"Expected IEEE-CIS data at {config.TRANSACTION_CSV}.\n"
            "Download train_transaction.csv and train_identity.csv from\n"
            "https://www.kaggle.com/competitions/ieee-fraud-detection/data\n"
            f"and place them in {config.DATA_DIR}/"
        )
    txn = pd.read_csv(config.TRANSACTION_CSV)
    if config.IDENTITY_CSV.exists():
        ident = pd.read_csv(config.IDENTITY_CSV)
        txn = txn.merge(ident, on="TransactionID", how="left")
    return txn


def temporal_split(df: pd.DataFrame):
    """Sort by TransactionDT and cut into train / calibration / test.

    The calibration slice sits BETWEEN train and test in time. It is used only
    to fit the probability calibrator, and is never seen by the booster or
    used for reporting.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    i_train = int(n * config.TRAIN_FRAC)
    i_calib = int(n * (config.TRAIN_FRAC + config.CALIB_FRAC))

    train = df.iloc[:i_train]
    calib = df.iloc[i_train:i_calib]
    test = df.iloc[i_calib:]

    assert train["TransactionDT"].max() <= calib["TransactionDT"].min()
    assert calib["TransactionDT"].max() <= test["TransactionDT"].min()
    return train, calib, test


def audit_leakage(df: pd.DataFrame) -> dict:
    """Classify every column as usable online, or excluded and why.

    Reported in the submission so the restriction is visible rather than
    implied.
    """
    label = {"isFraud"}
    ids = {"TransactionID", "TransactionDT"}

    excluded_drift, excluded_latency, online = [], [], []
    for col in df.columns:
        if col in label or col in ids:
            continue
        if col.startswith(DRIFT_PREFIXES) and col[1:2].isdigit():
            excluded_drift.append(col)
        elif col.startswith(LATENCY_PREFIXES) and col[1:2].isdigit():
            excluded_latency.append(col)
        else:
            online.append(col)

    return {
        "online": online,
        "excluded_drift": excluded_drift,
        "excluded_latency": excluded_latency,
    }


def summarise(df: pd.DataFrame, train, calib, test, audit: dict) -> str:
    fraud_rate = df["isFraud"].mean()
    lines = [
        "=" * 62,
        "DATA SUMMARY",
        "=" * 62,
        f"rows                {len(df):,}",
        f"columns             {df.shape[1]:,}",
        f"fraud rate          {fraud_rate:.4%}  ({int(df['isFraud'].sum()):,} positives)",
        "",
        "TEMPORAL SPLIT (sorted by TransactionDT, never shuffled)",
        f"  train             {len(train):,}  fraud {train['isFraud'].mean():.4%}",
        f"  calibration       {len(calib):,}  fraud {calib['isFraud'].mean():.4%}",
        f"  test              {len(test):,}  fraud {test['isFraud'].mean():.4%}",
        "",
        "FEATURE AUDIT",
        f"  online (usable)   {len(audit['online'])}",
        f"  excluded: drift   {len(audit['excluded_drift'])}  (D_* time-offsets)",
        f"  excluded: latency {len(audit['excluded_latency'])}  (V_* offline aggregates)",
        "=" * 62,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    df = load_raw()
    audit = audit_leakage(df)
    train, calib, test = temporal_split(df)
    print(summarise(df, train, calib, test, audit))

    out = config.ARTIFACTS / "feature_audit.md"
    out.write_text(
        "# Feature audit\n\n"
        "## Online (used at inference)\n\n"
        + "\n".join(f"- `{c}`" for c in audit["online"])
        + "\n\n## Excluded — temporal drift (D_* day-offsets grow with TransactionDT)\n\n"
        + "\n".join(f"- `{c}`" for c in audit["excluded_drift"])
        + "\n\n## Excluded — latency (offline aggregates)\n\n"
        + "\n".join(f"- `{c}`" for c in audit["excluded_latency"])
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
