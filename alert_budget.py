"""Evaluation under an analyst alert budget (Dal Pozzolo et al., TNNLS 2017).

Everything this project measured until now assumed every transaction can be
acted on. Real fraud operations cannot. From "Credit Card Fraud Detection: A
Realistic Modeling and a Novel Learning Strategy":

  - VERIFICATION LATENCY: only a small set of transactions are timely checked
    by investigators.
  - ALERT PRECISION @ k: with a fixed investigator budget k per day, the
    precision of the top-k alerts is the operationally meaningful metric --
    not PR-AUC over the whole stream.
  - CARD precision, not transaction precision: investigators contact a
    CARDHOLDER and then review all that card's recent activity, so multiple
    flagged transactions on one card are a single alert.

This changes what "good" means. A model with mediocre PR-AUC that ranks
extremely well at the very top is operationally better than one with higher
PR-AUC that spreads its confidence.

It also supplies the economic justification for the investigator layer: if
analysts can only work k cases a day, each case must arrive with a readable
brief. That is a capacity argument, not a decorative one.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import calibrate
import config
import data
import features
import model

SECONDS_PER_DAY = 86_400
BUDGETS = [10, 25, 50, 100, 200]


def daily_precision_at_k(df: pd.DataFrame, k: int, by_card: bool) -> dict:
    """Mean precision of the top-k alerts per day.

    by_card=False -> rank transactions, precision = frauds / k
    by_card=True  -> collapse to cards first (a card is one alert, scored by
                     its highest-scoring transaction that day), which is how
                     investigators actually work.
    """
    hits, total, days = 0, 0, 0
    for _, day in df.groupby("day"):
        if by_card:
            g = day.groupby("card1").agg(score=("score", "max"),
                                         fraud=("isFraud", "max"))
        else:
            g = day[["score", "isFraud"]].rename(columns={"isFraud": "fraud"})
        if len(g) == 0:
            continue
        top = g.nlargest(min(k, len(g)), "score")
        hits += int(top["fraud"].sum())
        total += len(top)
        days += 1
    return {"precision_at_k": hits / total if total else 0.0,
            "alerts_worked": total, "frauds_caught": hits, "days": days}


def total_recall_under_budget(df: pd.DataFrame, k: int, by_card: bool) -> float:
    """What fraction of all fraudulent cards/transactions does the budget reach?"""
    caught, universe = 0, 0
    for _, day in df.groupby("day"):
        if by_card:
            g = day.groupby("card1").agg(score=("score", "max"),
                                         fraud=("isFraud", "max"))
        else:
            g = day[["score", "isFraud"]].rename(columns={"isFraud": "fraud"})
        universe += int(g["fraud"].sum())
        top = g.nlargest(min(k, len(g)), "score")
        caught += int(top["fraud"].sum())
    return caught / universe if universe else 0.0


def main() -> None:
    print("training ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    tr_df, ca_df, te_df = data.temporal_split(raw)
    cols = audit["online"]

    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, _, _ = features.build(te_df, cols, cats)

    b = lgb.train(model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = calibrate.fit_calibrator(b.predict(X_ca), y_ca)(b.predict(X_te))

    df = pd.DataFrame({
        "score": p,
        "isFraud": y_te,
        "card1": te_df["card1"].to_numpy(),
        "day": (te_df["TransactionDT"].to_numpy() // SECONDS_PER_DAY).astype(int),
    })
    df["day"] -= df["day"].min()

    n_days = df["day"].nunique()
    frauds_per_day = df.groupby("day")["isFraud"].sum()
    cards_per_day = df.groupby("day")["card1"].nunique()

    w = 92
    print("\n" + "=" * w)
    print("EVALUATION UNDER AN ANALYST ALERT BUDGET  (Dal Pozzolo et al., 2017)")
    print("=" * w)
    print(f"test period            {n_days} days")
    print(f"transactions/day       {len(df)/n_days:,.0f}")
    print(f"distinct cards/day     {cards_per_day.mean():,.0f}")
    print(f"frauds/day             {frauds_per_day.mean():,.1f} "
          f"(min {frauds_per_day.min()}, max {frauds_per_day.max()})")
    print("-" * w)
    print(f"{'budget k/day':<14}{'txn Prec@k':>13}{'card Prec@k':>14}"
          f"{'txn recall':>13}{'card recall':>14}{'alerts':>12}")
    print("-" * w)

    rows = []
    for k in BUDGETS:
        t = daily_precision_at_k(df, k, by_card=False)
        c = daily_precision_at_k(df, k, by_card=True)
        tr_ = total_recall_under_budget(df, k, by_card=False)
        cr = total_recall_under_budget(df, k, by_card=True)
        rows.append({"k": k, "txn_precision_at_k": t["precision_at_k"],
                     "card_precision_at_k": c["precision_at_k"],
                     "txn_recall": tr_, "card_recall": cr,
                     "alerts_worked": t["alerts_worked"]})
        print(f"{k:<14}{t['precision_at_k']:>13.3f}{c['precision_at_k']:>14.3f}"
              f"{tr_:>13.3f}{cr:>14.3f}{t['alerts_worked']:>12,}")
    print("-" * w)

    base = float(df["isFraud"].mean())
    best = rows[0]
    print(f"random-baseline precision {base:.4f}")
    print(f"lift at k=10: transactions {best['txn_precision_at_k']/base:.1f}x   "
          f"cards {best['card_precision_at_k']/base:.1f}x")
    print()
    print("Reading this: at k=10 cards/day an analyst team sees a tiny slice of")
    print("the stream. Precision at the top is what matters operationally --")
    print("PR-AUC over the whole stream describes work nobody will ever do.")
    print()
    unreachable = 1 - rows[-1]["card_recall"]
    print(f"At the largest budget tested (k={BUDGETS[-1]}/day), "
          f"{unreachable:.1%} of fraudulent cards are")
    print("never even looked at. That is the capacity argument for giving each")
    print("reviewed case a readable brief rather than raw feature vectors.")
    print("=" * w)

    (config.ARTIFACTS / "alert_budget.json").write_text(
        json.dumps({"days": int(n_days), "baseline_precision": base,
                    "budgets": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'alert_budget.json'}")


if __name__ == "__main__":
    main()
