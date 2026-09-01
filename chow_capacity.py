"""Chow's reject option under a real analyst budget.

Unconstrained Chow (chow_band.py) reported 82.96% savings. That number is not
achievable: at a Rs 150 review cost it routes 845 cases per day to human review,
roughly 30% of all traffic, when Dal Pozzolo-style capacity is 10-200 cards/day.
It wins by assuming an analyst team nobody has.

The operationally correct policy combines all three results:

  Elkan / Bahnsen   per-instance two-way threshold tau*(x) decides everything
                    that is NOT reviewed.
  Chow              gives the value of reviewing a case:
                        benefit(x) = min(E[cost|allow], E[cost|block]) - C_REVIEW
                    i.e. what you save by handing it to a human instead of
                    deciding automatically.
  Dal Pozzolo       only k cases per day can actually be reviewed.

So: rank each day's transactions by review benefit, review the top k, and apply
the two-way rule to the rest. This is Chow subject to a capacity constraint.

The assumption that a review resolves correctly is retained but now matters far
less, because it applies to k cases per day rather than a third of the stream.
A sensitivity row shows what happens if analysts are only 80% accurate.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
import data
import pipeline

REVIEW_COST_INR = 150.0
BUDGETS = [0, 10, 25, 50, 100, 200, 500]
ANALYST_ACCURACY = [1.0, 0.9, 0.8]
SECONDS_PER_DAY = 86_400


def costs(p, amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return p * c_fn, (1 - p) * c_fp          # E[cost|allow], E[cost|block]


def policy_loss(p, y, amount, day, k, review_cost, accuracy, rng):
    """Realised rupees for: review top-k/day by benefit, two-way rule elsewhere."""
    e_allow, e_block = costs(p, amount)
    auto_block = e_block < e_allow                    # per-instance tau*
    benefit = np.minimum(e_allow, e_block) - review_cost

    reviewed = np.zeros(len(p), bool)
    if k > 0:
        df = pd.DataFrame({"benefit": benefit, "day": day})
        for _, g in df.groupby("day"):
            top = g.nlargest(min(k, len(g)), "benefit")
            reviewed[top.index[top["benefit"] > 0]] = True

    # reviewed cases: analyst is right with probability `accuracy`
    correct = rng.random(len(p)) < accuracy
    blocked = np.where(reviewed, np.where(correct, y == 1, y == 0), auto_block)

    fn = (~blocked) & (y == 1)
    fp = blocked & (y == 0)
    return float((amount[fn] + config.CHARGEBACK_FEE_INR).sum()
                 + (amount[fp] * config.MERCHANT_MARGIN_RATE
                    + config.CUSTOMER_LTV_INR).sum()
                 + reviewed.sum() * review_cost), int(reviewed.sum())


def main() -> None:
    p = np.load(config.ARTIFACTS / "cal_test_scores.npy")
    y, amount = pipeline.load_test_arrays()

    raw = data.load_raw()
    _, _, te_df = data.temporal_split(raw)
    day = (te_df["TransactionDT"].to_numpy() // SECONDS_PER_DAY).astype(int)
    day = day - day.min()
    n_days = len(np.unique(day))

    rng = np.random.default_rng(config.SEED)
    do_nothing = float((amount[y == 1] + config.CHARGEBACK_FEE_INR).sum())

    w = 92
    print("=" * w)
    print("CHOW'S REJECT OPTION UNDER A REAL ANALYST BUDGET")
    print("=" * w)
    print(f"test period {n_days} days   review cost Rs {REVIEW_COST_INR:,.0f}"
          f"   do-nothing loss Rs {do_nothing:,.0f}")
    print("-" * w)
    print(f"{'budget k/day':<14}{'reviewed':>11}{'analyst 100%':>16}"
          f"{'analyst 90%':>15}{'analyst 80%':>15}{'savings@100%':>15}")
    print("-" * w)

    rows = []
    for k in BUDGETS:
        losses = {}
        n_rev = 0
        for acc in ANALYST_ACCURACY:
            l, n_rev = policy_loss(p, y, amount, day, k, REVIEW_COST_INR, acc, rng)
            losses[acc] = l
        sav = (do_nothing - losses[1.0]) / do_nothing
        rows.append({"k": k, "reviewed": n_rev,
                     "loss_by_accuracy": {str(a): losses[a] for a in losses},
                     "savings_at_perfect": sav})
        print(f"{k:<14}{n_rev:>11,}{losses[1.0]:>16,.0f}{losses[0.9]:>15,.0f}"
              f"{losses[0.8]:>15,.0f}{sav:>15.2%}")
    print("-" * w)

    zero = rows[0]["loss_by_accuracy"]["1.0"]
    print(f"k=0 is the pure two-way policy (no review): Rs {zero:,.0f}")
    for r in rows[1:]:
        gain = zero - r["loss_by_accuracy"]["1.0"]
        per = gain / max(r["reviewed"], 1)
        print(f"  k={r['k']:<4} reviews {r['reviewed']:>6,} cases, "
              f"saves Rs {gain:>10,.0f}  (Rs {per:,.0f} per review)")
    print("-" * w)
    print("Unconstrained Chow claimed 82.96% savings by reviewing 845 cases/day.")
    print("At a realistic budget the review action is still clearly worth it, but")
    print("the gain is bounded by capacity -- and it degrades if analysts are")
    print("wrong 10-20% of the time, which the columns above quantify.")
    print("=" * w)

    (config.ARTIFACTS / "chow_capacity.json").write_text(
        json.dumps({"do_nothing": do_nothing, "days": int(n_days),
                    "review_cost": REVIEW_COST_INR, "budgets": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'chow_capacity.json'}")


if __name__ == "__main__":
    main()
