"""Deriving the review band from costs instead of inventing it (Chow, 1970).

Until now the review band was [0.20, 0.80) because those numbers looked
reasonable. They were invented. Chow's rule for classification with a reject
option gives the optimal three-way decision directly from the costs.

For each transaction, with calibrated p = P(fraud):

    E[cost | BLOCK]  = (1 - p) * C_FP(x)      we block, it was genuine
    E[cost | ALLOW]  =      p  * C_FN(x)      we allow, it was fraud
    E[cost | REVIEW] =           C_REVIEW     analyst time; assume they resolve it

Take the arg-min of the three. Because C_FP and C_FN both depend on the
transaction amount, the resulting band is INSTANCE-SPECIFIC -- consistent with
Bahnsen et al. on instance-dependent costs, and a strict generalisation of the
per-instance two-way threshold from Elkan.

This also connects to the analyst budget (Dal Pozzolo et al.): the rule decides
HOW MANY cases land in review, so we can check that volume against real
capacity -- or invert it, and ask what review cost is implied by the capacity a
team actually has.

Assumption stated plainly: reviews are treated as resolving correctly. Real
analysts are not perfect, so this is optimistic about the review action. The
sweep below shows how sensitive the conclusion is to the review cost.
"""
from __future__ import annotations

import json

import numpy as np

import config
import pipeline

# Cost of one human review: analyst minutes, not a transaction loss.
# Swept below rather than trusted as a point estimate.
REVIEW_COST_INR = 150.0
SWEEP = [25.0, 50.0, 150.0, 400.0, 1000.0]
SECONDS_PER_DAY = 86_400
TEST_DAYS = 42  # measured in alert_budget.py


def chow_actions(p, amount, review_cost):
    """Return the arg-min action per transaction: 0=allow, 1=review, 2=block."""
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    cost = np.stack([p * c_fn,                       # allow
                     np.full_like(p, review_cost),   # review
                     (1 - p) * c_fp])                # block
    return cost.argmin(axis=0), cost


def total_cost(actions, cost_matrix):
    return float(cost_matrix[actions, np.arange(cost_matrix.shape[1])].sum())


def realised_loss(actions, y, amount, review_cost):
    """Actual rupees, given the true labels -- not the expected-cost surrogate."""
    allow, review, block = actions == 0, actions == 1, actions == 2
    fn = allow & (y == 1)
    fp = block & (y == 0)
    return float((amount[fn] + config.CHARGEBACK_FEE_INR).sum()
                 + (amount[fp] * config.MERCHANT_MARGIN_RATE
                    + config.CUSTOMER_LTV_INR).sum()
                 + review.sum() * review_cost)


def main() -> None:
    p = np.load(config.ARTIFACTS / "cal_test_scores.npy")
    y, amount = pipeline.load_test_arrays()

    w = 94
    print("=" * w)
    print("CHOW'S RULE: THE REVIEW BAND DERIVED FROM COSTS, NOT CHOSEN")
    print("=" * w)
    print(f"{'review cost':>13}{'allow':>11}{'review':>11}{'block':>10}"
          f"{'review/day':>13}{'realised loss':>17}{'fraud in review':>18}")
    print("-" * w)

    rows = []
    for rc in SWEEP:
        a, cm = chow_actions(p, amount, rc)
        n_rev = int((a == 1).sum())
        loss = realised_loss(a, y, amount, rc)
        fr = float(y[a == 1].mean()) if n_rev else 0.0
        rows.append({"review_cost": rc, "n_allow": int((a == 0).sum()),
                     "n_review": n_rev, "n_block": int((a == 2).sum()),
                     "review_per_day": n_rev / TEST_DAYS,
                     "realised_loss": loss, "fraud_rate_in_review": fr})
        print(f"{rc:>13,.0f}{(a == 0).sum():>11,}{n_rev:>11,}{(a == 2).sum():>10,}"
              f"{n_rev / TEST_DAYS:>13,.0f}{loss:>17,.0f}{fr:>18.2%}")
    print("-" * w)

    # the band we had been using, for comparison
    arb = ((p >= 0.20) & (p < 0.80))
    print(f"previous ARBITRARY band [0.20, 0.80): {int(arb.sum()):,} cases "
          f"({arb.sum() / TEST_DAYS:,.0f}/day), fraud rate {y[arb].mean():.2%}")

    a_def, _ = chow_actions(p, amount, REVIEW_COST_INR)
    der = a_def == 1
    print(f"DERIVED band at Rs {REVIEW_COST_INR:,.0f} review cost:  "
          f"{int(der.sum()):,} cases ({der.sum() / TEST_DAYS:,.0f}/day), "
          f"fraud rate {y[der].mean():.2%}")
    overlap = float((arb & der).sum() / max(der.sum(), 1))
    print(f"overlap with the arbitrary band: {overlap:.1%}")

    # two-way (Elkan) vs three-way (Chow), on realised rupees
    tau = ((amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR)
           / (amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
              + amount + config.CHARGEBACK_FEE_INR))
    two_way = np.where(p >= tau, 2, 0)
    loss_2 = realised_loss(two_way, y, amount, REVIEW_COST_INR)
    loss_3 = realised_loss(a_def, y, amount, REVIEW_COST_INR)
    do_nothing = realised_loss(np.zeros_like(a_def), y, amount, REVIEW_COST_INR)

    print("-" * w)
    print(f"{'policy':<44}{'realised loss':>18}{'savings':>14}")
    print("-" * w)
    for name, l in (("do nothing", do_nothing),
                    ("two-way, per-instance tau* (Elkan/Bahnsen)", loss_2),
                    ("three-way with review (Chow)", loss_3)):
        print(f"{name:<44}{l:>18,.0f}{(do_nothing - l) / do_nothing:>14.2%}")
    print("-" * w)
    print(f"adding a review action is worth Rs {loss_2 - loss_3:,.0f} "
          f"at a Rs {REVIEW_COST_INR:,.0f} review cost")
    print()
    print("Caveat stated up front: this assumes a reviewed case is resolved")
    print("correctly. Real analysts are not perfect, so the three-way number is")
    print("optimistic. The sweep above shows how fast the review volume moves")
    print("with the assumed cost of an analyst's time.")
    print("=" * w)

    (config.ARTIFACTS / "chow_band.json").write_text(
        json.dumps({"sweep": rows, "default_review_cost": REVIEW_COST_INR,
                    "loss_two_way": loss_2, "loss_three_way": loss_3,
                    "loss_do_nothing": do_nothing,
                    "overlap_with_arbitrary_band": overlap}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'chow_band.json'}")


if __name__ == "__main__":
    main()
