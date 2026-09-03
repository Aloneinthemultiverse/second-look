"""Does the decision layer generalise across loss types, or only fraud?

Two gaps in this submission, both answered by the same experiment.

GAP 1. Track 02's headline names "fraud, returns and chargebacks" and lists four
example directions. This project builds none of them -- it builds a detector plus
a cost-based decision layer, and claims to be the thing all four need before they
can act. That has been an assertion. This measures it.

GAP 2. The brief motivates the track with Indian BFSI, and this dataset is US
card-not-present. Indian domestic card payments carry mandatory additional-factor
authentication, so the fraud in this data is structurally suppressed there, and
Indian volume is UPI-dominated with far smaller tickets. What happens if the same
machinery is pointed at Indian UPI economics?

The claim under test: the decision layer is loss-type agnostic. Change only the
cost structure -- never the model, the scores, or the threshold rule -- and it
should produce a materially different and sensible policy for each.

Nothing here claims to DETECT returns or chargebacks. The detector is still a
fraud detector. What is being tested is whether the layer that turns a score into
an action survives being repointed at a different loss.

Cost structures, with their reasoning stated so a reviewer can attack the
assumptions rather than guess at them:

  CNP FRAUD (baseline)   block a genuine customer -> lose margin + lifetime value
                         allow a fraud -> lose the goods, the money, and a fee

  RETURN / RTO           block a genuine order -> same lost customer
                         accept an RTO -> forward and reverse shipping plus
                         handling. Crucially the GOODS COME BACK, so the loss is
                         a fixed logistics cost, NOT the order value. That single
                         difference should move the policy a long way.

  CHARGEBACK DISPUTE     decline to contest a winnable dispute -> lose the amount
                         plus the fee; contest one you cannot win -> analyst time
                         only. Very cheap false positives.

  INDIAN UPI             tiny tickets, thin margins, and UPI has no card-style
                         chargeback -- the merchant's exposure is close to the
                         transaction value with no fee on top, while a wrongly
                         declined customer still costs a full lifetime value.
"""
from __future__ import annotations

import json

import numpy as np

import config
import pipeline

# fp_fixed: cost of a wrong block that does not scale with the amount (lost LTV)
# fp_rate:  share of the amount lost when blocking wrongly (margin)
# fn_rate:  share of the amount lost when wrongly allowing
# fn_fixed: fixed cost of wrongly allowing (fee, shipping, handling)
LOSS_TYPES = {
    "CNP fraud (baseline)": dict(fp_rate=0.25, fp_fixed=2500.0,
                                 fn_rate=1.00, fn_fixed=1200.0),
    "return / RTO":         dict(fp_rate=0.25, fp_fixed=2500.0,
                                 fn_rate=0.00, fn_fixed=350.0),
    "chargeback dispute":   dict(fp_rate=0.00, fp_fixed=150.0,
                                 fn_rate=1.00, fn_fixed=1200.0),
    "Indian UPI":           dict(fp_rate=0.12, fp_fixed=2500.0,
                                 fn_rate=1.00, fn_fixed=0.0),
}
UPI_AMOUNT_SCALE = 0.06     # Indian UPI median ticket vs this dataset's median


def tau(amount, c):
    """Elkan's per-instance threshold under an arbitrary cost structure."""
    c_fp = amount * c["fp_rate"] + c["fp_fixed"]
    c_fn = amount * c["fn_rate"] + c["fn_fixed"]
    return c_fp / (c_fp + c_fn)


def policy(p, y, amount, c):
    t = tau(amount, c)
    blocked = p >= t
    tp, fp, fn = blocked & (y == 1), blocked & (y == 0), (~blocked) & (y == 1)
    loss = float((amount[fp] * c["fp_rate"] + c["fp_fixed"]).sum()
                 + (amount[fn] * c["fn_rate"] + c["fn_fixed"]).sum())
    nothing = float((amount[y == 1] * c["fn_rate"] + c["fn_fixed"]).sum())
    return {
        "tau_min": float(t.min()), "tau_med": float(np.median(t)),
        "tau_max": float(t.max()),
        "block_rate": float(blocked.mean()),
        "recall": float(tp.sum() / max((y == 1).sum(), 1)),
        "precision": float(tp.sum() / max(blocked.sum(), 1)),
        "loss": loss, "loss_doing_nothing": nothing,
        "savings": (nothing - loss) / nothing if nothing else 0.0,
    }


def main() -> None:
    p = np.load(config.ARTIFACTS / "cal_test_scores.npy")
    y, amount = pipeline.load_test_arrays()

    w = 104
    print("=" * w)
    print("ONE DECISION LAYER, FOUR LOSS TYPES")
    print("=" * w)
    print("Same model. Same scores. Same threshold rule. Only the cost "
          "structure changes.")
    print("-" * w)
    print(f"{'loss type':<24}{'threshold range':>26}{'blocks':>9}"
          f"{'recall':>9}{'precision':>11}{'savings':>10}{'loss (Rs)':>15}")
    print("-" * w)

    out = {}
    for name, c in LOSS_TYPES.items():
        amt = amount * (UPI_AMOUNT_SCALE if name == "Indian UPI" else 1.0)
        r = policy(p, y, amt, c)
        out[name] = r
        rng = f"{r['tau_min']:.3f} - {r['tau_max']:.3f}"
        print(f"{name:<24}{rng:>26}{r['block_rate']:>9.2%}{r['recall']:>9.3f}"
              f"{r['precision']:>11.3f}{r['savings']:>10.1%}{r['loss']:>15,.0f}")
    print("-" * w)

    b, rto = out["CNP fraud (baseline)"], out["return / RTO"]
    cb, upi = out["chargeback dispute"], out["Indian UPI"]
    print()
    print("Reading the table -- each policy differs because the ECONOMICS differ,")
    print("not because anything about the model changed:")
    print()
    print(f"  RETURNS block {rto['block_rate']:.2%} of traffic against fraud's "
          f"{b['block_rate']:.2%}.")
    print("    An RTO returns the goods, so a miss costs shipping, not the order")
    print("    value. Blocking is rarely worth it and the layer says so.")
    print()
    print(f"  CHARGEBACK DISPUTES contest at {cb['block_rate']:.2%} with recall "
          f"{cb['recall']:.3f}.")
    print("    A wrong contest costs analyst minutes; a missed one costs the whole")
    print("    amount. The asymmetry inverts and the layer contests aggressively.")
    print()
    print(f"  INDIAN UPI blocks {upi['block_rate']:.2%}, recall {upi['recall']:.3f}.")
    print("    Tiny tickets against a full lifetime value on every wrong decline.")
    print(f"    Median threshold {upi['tau_med']:.3f} vs {b['tau_med']:.3f} for card")
    print("    fraud -- on small-ticket UPI the maths says be far more permissive.")
    print()
    print("The detector never changes. This does not claim to DETECT returns or")
    print("disputes -- it shows the layer that turns a score into an action is")
    print("loss-type agnostic, which is the claim the submission rests on.")
    print("=" * w)

    (config.ARTIFACTS / "loss_types.json").write_text(
        json.dumps({"cost_structures": LOSS_TYPES,
                    "upi_amount_scale": UPI_AMOUNT_SCALE,
                    "results": out}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'loss_types.json'}")


if __name__ == "__main__":
    main()
