"""What does the inference-latency constraint actually cost us?

We restricted the model to ~77 features computable in <50ms at checkout, and
excluded Vesta's 339 V_* offline aggregates. Almost every public notebook on
this dataset uses them. That is legitimate for a Kaggle score and illegitimate
for a production scorer.

This script quantifies the trade rather than asserting it: train both, compare
PR-AUC and rupee loss at each model's own rupee-optimal threshold.
"""
from __future__ import annotations

import json

import numpy as np

import config
import pipeline
from sensitivity import describe, optimal


def main() -> None:
    results = {}
    for tag, kwargs in (
        ("online only (production-feasible)", dict()),
        ("online + V_* (offline aggregates)", dict(include_latency=True)),
    ):
        print(f"training: {tag} ...")
        r = pipeline.train_and_score(**kwargs)
        p, y, amount = r["scores"], r["y"], r["amount"]
        grid = np.unique(np.quantile(p, np.linspace(0.50, 0.9999, 300)))
        t, loss = optimal(p, y, amount,
                          config.MERCHANT_MARGIN_RATE,
                          config.CUSTOMER_LTV_INR,
                          config.CHARGEBACK_FEE_INR,
                          grid)
        prec, rec, blocked = describe(p, y, t)
        results[tag] = {
            "n_features": r["n_features"], "pr_auc": r["pr_auc"],
            "threshold": t, "rupee_loss": loss,
            "precision": prec, "recall": rec, "blocked_rate": blocked,
        }
        print(f"  {r['n_features']} features   PR-AUC {r['pr_auc']:.4f}   "
              f"loss Rs {loss:,.0f}")

    a, b = results.values()
    ka, kb = results.keys()

    w = 78
    print("\n" + "=" * w)
    print("COST OF THE INFERENCE-LATENCY CONSTRAINT")
    print("=" * w)
    print(f"{'feature set':<38}{'n':>6}{'PR-AUC':>9}{'recall':>9}{'loss (Rs)':>16}")
    print("-" * w)
    for k, r in results.items():
        print(f"{k:<38}{r['n_features']:>6}{r['pr_auc']:>9.4f}"
              f"{r['recall']:>9.3f}{r['rupee_loss']:>16,.0f}")
    print("-" * w)
    print(f"PR-AUC given up      {b['pr_auc'] - a['pr_auc']:+.4f}")
    print(f"extra rupee loss     {a['rupee_loss'] - b['rupee_loss']:+,.0f}")
    print()
    print("The V_* model is not deployable: those aggregates cannot be computed")
    print("inside a checkout latency budget. We report the gap rather than")
    print("quoting the unreachable number.")
    print("=" * w)

    (config.ARTIFACTS / "restriction_cost.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'restriction_cost.json'}")


if __name__ == "__main__":
    main()
