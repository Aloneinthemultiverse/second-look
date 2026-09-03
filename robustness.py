"""Adversarial pressure, evaluated defence-only.

The deployment-evidence checklist (arXiv:2607.13078) asks for evidence under
adversarial pressure. Four of its five items were already met; this closes the
fifth without breaking Track 02's rule that "anything offense-capable is
disqualified".

WHAT IS DELIBERATELY NOT BUILT
No evasion search. No gradient or greedy optimiser that finds the smallest
perturbation flipping a decision. No synthetic fraud generator. Those are attack
tools regardless of intent, and building one to "test robustness" would be
exactly the thing the rules exclude.

WHAT IS BUILT -- three defensive audits that ask "where am I brittle?" rather
than "how would I break this?":

  1 SIGNAL LOSS       Blank one feature at a time (as if a cache missed, a
                      device fingerprint failed, or a field were unreliable) and
                      measure how many decisions flip and what it costs. This is
                      an availability and over-reliance audit.

  2 CONCENTRATION     What share of blocked transactions depend on a single
                      feature being present? High concentration is fragility --
                      operationally and adversarially.

  3 NOISE STABILITY   Add small random noise to numeric features and measure
                      decision churn. A model whose decisions move under
                      immaterial jitter is unstable regardless of attackers.

All three degrade the model's own inputs. None produce an adversarial example,
and none of the outputs would help anyone evade the system -- knowing that a
detector uses card velocity is not an exploit, it is a documented feature.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import calibrate
import config
import features
import model

TOP_N = 12          # audit the most important features
NOISE_LEVELS = [0.01, 0.05, 0.10]


def per_instance_tau(amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return c_fp / (c_fp + c_fn)


def decisions(p, amount):
    return p >= per_instance_tau(amount)


def rupee_loss(blocked, y, amount):
    fn, fp = (~blocked) & (y == 1), blocked & (y == 0)
    return float((amount[fn] + config.CHARGEBACK_FEE_INR).sum()
                 + (amount[fp] * config.MERCHANT_MARGIN_RATE
                    + config.CUSTOMER_LTV_INR).sum())


def main() -> None:
    print("training ...")
    s = features.load_splits()
    X_tr, y_tr, _ = s["train"]
    X_ca, y_ca, _ = s["calib"]
    X_te, y_te, amt_te = s["test"]

    b = lgb.train(model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    cal = calibrate.fit_calibrator(b.predict(X_ca), y_ca)

    p0 = cal(b.predict(X_te))
    d0 = decisions(p0, amt_te)
    loss0 = rupee_loss(d0, y_te, amt_te)
    print(f"baseline: {d0.sum():,} blocked, loss Rs {loss0:,.0f}")

    imp = pd.Series(b.feature_importance("gain"),
                    index=b.feature_name()).sort_values(ascending=False)
    audit_cols = list(imp.head(TOP_N).index)

    # --- 1. signal loss -----------------------------------------------------
    print(f"\nauditing signal loss across {len(audit_cols)} features ...")
    rows = []
    for c in audit_cols:
        Xm = X_te.copy()
        # blank the feature while preserving dtype -- assigning NaN to a
        # categorical column silently converts it to float and LightGBM then
        # rejects the frame for mismatched categorical features
        if str(X_te[c].dtype) == "category":
            Xm[c] = pd.Categorical([None] * len(Xm),
                                   categories=X_te[c].cat.categories)
        else:
            Xm[c] = np.nan                  # feature unavailable at inference
        pm = cal(b.predict(Xm))
        dm = decisions(pm, amt_te)
        flips = int((dm != d0).sum())
        rows.append({
            "feature": c,
            "gain_share": float(imp[c] / imp.sum()),
            "decisions_flipped": flips,
            "flip_rate": flips / len(d0),
            "blocked_after": int(dm.sum()),
            "loss_after": rupee_loss(dm, y_te, amt_te),
        })
    rows.sort(key=lambda r: -r["loss_after"])

    w = 96
    print("\n" + "=" * w)
    print("1. SIGNAL LOSS -- one feature made unavailable at a time")
    print("=" * w)
    print(f"{'feature':<20}{'gain share':>12}{'flipped':>10}{'flip rate':>12}"
          f"{'blocked':>10}{'loss (Rs)':>16}{'delta':>14}")
    print("-" * w)
    for r in rows:
        print(f"{r['feature']:<20}{r['gain_share']:>12.1%}"
              f"{r['decisions_flipped']:>10,}{r['flip_rate']:>12.2%}"
              f"{r['blocked_after']:>10,}{r['loss_after']:>16,.0f}"
              f"{r['loss_after'] - loss0:>+14,.0f}")
    print("-" * w)
    worst = rows[0]
    print(f"worst single-signal loss: {worst['feature']} "
          f"(+Rs {worst['loss_after'] - loss0:,.0f}, "
          f"{worst['flip_rate']:.2%} of decisions change)")

    # --- 2. concentration ---------------------------------------------------
    print("\n" + "=" * w)
    print("2. CONCENTRATION -- how dependent is the model on any one signal?")
    print("=" * w)
    share = imp / imp.sum()
    print(f"top feature gain share        {share.iloc[0]:.1%}  ({share.index[0]})")
    print(f"top-3 combined                {share.head(3).sum():.1%}")
    print(f"top-10 combined               {share.head(10).sum():.1%}")
    print(f"features contributing at all  {(imp > 0).sum()} / {len(imp)}")
    hhi = float((share ** 2).sum())
    print(f"Herfindahl index              {hhi:.4f}  "
          f"({'concentrated' if hhi > 0.15 else 'well diversified'})")
    max_delta = max(r["loss_after"] - loss0 for r in rows)
    print(f"worst-case single-signal cost {max_delta / loss0:+.2%} of baseline loss")

    # --- 3. noise stability -------------------------------------------------
    print("\n" + "=" * w)
    print("3. NOISE STABILITY -- decision churn under immaterial jitter")
    print("=" * w)
    num_cols = [c for c in X_te.columns
                if pd.api.types.is_numeric_dtype(X_te[c])]
    rng = np.random.default_rng(config.SEED)
    print(f"{'noise (sd)':>12}{'decisions changed':>20}{'churn rate':>14}"
          f"{'loss (Rs)':>16}{'delta':>14}")
    print("-" * w)
    noise_rows = []
    for lvl in NOISE_LEVELS:
        Xn = X_te.copy()
        for c in num_cols:
            col = Xn[c].astype(float)
            sd = col.std()
            if sd and np.isfinite(sd):
                Xn[c] = col + rng.normal(0, lvl * sd, len(col))
        pn = cal(b.predict(Xn))
        dn = decisions(pn, amt_te)
        churn = int((dn != d0).sum())
        ln = rupee_loss(dn, y_te, amt_te)
        noise_rows.append({"noise_sd": lvl, "changed": churn,
                           "churn_rate": churn / len(d0), "loss": ln})
        print(f"{lvl:>12.0%}{churn:>20,}{churn/len(d0):>14.2%}"
              f"{ln:>16,.0f}{ln - loss0:>+14,.0f}")
    print("-" * w)
    print("Interpretation: low churn under small jitter means decisions are not")
    print("balanced on knife-edges. It does not prove resistance to a motivated")
    print("adversary -- that would require evasion testing, which is out of")
    print("scope under the defence-only rule.")
    print("=" * w)

    (config.ARTIFACTS / "robustness.json").write_text(
        json.dumps({"baseline_loss": loss0,
                    "baseline_blocked": int(d0.sum()),
                    "signal_loss": rows,
                    "concentration": {"top_share": float(share.iloc[0]),
                                      "top3": float(share.head(3).sum()),
                                      "hhi": hhi},
                    "noise": noise_rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'robustness.json'}")


if __name__ == "__main__":
    main()
