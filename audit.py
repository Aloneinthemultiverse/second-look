"""Full model audit: the checks that could invalidate the headline numbers.

Aggregate metrics hide a lot. This script looks for the specific ways this
project could be quietly wrong:

  1. SEGMENTED CALIBRATION  the cost model computes P(fraud) x amount, so
     calibration must hold WITHIN amount buckets, not just on average. If the
     model is overconfident on large transactions, every rupee figure is wrong
     exactly where it matters most.
  2. DEGENERACY             is the model actually using the features, or riding
     one dominant column / emitting a near-constant score?
  3. ERROR ANATOMY          what do the false positives and false negatives
     look like, in rupees? This is what a risk analyst would ask first.
  4. CONFUSION + PR CURVE   the raw numbers behind precision/recall.
  5. FEATURE DRIFT          do the test-set feature distributions still look
     like training? Temporal splits invite drift.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

import calibrate
import config
import data
import features
import model

N_BUCKETS = 5


def segmented_calibration(p, y, amount, n=N_BUCKETS):
    """Calibration inside amount buckets. The cost model depends on this."""
    edges = np.quantile(amount, np.linspace(0, 1, n + 1))
    rows = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        m = (amount >= lo) & (amount <= hi if i == n - 1 else amount < hi)
        if m.sum() == 0 or y[m].sum() == 0:
            continue
        rows.append({
            "bucket": i + 1,
            "amount_lo": float(lo), "amount_hi": float(hi),
            "n": int(m.sum()),
            "observed_fraud_rate": float(y[m].mean()),
            "mean_predicted": float(p[m].mean()),
            "bias": float(p[m].mean() - y[m].mean()),
            "ece": calibrate.ece(y[m], p[m], bins=5),
        })
    return rows


def error_anatomy(p, y, amount, threshold):
    blocked = p >= threshold
    fp, fn = blocked & (y == 0), (~blocked) & (y == 1)
    tp = blocked & (y == 1)

    def stats(mask):
        a = amount[mask]
        if len(a) == 0:
            return {"n": 0}
        return {"n": int(len(a)), "total": float(a.sum()),
                "mean": float(a.mean()), "median": float(np.median(a)),
                "max": float(a.max()),
                "top_decile_share": float(np.sort(a)[-max(1, len(a) // 10):].sum() / a.sum())}

    return {"true_positive": stats(tp), "false_positive": stats(fp),
            "false_negative": stats(fn)}


def main() -> None:
    print("training ...")
    s = features.load_splits()
    X_tr, y_tr, _ = s["train"]
    X_ca, y_ca, _ = s["calib"]
    X_te, y_te, amt_te = s["test"]

    booster = lgb.train(
        model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=model.NUM_ROUNDS,
        valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p = calibrate.fit_calibrator(booster.predict(X_ca), y_ca)(booster.predict(X_te))

    rows = model.sweep(p, y_te, amt_te)
    best = min(rows, key=lambda r: r["rupee_loss"])
    t = best["threshold"]

    w = 88
    # --- 1. segmented calibration -----------------------------------------
    seg = segmented_calibration(p, y_te, amt_te)
    print("\n" + "=" * w)
    print("1. CALIBRATION WITHIN AMOUNT BUCKETS  (the cost model depends on this)")
    print("=" * w)
    print(f"{'bucket':<8}{'amount range (Rs)':>26}{'n':>9}{'observed':>11}"
          f"{'predicted':>11}{'bias':>10}{'ECE':>9}")
    print("-" * w)
    for r in seg:
        rng = f"{r['amount_lo']:,.0f} - {r['amount_hi']:,.0f}"
        print(f"{r['bucket']:<8}{rng:>26}{r['n']:>9,}{r['observed_fraud_rate']:>11.4%}"
              f"{r['mean_predicted']:>11.4%}{r['bias']:>+10.4%}{r['ece']:>9.4f}")
    biases = [abs(r["bias"]) for r in seg]
    print("-" * w)
    print(f"worst absolute bias {max(biases):.4%} in bucket "
          f"{seg[int(np.argmax(biases))]['bucket']}"
          f"   (overall ECE {calibrate.ece(y_te, p):.4f})")
    top = seg[-1]
    verdict = "OK" if abs(top["bias"]) < 0.01 else "PROBLEM"
    print(f"highest-amount bucket bias {top['bias']:+.4%}  -> {verdict}: "
          "the rupee figures are "
          + ("trustworthy where amounts are largest." if verdict == "OK"
             else "NOT trustworthy where amounts are largest."))

    # --- 2. degeneracy ------------------------------------------------------
    imp = pd.Series(booster.feature_importance("gain"),
                    index=booster.feature_name()).sort_values(ascending=False)
    share = imp / imp.sum()
    print("\n" + "=" * w)
    print("2. DEGENERACY CHECKS")
    print("=" * w)
    print(f"unique predicted scores      {len(np.unique(p)):,} / {len(p):,}")
    print(f"score range                  {p.min():.6f} - {p.max():.6f}")
    print(f"features with any gain       {(imp > 0).sum()} / {len(imp)}")
    print(f"top feature gain share       {share.iloc[0]:.1%}  ({share.index[0]})")
    print(f"top-5 feature gain share     {share.head(5).sum():.1%}")
    print("top 8 features: " + ", ".join(share.head(8).index))
    single = "PASS" if share.iloc[0] < 0.5 else "FAIL - one feature dominates"
    print(f"not single-feature dominated: {single}")

    # --- 3. error anatomy ---------------------------------------------------
    ea = error_anatomy(p, y_te, amt_te, t)
    print("\n" + "=" * w)
    print(f"3. ERROR ANATOMY at the rupee-optimal threshold ({t:.4f})")
    print("=" * w)
    print(f"{'group':<18}{'n':>9}{'total (Rs)':>16}{'mean':>12}{'median':>12}"
          f"{'max':>14}{'top-10% share':>15}")
    print("-" * w)
    for k, v in ea.items():
        if v["n"] == 0:
            continue
        print(f"{k:<18}{v['n']:>9,}{v['total']:>16,.0f}{v['mean']:>12,.0f}"
              f"{v['median']:>12,.0f}{v['max']:>14,.0f}{v['top_decile_share']:>15.1%}")
    print("-" * w)
    print("If false positives concentrate in high amounts, blocking is expensive")
    print("even when precision looks fine -- that is what happened on ULB.")

    # --- 4. confusion -------------------------------------------------------
    tn, fp_, fn_, tp_ = confusion_matrix(y_te, (p >= t).astype(int)).ravel()
    prec_c, rec_c, _ = precision_recall_curve(y_te, p)
    print("\n" + "=" * w)
    print("4. CONFUSION MATRIX")
    print("=" * w)
    print(f"  true negative  {tn:>8,}      false positive {fp_:>8,}")
    print(f"  false negative {fn_:>8,}      true positive  {tp_:>8,}")
    print(f"  precision {tp_/max(tp_+fp_,1):.4f}   recall {tp_/max(tp_+fn_,1):.4f}"
          f"   block rate {(tp_+fp_)/len(y_te):.4%}")
    print(f"  PR curve points: {len(prec_c):,}   baseline (random) precision "
          f"{y_te.mean():.4%}")

    # --- 5. feature drift ---------------------------------------------------
    print("\n" + "=" * w)
    print("5. NUMERIC FEATURE DRIFT, train vs test (standardised mean shift)")
    print("=" * w)
    num = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    shifts = {}
    for c in num:
        a, b = X_tr[c].astype(float), X_te[c].astype(float)
        sd = a.std()
        if sd and np.isfinite(sd) and sd > 0:
            shifts[c] = abs((b.mean() - a.mean()) / sd)
    ser = pd.Series(shifts).sort_values(ascending=False)
    print(f"numeric features checked {len(ser)}")
    print(f"features shifted > 0.5 sd  {(ser > 0.5).sum()}")
    print(f"features shifted > 1.0 sd  {(ser > 1.0).sum()}")
    print("largest shifts: " + ", ".join(f"{k} {v:.2f}sd" for k, v in ser.head(5).items()))
    print("=" * w)

    (config.ARTIFACTS / "audit.json").write_text(json.dumps({
        "segmented_calibration": seg,
        "error_anatomy": ea,
        "confusion": {"tn": int(tn), "fp": int(fp_), "fn": int(fn_), "tp": int(tp_)},
        "top_features": share.head(15).to_dict(),
        "drift_over_half_sd": int((ser > 0.5).sum()),
        "threshold": t,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'audit.json'}")


if __name__ == "__main__":
    main()
