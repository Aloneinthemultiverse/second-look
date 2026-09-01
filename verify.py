"""Verification: do the headline numbers survive contact with other conditions?

Three checks, because a single split with a single seed proves very little:

  1. LATENCY    we claim the 77 online features are usable inside a checkout
                budget. Measure single-row inference instead of asserting it.
  2. SEEDS      retrain across seeds. If the rupee-optimal threshold or the
                F1-vs-rupee gap swings wildly, the headline is noise.
  3. SPLITS     move the train/test boundary to different points in time. The
                finding should hold on different periods, not just this one.

Loads the CSV once; everything else reuses it.
"""
from __future__ import annotations

import json
import statistics as stats
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

import calibrate
import config
import data
import features
import model
from sensitivity import optimal

SEEDS = [42, 7, 2024]
SPLITS = [(0.60, 0.10), (0.70, 0.10), (0.80, 0.10)]
LATENCY_SAMPLES = 300


def _split_at(df, train_frac, calib_frac):
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    i_tr, i_ca = int(n * train_frac), int(n * (train_frac + calib_frac))
    return df.iloc[:i_tr], df.iloc[i_tr:i_ca], df.iloc[i_ca:]


def run_once(raw, cols, seed=config.SEED, train_frac=None, calib_frac=None):
    """Train + calibrate + choose thresholds under one configuration."""
    train_frac = train_frac if train_frac is not None else config.TRAIN_FRAC
    calib_frac = calib_frac if calib_frac is not None else config.CALIB_FRAC
    tr_df, ca_df, te_df = _split_at(raw, train_frac, calib_frac)

    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te_df, cols, cats)

    params = dict(model.PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    booster = lgb.train(
        params, lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=model.NUM_ROUNDS,
        valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p = calibrate.fit_calibrator(booster.predict(X_ca), y_ca)(booster.predict(X_te))

    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.9999, 300)))
    t_rupee, loss_rupee = optimal(p, y_te, amt_te,
                                  config.MERCHANT_MARGIN_RATE,
                                  config.CUSTOMER_LTV_INR,
                                  config.CHARGEBACK_FEE_INR, grid)
    best_f1, t_f1 = -1.0, None
    for t in grid:
        _, _, f1, _ = precision_recall_fscore_support(
            y_te, (p >= t).astype(int), average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, t_f1 = f1, float(t)
    loss_f1 = model.expected_loss(p, y_te, amt_te, t_f1)

    return {
        "pr_auc": float(average_precision_score(y_te, p)),
        "ece": calibrate.ece(y_te, p),
        "t_rupee": t_rupee,
        "t_f1": t_f1,
        "loss_rupee": loss_rupee,
        "loss_f1": loss_f1,
        "gap": loss_f1 - loss_rupee,
        "test_rows": int(len(y_te)),
    }, booster, X_te


def measure_latency(booster, X_te) -> dict:
    """Single-row prediction time -- the checkout case, not batch scoring."""
    rows = [X_te.iloc[[i]] for i in range(min(LATENCY_SAMPLES, len(X_te)))]
    booster.predict(rows[0])  # warm up
    times = []
    for r in rows:
        t0 = time.perf_counter()
        booster.predict(r)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "n": len(times),
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "max_ms": times[-1],
    }


def main() -> None:
    print("loading data once ...")
    raw = data.load_raw()
    cols = data.audit_leakage(raw)["online"]

    # --- seeds -------------------------------------------------------------
    print(f"\ntraining across {len(SEEDS)} seeds ...")
    seed_runs = []
    booster = X_te = None
    for s in SEEDS:
        r, booster, X_te = run_once(raw, cols, seed=s)
        seed_runs.append(r)
        print(f"  seed {s:<6} PR-AUC {r['pr_auc']:.4f}  T* {r['t_rupee']:.4f}  "
              f"gap Rs {r['gap']:,.0f}")

    # --- splits ------------------------------------------------------------
    print(f"\ntraining across {len(SPLITS)} split points ...")
    split_runs = []
    for tf, cf in SPLITS:
        r, _, _ = run_once(raw, cols, train_frac=tf, calib_frac=cf)
        r["train_frac"] = tf
        split_runs.append(r)
        print(f"  train {tf:.0%}  test {1-tf-cf:.0%}  PR-AUC {r['pr_auc']:.4f}  "
              f"T* {r['t_rupee']:.4f}  gap Rs {r['gap']:,.0f}")

    # --- latency -----------------------------------------------------------
    print("\nmeasuring single-row inference latency ...")
    lat = measure_latency(booster, X_te)

    # --- report ------------------------------------------------------------
    def band(runs, key):
        v = [r[key] for r in runs]
        sd = stats.stdev(v) if len(v) > 1 else 0.0
        return min(v), max(v), stats.mean(v), sd

    w = 78
    print("\n" + "=" * w)
    print("VERIFICATION")
    print("=" * w)
    print(f"{'quantity':<24}{'min':>13}{'max':>13}{'mean':>13}{'sd':>13}")
    print("-" * w)
    for label, key, fmt in (("PR-AUC (seeds)", "pr_auc", ".4f"),
                            ("rupee threshold", "t_rupee", ".4f"),
                            ("F1-vs-rupee gap", "gap", ",.0f")):
        lo, hi, mu, sd = band(seed_runs, key)
        print(f"{label:<24}{lo:>13{fmt}}{hi:>13{fmt}}{mu:>13{fmt}}{sd:>13{fmt}}")
    print("-" * w)
    print("across split points (different periods of the data):")
    for r in split_runs:
        print(f"  train {r['train_frac']:.0%}  test rows {r['test_rows']:>7,}  "
              f"PR-AUC {r['pr_auc']:.4f}  ECE {r['ece']:.4f}  "
              f"gap Rs {r['gap']:>10,.0f}")
    gaps = [r["gap"] for r in split_runs]
    print(f"  gap is positive in {sum(g > 0 for g in gaps)}/{len(gaps)} split points")
    print("-" * w)
    print(f"single-row inference: p50 {lat['p50_ms']:.2f} ms   "
          f"p95 {lat['p95_ms']:.2f} ms   max {lat['max_ms']:.2f} ms  "
          f"(n={lat['n']})")
    print("  NOTE: this is model scoring only. Feature retrieval from a cache")
    print("  is NOT included and would add to the budget in production.")
    print("=" * w)

    (config.ARTIFACTS / "verification.json").write_text(
        json.dumps({"seeds": seed_runs, "splits": split_runs, "latency": lat},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'verification.json'}")


if __name__ == "__main__":
    main()
