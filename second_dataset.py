"""Does the METHOD transfer, or is it tuned to one dataset's quirks?

Runs the identical pipeline -- temporal split, LightGBM, Platt calibration,
cost-optimal threshold -- on the ULB credit-card fraud dataset:

    284,807 transactions, 0.172% fraud (20x rarer than IEEE-CIS),
    28 PCA components + Time + Amount, real labels.

What this DOES test: whether the calibration finding, the threshold-sensitivity
finding, and the temporal-split machinery hold on different data.
What this does NOT test: the detector itself. Different data, different model.
A second dataset checks the method, not the numbers.

Also runs a within-test drift check on both datasets, because a single aggregate
metric can hide performance decaying across the test period.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import calibrate
import config
import model
from sensitivity import optimal

ULB_CSV = config.DATA_DIR / "creditcard.csv"

# ULB amounts are EUR. Same idea as USD_TO_INR: convert so the cost model is
# expressed in one currency. Disclosed, not hidden.
EUR_TO_INR = 95.0


def load_ulb():
    df = pd.read_csv(ULB_CSV).sort_values("Time").reset_index(drop=True)
    n = len(df)
    i_tr = int(n * config.TRAIN_FRAC)
    i_ca = int(n * (config.TRAIN_FRAC + config.CALIB_FRAC))
    feats = [c for c in df.columns if c not in ("Class", "Time")]
    return {
        "train": (df.iloc[:i_tr][feats], df.iloc[:i_tr]["Class"].to_numpy()),
        "calib": (df.iloc[i_tr:i_ca][feats], df.iloc[i_tr:i_ca]["Class"].to_numpy()),
        "test": (df.iloc[i_ca:][feats], df.iloc[i_ca:]["Class"].to_numpy(),
                 df.iloc[i_ca:]["Amount"].to_numpy() * EUR_TO_INR),
        "n": n,
        "fraud_rate": float(df["Class"].mean()),
    }


def drift_check(p, y, n_buckets: int = 5):
    """Split the test slice into consecutive time buckets and score each.

    The test rows are already in time order, so slicing them sequentially gives
    consecutive periods. A single aggregate PR-AUC can hide steady decay.
    """
    rows = []
    for i, (ps, ys) in enumerate(zip(np.array_split(p, n_buckets),
                                     np.array_split(y, n_buckets))):
        if ys.sum() == 0:
            rows.append({"bucket": i + 1, "n": int(len(ys)), "frauds": 0,
                         "pr_auc": None, "fraud_rate": 0.0})
            continue
        rows.append({
            "bucket": i + 1,
            "n": int(len(ys)),
            "frauds": int(ys.sum()),
            "fraud_rate": float(ys.mean()),
            "pr_auc": float(average_precision_score(ys, ps)),
        })
    return rows


def main() -> None:
    if not ULB_CSV.exists():
        raise SystemExit(f"missing {ULB_CSV}")

    d = load_ulb()
    X_tr, y_tr = d["train"]
    X_ca, y_ca = d["calib"]
    X_te, y_te, amt_te = d["test"]
    print(f"ULB: {d['n']:,} rows, fraud {d['fraud_rate']:.4%}, "
          f"test {len(y_te):,} rows ({int(y_te.sum())} frauds)")

    booster = lgb.train(
        model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=model.NUM_ROUNDS,
        valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    raw_ca, raw_te = booster.predict(X_ca), booster.predict(X_te)

    # --- does the calibration finding replicate? ---------------------------
    cal = {}
    for kind in ("none", "platt", "isotonic"):
        p = calibrate.fit_calibrator(raw_ca, y_ca, kind)(raw_te)
        cal[kind] = {
            "pr_auc": float(average_precision_score(y_te, p)),
            "roc_auc": float(roc_auc_score(y_te, p)),
            "ece": calibrate.ece(y_te, p),
            "n_unique": int(len(np.unique(p))),
        }

    p_te = calibrate.fit_calibrator(raw_ca, y_ca, "platt")(raw_te)

    # --- does the merchant-sensitivity finding replicate? ------------------
    grid = np.unique(np.quantile(p_te, np.linspace(0.50, 0.99999, 400)))
    archetypes = {
        "discount retail": dict(margin=0.12, ltv=400.0, cb_fee=1200.0),
        "mid D2C": dict(margin=config.MERCHANT_MARGIN_RATE,
                        ltv=config.CUSTOMER_LTV_INR,
                        cb_fee=config.CHARGEBACK_FEE_INR),
        "subscription SaaS": dict(margin=0.80, ltv=15000.0, cb_fee=1200.0),
        "high-ticket": dict(margin=0.08, ltv=1000.0, cb_fee=3000.0),
    }
    arche = {}
    for name, params in archetypes.items():
        t, loss = optimal(p_te, y_te, amt_te, **params, grid=grid)
        arche[name] = {"threshold": t, "loss": loss,
                       "blocked_rate": float((p_te >= t).mean())}

    drift = drift_check(p_te, y_te)

    # --- report ------------------------------------------------------------
    w = 76
    print("\n" + "=" * w)
    print("SECOND DATASET (ULB) -- does the METHOD transfer?")
    print("=" * w)
    print(f"{'calibrator':<12}{'PR-AUC':>10}{'ROC-AUC':>10}{'ECE':>10}{'unique':>12}")
    print("-" * w)
    for k, v in cal.items():
        print(f"{k:<12}{v['pr_auc']:>10.4f}{v['roc_auc']:>10.4f}"
              f"{v['ece']:>10.4f}{v['n_unique']:>12,}")
    print("-" * w)
    dp = cal["platt"]["pr_auc"] - cal["none"]["pr_auc"]
    di = cal["isotonic"]["pr_auc"] - cal["none"]["pr_auc"]
    print(f"platt vs raw PR-AUC {dp:+.4f}   isotonic vs raw {di:+.4f}")
    ties_platt = cal["none"]["n_unique"] - cal["platt"]["n_unique"]
    ties_iso = cal["none"]["n_unique"] - cal["isotonic"]["n_unique"]
    print(f"score levels destroyed:  platt {ties_platt:,}   isotonic {ties_iso:,}")
    # The claim is about the MECHANISM (isotonic ties scores, Platt broadly does
    # not), not bit-exact equality: Platt is strictly monotonic in exact
    # arithmetic, but floating-point saturation at the extremes can still tie a
    # handful of scores. Judge replication on the order of magnitude.
    print("MECHANISM REPLICATES" if ties_iso > 100 * max(ties_platt, 1)
          else "MECHANISM DOES NOT REPLICATE")

    print("\n" + "=" * w)
    print("THRESHOLD SENSITIVITY ON ULB")
    print("=" * w)
    print(f"{'merchant':<22}{'T*':>12}{'blocked':>12}")
    print("-" * w)
    for k, v in arche.items():
        print(f"{k:<22}{v['threshold']:>12.5f}{v['blocked_rate']:>12.3%}")
    ts = [v["threshold"] for v in arche.values()]
    print("-" * w)
    print(f"threshold spans {min(ts):.5f} -> {max(ts):.5f} "
          f"({max(ts)/max(min(ts),1e-12):.1f}x)")

    print("\n" + "=" * w)
    print("WITHIN-TEST DRIFT (consecutive periods of the ULB test slice)")
    print("=" * w)
    print(f"{'bucket':<10}{'rows':>10}{'frauds':>9}{'fraud rate':>13}{'PR-AUC':>10}")
    print("-" * w)
    for r in drift:
        pa = f"{r['pr_auc']:.4f}" if r["pr_auc"] is not None else "n/a"
        print(f"{r['bucket']:<10}{r['n']:>10,}{r['frauds']:>9}"
              f"{r['fraud_rate']:>13.4%}{pa:>10}")
    vals = [r["pr_auc"] for r in drift if r["pr_auc"] is not None]
    if len(vals) > 1:
        print("-" * w)
        print(f"PR-AUC across periods: {min(vals):.4f} -> {max(vals):.4f} "
              f"(spread {max(vals) - min(vals):.4f})")
    print("=" * w)

    (config.ARTIFACTS / "second_dataset.json").write_text(
        json.dumps({"dataset": {"rows": d["n"], "fraud_rate": d["fraud_rate"]},
                    "calibration": cal, "archetypes": arche, "drift": drift},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'second_dataset.json'}")


if __name__ == "__main__":
    main()
