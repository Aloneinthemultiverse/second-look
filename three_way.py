"""The three-way ALLOW / REVIEW / BLOCK breakdown, on the SHIPPED ensemble.

chow_band.py derives the review band from costs (Chow, 1970) rather than
inventing one, but it reads `cal_test_scores.npy` -- the single LightGBM from
model.py. The headline detector is the three-model ensemble in canonical.py,
which catches 1,771 frauds against the single model's 1,678.

So every three-way number published so far describes a weaker detector than the
one being submitted. Quoting the ensemble's 1,771 alongside the single model's
83.1% fraud reach would be mixing two runs, which is the kind of thing this
project has retracted before.

This regenerates the ensemble exactly as canonical.py does, SAVES its calibrated
test scores so no future script has to retrain to get them, and reports the
two-way and three-way breakdowns on the same scores.

Nothing is fitted on test. Platt is fit on the calibration slice; the Elkan
threshold and the Chow bands are closed forms from the cost structure.
"""
from __future__ import annotations

import json

import numpy as np

import calibrate
import canonical
import config
import features
from chow_band import chow_actions, realised_loss

REVIEW_COSTS = [25.0, 50.0, 150.0, 400.0, 1000.0]
DEFAULT_REVIEW_COST = 150.0
TEST_DAYS = 42


def ensemble_scores(s, seed):
    """Exactly canonical.run's model, but returning probabilities not metrics."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from leaderboard import LGB_P, XGB_P

    X_tr, y_tr, _ = s["train"]
    X_ca, y_ca, _ = s["calib"]
    X_te, _, _ = s["test"]
    R = canonical.ROUNDS

    m = lgb.train(dict(LGB_P, seed=seed, bagging_seed=seed,
                       feature_fraction_seed=seed),
                  lgb.Dataset(X_tr, label=y_tr), num_boost_round=R)
    xm = xgb.train(dict(XGB_P, seed=seed),
                   xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True),
                   num_boost_round=R)
    rf = RandomForestClassifier(n_estimators=R, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=seed).fit(canonical.numeric(X_tr), y_tr)

    d_ca = xgb.DMatrix(X_ca, enable_categorical=True)
    d_te = xgb.DMatrix(X_te, enable_categorical=True)
    ca = np.mean([m.predict(X_ca), xm.predict(d_ca),
                  rf.predict_proba(canonical.numeric(X_ca))[:, 1]], axis=0)
    te = np.mean([m.predict(X_te), xm.predict(d_te),
                  rf.predict_proba(canonical.numeric(X_te))[:, 1]], axis=0)
    return calibrate.fit_calibrator(ca, y_ca)(te)


def buckets(mask, y):
    f = int((mask & (y == 1)).sum())
    return {"n": int(mask.sum()), "frauds": f,
            "genuine": int((mask & (y == 0)).sum())}


def main() -> None:
    print("loading ...")
    s = features.load_splits()
    _, y_te, amt_te = s["test"]
    y_te = y_te.astype(int)
    F, n = int((y_te == 1).sum()), len(y_te)

    print(f"training the ensemble at {canonical.ROUNDS} trees "
          f"(seed {config.SEED}) ...")
    p = ensemble_scores(s, config.SEED)
    np.save(config.ARTIFACTS / "ensemble_test_scores.npy", p)
    print(f"saved ensemble_test_scores.npy -- no retrain needed next time")

    c_fp = amt_te * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amt_te + config.CHARGEBACK_FEE_INR
    blocked = p >= c_fp / (c_fp + c_fn)

    w = 96
    print("\n" + "=" * w)
    print("ALLOW / REVIEW / BLOCK ON THE SHIPPED ENSEMBLE")
    print("=" * w)
    print(f"test rows {n:,}   frauds {F:,}   fraud rate {y_te.mean():.3%}")
    print("-" * w)
    print(f"{'policy':<34}{'txns':>12}{'share':>9}{'frauds':>10}"
          f"{'of all':>9}{'genuine':>11}")
    print("-" * w)

    print("TWO-WAY  (Elkan tau*, the headline policy)")
    two = {}
    for nm, m in (("ALLOW", ~blocked), ("BLOCK", blocked)):
        b = buckets(m, y_te)
        two[nm] = b
        print(f"  {nm:<32}{b['n']:>12,}{b['n']/n:>9.2%}{b['frauds']:>10,}"
              f"{b['frauds']/F:>9.1%}{b['genuine']:>11,}")

    out = {"test_rows": n, "frauds_total": F, "two_way": two, "three_way": {}}

    for rc in REVIEW_COSTS:
        a, cm = chow_actions(p, amt_te, rc)
        print(f"\nTHREE-WAY  (Chow, review cost Rs {rc:,.0f})")
        row = {}
        for i, nm in enumerate(("ALLOW", "REVIEW", "BLOCK")):
            b = buckets(a == i, y_te)
            row[nm] = b
            print(f"  {nm:<32}{b['n']:>12,}{b['n']/n:>9.2%}{b['frauds']:>10,}"
                  f"{b['frauds']/F:>9.1%}{b['genuine']:>11,}")
        reached = row["REVIEW"]["frauds"] + row["BLOCK"]["frauds"]
        per_day = row["REVIEW"]["n"] / TEST_DAYS
        prec = row["BLOCK"]["frauds"] / max(row["BLOCK"]["n"], 1)
        loss = realised_loss(a, y_te, amt_te, rc)
        row.update({"fraud_reached": reached, "fraud_reached_rate": reached / F,
                    "reviews_per_day": per_day, "auto_block_precision": prec,
                    "realised_loss": loss})
        out["three_way"][str(rc)] = row
        print(f"  {'-> fraud reached (review+block)':<32}"
              f"{reached:>12,}{reached/F:>9.1%}")
        print(f"  {'-> reviews per day':<32}{per_day:>12,.0f}")
        print(f"  {'-> auto-block precision':<32}{prec:>12.1%}")
        print(f"  {'-> realised loss':<32}{loss:>12,.0f}")

    d = out["three_way"][str(DEFAULT_REVIEW_COST)]
    print("\n" + "-" * w)
    print("Reading it: auto-block stays small and near-certain, the review lane")
    print("concentrates the fraud the model cannot call, and everything else is")
    print("allowed. Adding a human lane lifts fraud reached from "
          f"{two['BLOCK']['frauds']/F:.1%} to {d['fraud_reached_rate']:.1%},")
    print(f"at the cost of {d['reviews_per_day']:,.0f} reviews a day.")
    print("Assumes analysts resolve reviews correctly; chow_capacity.py shows")
    print("what happens at 80% and 90% analyst accuracy, and it is not pretty.")
    print("=" * w)

    (config.ARTIFACTS / "three_way.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'three_way.json'}")


if __name__ == "__main__":
    main()
