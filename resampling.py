"""Does resampling help once errors are priced in rupees?

A reviewer's notebook used the standard imbalanced-learning recipe -- SMOTEENN
oversampling plus class_weight='balanced' -- and reported large recall gains.
This project never tested resampling at all, so it is a real gap.

The reason to be suspicious rather than just adopt it: resampling deliberately
changes the class prior. A model trained on rebalanced data no longer estimates
P(fraud) on the real population -- it estimates it on an invented one. Every
threshold in this project is derived from calibrated probabilities via
tau*(x) = C_FP/(C_FP + C_FN). If calibration breaks, the cost machinery breaks
with it, and a recall improvement measured at a fixed 0.5 threshold would be
telling us nothing about money.

So each arm is measured twice:
  - recall and precision at a fixed 0.5 threshold (how the notebook measured)
  - calibration error and rupee loss at the cost-optimal threshold (how this
    project measures)

If resampling wins on the first and loses on the second, that is the finding.
"""
from __future__ import annotations

import json

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

import calibrate
import config
import features
from leaderboard import evaluate

SUBSAMPLE = 150_000     # SMOTEENN on 400k x 77 is not tractable in this window


def to_numeric(df):
    return df.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c).fillna(-999)


def main() -> None:
    print("loading ...")
    s = features.load_splits()
    X_tr, y_tr, _ = s["train"]
    X_ca, y_ca, _ = s["calib"]
    X_te, y_te, amt_te = s["test"]
    N_tr, N_ca, N_te = to_numeric(X_tr), to_numeric(X_ca), to_numeric(X_te)

    # subsample the training set so SMOTEENN finishes; keep every fraud row
    rng = np.random.default_rng(config.SEED)
    fraud_idx = np.where(y_tr == 1)[0]
    genuine_idx = np.where(y_tr == 0)[0]
    keep_genuine = rng.choice(genuine_idx,
                              size=min(SUBSAMPLE - len(fraud_idx), len(genuine_idx)),
                              replace=False)
    sub = np.concatenate([fraud_idx, keep_genuine])
    rng.shuffle(sub)
    Ns, ys = N_tr.iloc[sub], y_tr[sub]
    print(f"training subsample {len(ys):,} rows, fraud {ys.mean():.2%} "
          f"(all {len(fraud_idx):,} frauds kept)")

    def rf(**kw):
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                      max_features="sqrt", n_jobs=-1,
                                      random_state=config.SEED, **kw)

    arms = {}

    print("baseline RF ...")
    arms["baseline RF"] = rf().fit(Ns, ys)

    print("RF + class_weight='balanced' ...")
    arms["class_weight=balanced"] = rf(class_weight="balanced").fit(Ns, ys)

    print("SMOTE ...")
    Xs, yss = SMOTE(random_state=config.SEED).fit_resample(Ns, ys)
    print(f"   after SMOTE: {len(yss):,} rows, fraud {yss.mean():.2%}")
    arms["SMOTE"] = rf().fit(Xs, yss)

    print("SMOTEENN (this is the slow one) ...")
    Xe, yse = SMOTEENN(random_state=config.SEED).fit_resample(Ns, ys)
    print(f"   after SMOTEENN: {len(yse):,} rows, fraud {yse.mean():.2%}")
    arms["SMOTEENN"] = rf().fit(Xe, yse)

    rows = {}
    for name, m in arms.items():
        p_ca_raw = m.predict_proba(N_ca)[:, 1]
        p_te_raw = m.predict_proba(N_te)[:, 1]

        # how the notebook measures: fixed 0.5 threshold, uncalibrated
        pred05 = (p_te_raw >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, pred05, average="binary", zero_division=0)

        # how this project measures: calibrate, then cost-optimal threshold
        p_cal = calibrate.fit_calibrator(p_ca_raw, y_ca)(p_te_raw)
        cost = evaluate(p_cal, y_te, amt_te)

        rows[name] = {
            "recall_at_0.5": float(rec), "precision_at_0.5": float(prec),
            "f1_at_0.5": float(f1),
            "ece_uncalibrated": calibrate.ece(y_te, p_te_raw),
            "ece_calibrated": calibrate.ece(y_te, p_cal),
            "mean_predicted": float(p_te_raw.mean()),
            **cost,
        }

    w = 108
    print("\n" + "=" * w)
    print("HOW THE NOTEBOOK MEASURES  (fixed 0.5 threshold, no calibration)")
    print("=" * w)
    print(f"{'arm':<26}{'recall':>10}{'precision':>12}{'F1':>10}"
          f"{'mean predicted P':>20}{'ECE (raw)':>13}")
    print("-" * w)
    for k, v in rows.items():
        print(f"{k:<26}{v['recall_at_0.5']:>10.3f}{v['precision_at_0.5']:>12.3f}"
              f"{v['f1_at_0.5']:>10.3f}{v['mean_predicted']:>20.4f}"
              f"{v['ece_uncalibrated']:>13.4f}")
    print("-" * w)
    print(f"true fraud rate in test: {y_te.mean():.4f}  -- compare to "
          f"'mean predicted P'")

    print("\n" + "=" * w)
    print("HOW THIS PROJECT MEASURES  (calibrated, cost-optimal threshold)")
    print("=" * w)
    print(f"{'arm':<26}{'ROC-AUC':>10}{'frauds caught':>16}{'recall':>9}"
          f"{'FPR':>8}{'ECE':>9}{'rupee loss':>16}")
    print("-" * w)
    for k, v in rows.items():
        print(f"{k:<26}{v['roc_auc']:>10.4f}"
              f"{v['frauds_caught']:>9,}/{v['frauds_total']:<6,}"
              f"{v['recall_count']:>9.3f}{v['fpr']:>8.2%}"
              f"{v['ece_calibrated']:>9.4f}{v['rupee_loss']:>16,.0f}")
    print("-" * w)

    base = rows["baseline RF"]
    best_recall = max(rows, key=lambda k: rows[k]["recall_at_0.5"])
    best_money = min(rows, key=lambda k: rows[k]["rupee_loss"])
    print(f"best recall at 0.5:        {best_recall} "
          f"({rows[best_recall]['recall_at_0.5']:.3f} vs baseline "
          f"{base['recall_at_0.5']:.3f})")
    print(f"best by rupee loss:        {best_money} "
          f"(Rs {rows[best_money]['rupee_loss']:,.0f})")
    if best_recall != best_money:
        print()
        print("The two metrics disagree. Resampling raises recall at a fixed")
        print("threshold by inflating predicted probabilities -- see 'mean")
        print("predicted P' against the true fraud rate -- which is a shifted")
        print("prior, not better discrimination. Once calibrated and priced,")
        print("the ranking changes.")
    print("=" * w)

    (config.ARTIFACTS / "resampling.json").write_text(
        json.dumps({"true_fraud_rate": float(y_te.mean()), "results": rows},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'resampling.json'}")


if __name__ == "__main__":
    main()
