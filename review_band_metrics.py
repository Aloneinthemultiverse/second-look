"""Does the investigation evidence add anything the score did not already know?

The investigator layer is justified on legibility grounds: analysts have a fixed
budget, so reviewed cases must arrive readable. That is a real argument, but it
is not the same as the evidence carrying information.

This measures it honestly. Within the review band:

  baseline   rank cases by the model score alone
  +playbook  rank by a small logistic regression over the five playbook lookups
  combined   score + playbook features together

Fitted on the CALIBRATION slice's review band, evaluated on the TEST slice's
review band. If the playbook adds nothing, the numbers will say so and the
layer's claim stays limited to legibility.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calibrate
import config
import data
import features
import model
from playbook import Playbook

REVIEW_LO, REVIEW_HI = 0.20, 0.80
NUMERIC = ["txn_1h", "txn_24h", "amt_24h", "card_seen", "device_txns",
           "device_age_days", "distinct_cards_on_device", "email_lift",
           "amount_z", "amount_history", "distance"]


def playbook_frame(pb: Playbook, df: pd.DataFrame) -> pd.DataFrame:
    """Playbook features, keeping missingness as information.

    An earlier version median-imputed every NaN. That was wrong: a NaN in
    device_txns means "device never seen before", which is itself one of the
    strongest signals available. Imputing it away deleted the thing the lookup
    exists to surface. Missing values are now kept as NaN (LightGBM splits on
    them natively) and an explicit indicator column is added alongside.
    """
    rows = [pb.run(r) for _, r in df.iterrows()]
    out = pd.DataFrame(rows)
    for c in NUMERIC:
        if c not in out:
            out[c] = np.nan
    out = out[NUMERIC].astype(float)
    for c in NUMERIC:
        out[f"{c}__missing"] = out[c].isna().astype(int)
    return out


def main() -> None:
    print("training detector ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    tr_df, ca_df, te_df = data.temporal_split(raw)
    cols = audit["online"]

    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, _, _ = features.build(te_df, cols, cats)

    b = lgb.train(model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    cal = calibrate.fit_calibrator(b.predict(X_ca), y_ca)
    p_ca, p_te = cal(b.predict(X_ca)), cal(b.predict(X_te))

    band_ca = (p_ca >= REVIEW_LO) & (p_ca < REVIEW_HI)
    band_te = (p_te >= REVIEW_LO) & (p_te < REVIEW_HI)
    print(f"review band: calib {band_ca.sum():,} cases "
          f"({y_ca[band_ca].mean():.2%} fraud), "
          f"test {band_te.sum():,} cases ({y_te[band_te].mean():.2%} fraud)")

    print("running playbook over the band (this is the slow part) ...")
    pb = Playbook(raw, tr_df)
    F_ca = playbook_frame(pb, ca_df[band_ca])
    F_te = playbook_frame(pb, te_df[band_te])
    yb_ca, yb_te = y_ca[band_ca], y_te[band_te]
    sb_ca, sb_te = p_ca[band_ca], p_te[band_te]

    def scores(name, tr_X, te_X):
        """Small gradient booster, not logistic regression.

        Velocity counts and z-scores do not relate to fraud linearly, and the
        band is small (1,198 calibration cases), so the model is deliberately
        tiny to limit overfitting.
        """
        m = lgb.LGBMClassifier(n_estimators=120, num_leaves=8,
                               min_child_samples=40, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=config.SEED, verbose=-1)
        m.fit(np.asarray(tr_X, dtype=float), yb_ca)
        pr = m.predict_proba(np.asarray(te_X, dtype=float))[:, 1]
        return {"name": name,
                "roc_auc": float(roc_auc_score(yb_te, pr)),
                "pr_auc": float(average_precision_score(yb_te, pr))}

    results = [
        {"name": "model score alone",
         "roc_auc": float(roc_auc_score(yb_te, sb_te)),
         "pr_auc": float(average_precision_score(yb_te, sb_te))},
        scores("playbook features only", F_ca, F_te),
        scores("score + playbook",
               np.column_stack([sb_ca, F_ca]), np.column_stack([sb_te, F_te])),
    ]

    w = 74
    print("\n" + "=" * w)
    print("DOES THE INVESTIGATION EVIDENCE ADD SIGNAL INSIDE THE REVIEW BAND?")
    print("=" * w)
    print(f"{'ranking by':<30}{'ROC-AUC':>12}{'PR-AUC':>12}{'vs score alone':>18}")
    print("-" * w)
    base = results[0]
    for r in results:
        d = r["roc_auc"] - base["roc_auc"]
        print(f"{r['name']:<30}{r['roc_auc']:>12.4f}{r['pr_auc']:>12.4f}"
              f"{d:>+18.4f}")
    print("-" * w)
    print(f"band base rate {yb_te.mean():.2%}  (random ROC-AUC = 0.5000)")

    gain = results[2]["roc_auc"] - base["roc_auc"]
    if gain > 0.01:
        verdict = ("The playbook adds ranking signal beyond the score. The layer "
                   "is not only legibility.")
    elif gain > -0.01:
        verdict = ("The playbook adds no meaningful ranking signal. Its value is "
                   "legibility for the analyst, which is what we claimed -- we do "
                   "NOT claim it improves detection.")
    else:
        verdict = ("The playbook degrades ranking; use it for presentation only.")
    print(verdict)
    print("=" * w)

    (config.ARTIFACTS / "review_band_metrics.json").write_text(
        json.dumps({"band": [REVIEW_LO, REVIEW_HI],
                    "n_calib": int(band_ca.sum()), "n_test": int(band_te.sum()),
                    "band_fraud_rate_test": float(yb_te.mean()),
                    "results": results, "verdict": verdict}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'review_band_metrics.json'}")


if __name__ == "__main__":
    main()
