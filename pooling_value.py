"""What is pooling data actually worth? (The prior question behind federation.)

Federated learning exists because pooling data across institutions helps, but raw
data cannot be shared. Recent work -- FedFraud (arXiv:2604.23437), the ACM CIKM
2025 empirical study of federated gradient boosting in banking, J.P. Morgan's
Project Aikya -- all rest on that premise.

We are NOT implementing federated learning. IEEE-CIS has no natural institution
boundary, so any "banks" would be a partition we invented and then graded
ourselves. That is the circularity this project has repeatedly refused.

What we can honestly measure is the premise itself: how much does a model gain
from seeing everyone's data instead of only its own segment? That number is the
business case for federation. If pooling buys little, the cryptographic overhead
is not worth paying.

Segments are `ProductCD` (W/C/H/R/S) -- a real column in the data, not a
partition we designed. Each segment plays the role of an institution that sees
only its own traffic.

This also puts a number on why Vulcan wins: it pools ~4 billion payments across
merchants. Here we can measure what pooling is worth at our scale.

STATUS: PRELIMINARY -- SINGLE SEED, NOT VERIFIED.
Three single-run findings in this project have already been retracted after
seed verification. This one is not exempt and is deliberately kept out of the
README's claims until it is checked across seeds. Result so far: pooling is
NEGATIVE, -Rs 1,037,220 (-2.41%), with segment fraud rates ranging 1.93%-13.35%.
That is consistent with the non-IID client heterogeneity problem that dominates
federated learning research, but consistency with theory is not verification.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import calibrate
import config
import data
import features
import model

MIN_SEGMENT_ROWS = 2_000


def fit(X, y, Xv, yv):
    return lgb.train(model.PARAMS, lgb.Dataset(X, label=y),
                     num_boost_round=model.NUM_ROUNDS,
                     valid_sets=[lgb.Dataset(Xv, label=yv)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def rupee_loss(p, y, amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    blocked = p >= c_fp / (c_fp + c_fn)
    return float((amount[(~blocked) & (y == 1)] + config.CHARGEBACK_FEE_INR).sum()
                 + (amount[blocked & (y == 0)] * config.MERCHANT_MARGIN_RATE
                    + config.CUSTOMER_LTV_INR).sum())


def main() -> None:
    print("loading ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    tr_df, ca_df, te_df = data.temporal_split(raw)
    cols = audit["online"]
    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te_df, cols, cats)

    seg_tr = tr_df["ProductCD"].to_numpy()
    seg_ca = ca_df["ProductCD"].to_numpy()
    seg_te = te_df["ProductCD"].to_numpy()

    print("pooled model (all segments) ...")
    pooled = fit(X_tr, y_tr, X_ca, y_ca)
    p_pooled = calibrate.fit_calibrator(pooled.predict(X_ca), y_ca)(
        pooled.predict(X_te))

    rows, isolated_pred = [], np.full(len(y_te), np.nan)
    for s in sorted(set(seg_tr)):
        m_tr, m_ca, m_te = seg_tr == s, seg_ca == s, seg_te == s
        if m_tr.sum() < MIN_SEGMENT_ROWS or m_te.sum() < 200 or y_te[m_te].sum() < 10:
            print(f"  segment {s}: too small, skipped "
                  f"({m_tr.sum():,} train rows)")
            continue
        print(f"  segment {s}: {m_tr.sum():,} train rows, "
              f"{y_tr[m_tr].mean():.2%} fraud")

        b = fit(X_tr[m_tr], y_tr[m_tr], X_ca[m_ca], y_ca[m_ca])
        p_iso = calibrate.fit_calibrator(b.predict(X_ca[m_ca]), y_ca[m_ca])(
            b.predict(X_te[m_te]))
        isolated_pred[m_te] = p_iso

        rows.append({
            "segment": str(s),
            "train_rows": int(m_tr.sum()),
            "test_rows": int(m_te.sum()),
            "fraud_rate": float(y_te[m_te].mean()),
            "isolated_auc": float(roc_auc_score(y_te[m_te], p_iso)),
            "pooled_auc": float(roc_auc_score(y_te[m_te], p_pooled[m_te])),
            "isolated_pr": float(average_precision_score(y_te[m_te], p_iso)),
            "pooled_pr": float(average_precision_score(y_te[m_te], p_pooled[m_te])),
            "isolated_loss": rupee_loss(p_iso, y_te[m_te], amt_te[m_te]),
            "pooled_loss": rupee_loss(p_pooled[m_te], y_te[m_te], amt_te[m_te]),
        })

    w = 96
    print("\n" + "=" * w)
    print("WHAT IS POOLING WORTH?  each segment alone vs one model over all data")
    print("=" * w)
    print(f"{'segment':<10}{'train rows':>12}{'fraud':>8}{'AUC alone':>12}"
          f"{'AUC pooled':>12}{'dAUC':>9}{'d rupee loss':>16}")
    print("-" * w)
    for r in rows:
        d_auc = r["pooled_auc"] - r["isolated_auc"]
        d_loss = r["isolated_loss"] - r["pooled_loss"]
        print(f"{r['segment']:<10}{r['train_rows']:>12,}{r['fraud_rate']:>8.2%}"
              f"{r['isolated_auc']:>12.4f}{r['pooled_auc']:>12.4f}"
              f"{d_auc:>+9.4f}{d_loss:>+16,.0f}")
    print("-" * w)

    covered = ~np.isnan(isolated_pred)
    tot_iso = rupee_loss(isolated_pred[covered], y_te[covered], amt_te[covered])
    tot_pool = rupee_loss(p_pooled[covered], y_te[covered], amt_te[covered])
    gain = tot_iso - tot_pool
    print(f"across {int(covered.sum()):,} covered test transactions:")
    print(f"  every segment isolated   Rs {tot_iso:>14,.0f}")
    print(f"  one pooled model         Rs {tot_pool:>14,.0f}")
    print(f"  value of pooling         Rs {gain:>+14,.0f}  "
          f"({gain / tot_iso:+.2%})")
    print()
    helped = sum(1 for r in rows if r["pooled_auc"] > r["isolated_auc"])
    print(f"pooling improved AUC for {helped}/{len(rows)} segments")
    print()
    print("Read this as the business case for federated learning: it is what an")
    print("institution gains from everyone's data rather than only its own. If")
    print("the gain is small, federation is not worth its cryptographic cost.")
    print("It is also why Vulcan wins -- it pools ~4 billion payments; the")
    print("largest segment here has a few hundred thousand.")
    print("=" * w)

    (config.ARTIFACTS / "pooling_value.json").write_text(
        json.dumps({"segments": rows, "isolated_total": tot_iso,
                    "pooled_total": tot_pool, "pooling_gain": gain,
                    "segments_improved": helped}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'pooling_value.json'}")


if __name__ == "__main__":
    main()
