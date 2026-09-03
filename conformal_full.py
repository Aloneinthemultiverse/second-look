"""The full temporal conformal test, and the time-series fix.

conformal.py left a gap and said so: it split the test slice in half rather than
calibrating on the earlier calibration slice and deciding on the later test
slice. That second setting is the deployment-realistic one -- in production you
always calibrate on the past and decide on the future. This runs it.

Two methods:

  STATIC CRC   Conformal Risk Control (Angelopoulos et al.). Pick one lambda on
               the calibration slice, freeze it, use it forever. Requires
               exchangeability between calibration and test, which a temporal
               split violates by construction.

  ACI          Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021).
               Instead of freezing lambda, treat distribution shift as a single
               parameter re-estimated online:

                   alpha_{t+1} = alpha_t + gamma * (target - err_t)

               where err_t is 1 if the auto-decision at step t was wrong. When
               recent errors exceed target the effective alpha shrinks, lambda
               rises, and the system abstains more. ACI achieves the target
               coverage frequency over long horizons WITHOUT assuming
               exchangeability -- which is exactly the assumption payments break.

The question: does static CRC fail across the real temporal boundary, and does
ACI recover the guarantee?
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

import calibrate
import config
import features
import model

ALPHAS = [0.01, 0.02, 0.05]
GAMMA = 0.01          # ACI learning rate (Gibbs & Candes use small constants)
GRID = 500


def per_instance_tau(amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return c_fp / (c_fp + c_fn)


def wrong_and_conf(p, y, amount):
    decision = p >= per_instance_tau(amount)
    return (decision != (y == 1)).astype(float), np.abs(2 * p - 1)


def static_crc(w_cal, c_cal, alpha, B=1.0):
    n = len(w_cal)
    for lam in np.quantile(c_cal, np.linspace(0, 1, GRID)):
        sel = c_cal >= lam
        if sel.sum() == 0:
            continue
        if (n * w_cal[sel].mean() + B) / (n + 1) <= alpha:
            return float(lam)
    return None


def run_aci(w, c, alpha, gamma=GAMMA, warmup=2000):
    """Adaptive Conformal Inference over the test stream, in time order.

    Maintains an effective alpha_t. lambda_t is the (1 - alpha_t) quantile of
    confidences seen so far, so a shrinking alpha_t raises the bar and abstains
    more. Updated after each auto-decision using its realised error.
    """
    a_t = alpha
    decided, errors, alphas = [], [], []
    hist = list(c[:warmup])
    for i in range(warmup, len(c)):
        q = np.clip(1.0 - a_t, 0.0, 1.0)
        lam = float(np.quantile(hist, q)) if hist else 0.0
        auto = c[i] >= lam
        if auto:
            err = w[i]
            decided.append(i)
            errors.append(err)
        else:
            err = 0.0           # abstained: routed to a human, not an error
        a_t = float(np.clip(a_t + gamma * (alpha - err), 1e-4, 0.999))
        alphas.append(a_t)
        hist.append(c[i])
        if len(hist) > 20000:
            hist = hist[-20000:]
    n = len(c) - warmup
    return {"auto_decided_frac": len(decided) / max(n, 1),
            "error_rate": float(np.mean(errors)) if errors else 0.0,
            "final_alpha": a_t,
            "alpha_min": float(min(alphas)), "alpha_max": float(max(alphas))}


def main() -> None:
    print("training and persisting calibration-slice scores ...")
    s = features.load_splits()
    X_tr, y_tr, _ = s["train"]
    X_ca, y_ca, amt_ca = s["calib"]
    X_te, y_te, amt_te = s["test"]

    b = lgb.train(model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                  num_boost_round=model.NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    f = calibrate.fit_calibrator(b.predict(X_ca), y_ca)
    p_ca, p_te = f(b.predict(X_ca)), f(b.predict(X_te))

    w_ca, c_ca = wrong_and_conf(p_ca, y_ca, amt_ca)
    w_te, c_te = wrong_and_conf(p_te, y_te, amt_te)
    print(f"calibration slice {len(w_ca):,} rows, error {w_ca.mean():.4%}")
    print(f"test slice        {len(w_te):,} rows, error {w_te.mean():.4%}")
    print("(these are strictly separated in time -- the deployment setting)")

    w = 100
    print("\n" + "=" * w)
    print("STATIC CRC ACROSS THE REAL TEMPORAL BOUNDARY")
    print("=" * w)
    print(f"{'target':>9}{'lambda':>10}{'auto-decided':>15}{'error (cal)':>14}"
          f"{'error (TEST)':>15}{'holds?':>9}{'overshoot':>13}")
    print("-" * w)
    static_rows = []
    for a in ALPHAS:
        lam = static_crc(w_ca, c_ca, a)
        if lam is None:
            print(f"{a:>9.0%}{'--':>10}{'unreachable':>15}")
            static_rows.append({"alpha": a, "achievable": False})
            continue
        sel = c_te >= lam
        e_cal = float(w_ca[c_ca >= lam].mean())
        e_te = float(w_te[sel].mean()) if sel.any() else 0.0
        holds = e_te <= a
        print(f"{a:>9.0%}{lam:>10.4f}{sel.mean():>15.1%}{e_cal:>14.4%}"
              f"{e_te:>15.4%}{'YES' if holds else 'NO':>9}"
              f"{(e_te - a) / a:>+13.0%}")
        static_rows.append({"alpha": a, "lambda": lam, "achievable": True,
                            "auto_decided": float(sel.mean()),
                            "error_cal": e_cal, "error_test": e_te,
                            "holds": bool(holds)})
    print("-" * w)

    print("\n" + "=" * w)
    print("ADAPTIVE CONFORMAL INFERENCE (Gibbs & Candes 2021) ON THE TEST STREAM")
    print("=" * w)
    print(f"{'target':>9}{'auto-decided':>15}{'error rate':>14}{'holds?':>9}"
          f"{'alpha range':>26}")
    print("-" * w)
    aci_rows = []
    for a in ALPHAS:
        r = run_aci(w_te, c_te, a)
        holds = r["error_rate"] <= a
        print(f"{a:>9.0%}{r['auto_decided_frac']:>15.1%}{r['error_rate']:>14.4%}"
              f"{'YES' if holds else 'NO':>9}"
              f"{r['alpha_min']:>13.4f}{r['alpha_max']:>13.4f}")
        r.update({"alpha": a, "holds": bool(holds)})
        aci_rows.append(r)
    print("-" * w)

    s_ok = sum(r.get("holds", False) for r in static_rows)
    a_ok = sum(r["holds"] for r in aci_rows)
    print(f"static CRC held {s_ok}/{len(ALPHAS)} targets across the temporal boundary")
    print(f"ACI       held {a_ok}/{len(ALPHAS)} targets on the same data")
    print()
    if a_ok > s_ok:
        print("ACI recovers guarantees that static conformal loses to drift, which")
        print("is what Gibbs & Candes claim: coverage without exchangeability.")
    elif a_ok == s_ok == len(ALPHAS):
        print("Both held. The temporal boundary was milder than expected here.")
    else:
        print("ACI did not recover the guarantee either. Reported as-is.")
    print("=" * w)

    (config.ARTIFACTS / "conformal_full.json").write_text(
        json.dumps({"static": static_rows, "aci": aci_rows, "gamma": GAMMA},
                   indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'conformal_full.json'}")


if __name__ == "__main__":
    main()
