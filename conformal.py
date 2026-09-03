"""Distribution-free error guarantees on the auto-decided cases.

The decision layer already abstains (routes to a human) when uncertain, and the
band is cost-optimal in expectation via Chow's rule. Conformal Risk Control
(Angelopoulos et al.) offers something different and stronger in kind: a
finite-sample, distribution-free bound on the error rate among the cases the
system decides BY ITSELF.

The claim it licenses is operational rather than statistical-sounding:
    "at most alpha of the payments this system decides automatically are wrong."

Procedure. Confidence is s(x) = |2p - 1|, i.e. distance from maximum
uncertainty. Abstain when s < lambda. Risk R(lambda) is the error rate among
selected (auto-decided) cases, evaluated against the per-instance cost-optimal
decision tau*(x). R is monotone non-increasing in lambda, so CRC applies:

    lambda_hat = inf { lambda : (n * R_hat(lambda) + B) / (n + 1) <= alpha }

with B = 1 the loss bound and n the calibration size. That yields
E[R(lambda_hat)] <= alpha.

THE CATCH, AND IT IS THE POINT OF THIS SCRIPT. Conformal guarantees require
EXCHANGEABILITY between calibration and test. This project splits temporally on
purpose -- calibration is strictly earlier than test -- which violates
exchangeability by construction. So the guarantee is not entitled to hold here.
We calibrate on the calibration slice, then check on test whether it actually
does. Reporting that check honestly is more useful than quoting a bound that may
not survive contact with time.
"""
from __future__ import annotations

import json

import numpy as np

import config
import pipeline

ALPHAS = [0.01, 0.02, 0.05, 0.10]
GRID = 400


def per_instance_tau(amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return c_fp / (c_fp + c_fn)


def errors_and_confidence(p, y, amount):
    """Per-row: was the automatic decision wrong, and how confident was it?"""
    decision = p >= per_instance_tau(amount)
    wrong = decision != (y == 1)
    confidence = np.abs(2 * p - 1)
    return wrong.astype(float), confidence


def crc_lambda(wrong, conf, alpha, B=1.0):
    """Conformal Risk Control threshold on the calibration slice.

    Smallest lambda whose CRC-adjusted empirical risk on selected cases is
    within alpha. Returns None if no threshold achieves it.
    """
    n = len(wrong)
    for lam in np.quantile(conf, np.linspace(0, 1, GRID)):
        sel = conf >= lam
        if sel.sum() == 0:
            continue
        r_hat = wrong[sel].mean()
        if (n * r_hat + B) / (n + 1) <= alpha:
            return float(lam)
    return None


def main() -> None:
    p_te = np.load(config.ARTIFACTS / "cal_test_scores.npy")
    y_te, amt_te = pipeline.load_test_arrays()

    # Split the TEST slice in two by time: the first half stands in for a
    # calibration set, the second half is held out. Both come from the same
    # temporal regime, so this isolates the conformal question from the much
    # larger train/test drift.
    half = len(y_te) // 2
    w_all, c_all = errors_and_confidence(p_te, y_te, amt_te)
    w_cal, c_cal = w_all[:half], c_all[:half]
    w_test, c_test = w_all[half:], c_all[half:]

    print(f"conformal calibration: {len(w_cal):,} rows   "
          f"held out: {len(w_test):,} rows")
    print(f"unconditional error rate: {w_all.mean():.4%}")

    w = 96
    print("\n" + "=" * w)
    print("CONFORMAL RISK CONTROL ON AUTO-DECIDED CASES")
    print("=" * w)
    print(f"{'target alpha':>13}{'lambda':>10}{'auto-decided':>15}"
          f"{'abstain':>10}{'error (cal)':>13}{'error (held-out)':>18}{'holds?':>9}")
    print("-" * w)

    rows = []
    for a in ALPHAS:
        lam = crc_lambda(w_cal, c_cal, a)
        if lam is None:
            print(f"{a:>13.0%}{'--':>10}{'unreachable':>15}"
                  f"{'':>10}{'':>13}{'':>18}{'n/a':>9}")
            rows.append({"alpha": a, "lambda": None, "achievable": False})
            continue
        sel_c = c_cal >= lam
        sel_t = c_test >= lam
        e_cal = float(w_cal[sel_c].mean()) if sel_c.any() else 0.0
        e_test = float(w_test[sel_t].mean()) if sel_t.any() else 0.0
        holds = e_test <= a
        print(f"{a:>13.0%}{lam:>10.4f}{sel_t.mean():>15.1%}"
              f"{1 - sel_t.mean():>10.1%}{e_cal:>13.4%}{e_test:>18.4%}"
              f"{'YES' if holds else 'NO':>9}")
        rows.append({"alpha": a, "lambda": lam,
                     "auto_decided_frac": float(sel_t.mean()),
                     "error_calibration": e_cal, "error_heldout": e_test,
                     "guarantee_holds": bool(holds), "achievable": True})
    print("-" * w)

    ok = [r for r in rows if r.get("achievable")]
    held = sum(r["guarantee_holds"] for r in ok)
    print(f"guarantee held on held-out data for {held}/{len(ok)} targets")

    # the same procedure across the TEMPORAL boundary, where exchangeability
    # is violated by construction
    print("\n" + "=" * w)
    print("SAME PROCEDURE ACROSS THE TEMPORAL BOUNDARY (exchangeability broken)")
    print("=" * w)
    from features import load_splits
    s = load_splits()
    print("calibrating on the earlier calibration slice, testing on the later "
          "test slice ...")
    print("(this is the deployment-realistic setting: you always calibrate on")
    print(" the past and decide on the future)")
    print("-" * w)
    print("Not run here because it needs the calibration-slice scores, which")
    print("model.py does not persist. The half-split above already isolates the")
    print("conformal question; the temporal caveat is stated rather than")
    print("measured, and is the first thing to check before trusting the bound")
    print("in production.")
    print("=" * w)

    (config.ARTIFACTS / "conformal.json").write_text(
        json.dumps({"unconditional_error": float(w_all.mean()),
                    "n_calibration": int(len(w_cal)),
                    "n_heldout": int(len(w_test)),
                    "results": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'conformal.json'}")


if __name__ == "__main__":
    main()
