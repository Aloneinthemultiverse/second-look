"""Stage 3 of the GraphSAGE tuning study, resumable on its own.

gnn_tune.py completed stages 1 and 2 and was killed partway through stage 3.
Rather than repeat two hours of search whose answers are already known, this
takes those answers as inputs and finishes the job.

WHAT STAGES 1 AND 2 ESTABLISHED (from artifacts/gnn_tune.log):

  learning rate    0.05 -> 0.1235   diverged after epoch 30
                   0.01 -> 0.3903   still improving when the 200-epoch cap hit
                   0.005 -> 0.3945  converged at epoch 185   <- winner
                   0.001 -> 0.3467  too slow, also capped

  architecture     mean       -> 0.3945   <- winner
                   mean+res   -> 0.3866
                   max+res    -> 0.3628

The architecture result refutes the heterophily hypothesis this study was built
to test. Fraud-graph papers argue mean aggregation is a low-pass filter that
erases the signal under heterophily, and that max aggregation or a self-residual
should recover it. On this graph both made it worse. Plain mean wins.

WHAT REMAINS: does the literature's K=25 fan-out buy anything over K=5, and what
does the tuned model actually do on the test set? The partial stage-3 run
reached epoch 75 at calib PR-AUC 0.3356, against K=5's 0.3350 at the same
epoch -- so the early evidence is that fan-out changes nothing. This settles it.

Test is touched once per arm, for scoring only. Nothing is selected on it.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

import calibrate
import config
import data
import features
import gnn
from gnn_tune import SAGE, build_graph, scores, train

BEST_LR = 0.005
BEST_AGGR = "mean"
BEST_RESIDUAL = False

STAGE1 = [{"lr": 0.05, "calib_pr_auc": 0.1235, "best_epoch": 30},
          {"lr": 0.01, "calib_pr_auc": 0.3903, "best_epoch": 200},
          {"lr": 0.005, "calib_pr_auc": 0.3945, "best_epoch": 185},
          {"lr": 0.001, "calib_pr_auc": 0.3467, "best_epoch": 200}]
STAGE2 = [{"aggr": "mean", "residual": False, "calib_pr_auc": 0.3945,
           "best_epoch": 185},
          {"aggr": "mean", "residual": True, "calib_pr_auc": 0.3866,
           "best_epoch": 190},
          {"aggr": "max", "residual": True, "calib_pr_auc": 0.3628,
           "best_epoch": 200}]


def main() -> None:
    print("loading ...", flush=True)
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    cols = audit["online"]
    df = raw.sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)
    i_tr = int(n * config.TRAIN_FRAC)
    i_ca = int(n * (config.TRAIN_FRAC + config.CALIB_FRAC))

    X, _, amt, _ = features.build(df, cols)
    Xn = X.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c)
    Xn = Xn.astype("float32").replace([np.inf, -np.inf], np.nan)
    mu, sd = Xn.iloc[:i_tr].mean(), Xn.iloc[:i_tr].std().replace(0, 1)
    Xn = ((Xn - mu) / sd).fillna(0.0).to_numpy()
    y = df["isFraud"].to_numpy().astype("float32")
    y_te, amt_te = y[i_ca:].astype(int), amt[i_ca:]

    def arm(k, tag):
        g, e = build_graph(df, Xn, y, k)
        t0 = time.time()
        net, ap_ca, ep = train(g, y, i_tr, i_ca, BEST_LR, BEST_AGGR,
                               BEST_RESIDUAL, tag)
        sc = scores(net, g)
        p = calibrate.fit_calibrator(sc[i_tr:i_ca], y[i_tr:i_ca])(sc[i_ca:])
        r = gnn.evaluate(p, y_te, amt_te)
        r.update({"edges": e, "calib_pr_auc": ap_ca, "best_epoch": ep,
                  "seconds": round(time.time() - t0)})
        return r

    print("\nK=5 arm (tuned, same fan-out as the original run) ...", flush=True)
    k5 = arm(5, "K=5")
    print(f"\nK=25 arm (literature fan-out) ...", flush=True)
    k25 = arm(25, "K=25")

    base = json.loads((config.ARTIFACTS / "gnn.json").read_text())["graphsage"]
    canon = json.loads((config.ARTIFACTS / "canonical.json").read_text())
    canon = canon["runs"][0] if "runs" in canon else canon

    w = 104
    print("\n" + "=" * w)
    print("DOES A PROPERLY TUNED GRAPHSAGE CHANGE THE FINDING?")
    print("=" * w)
    print(f"{'arm':<38}{'PR-AUC':>9}{'ROC-AUC':>10}{'caught':>16}"
          f"{'recall':>9}{'prec':>8}{'FPR':>8}")
    print("-" * w)
    for label, r, rec in (
            ("GraphSAGE untuned (60ep, K=5)", base, base["recall"]),
            ("GraphSAGE tuned (K=5)", k5, k5["recall"]),
            ("GraphSAGE tuned (K=25)", k25, k25["recall"]),
            ("ENSEMBLE (shipped)", canon, canon["recall_count"])):
        print(f"{label:<38}{r['pr_auc']:>9.4f}{r['roc_auc']:>10.4f}"
              f"{r['frauds_caught']:>9,}/{r['frauds_total']:<5,}"
              f"{rec:>9.1%}{r['precision']:>8.3f}{r['fpr']:>8.2%}")
    print("-" * w)

    gain = k25["pr_auc"] - base["pr_auc"]
    fan = k25["pr_auc"] - k5["pr_auc"]
    gap = canon["pr_auc"] - k25["pr_auc"]
    extra = canon["frauds_caught"] - max(k5["frauds_caught"],
                                         k25["frauds_caught"])
    print(f"tuning recovered PR-AUC          {gain:+.4f}  "
          f"({base['frauds_caught']:,} -> "
          f"{max(k5['frauds_caught'], k25['frauds_caught']):,} frauds)")
    print(f"K=25 fan-out over K=5            {fan:+.4f}  "
          f"({k25['edges']:,} edges vs {k5['edges']:,})")
    print(f"remaining gap to the ensemble    {gap:+.4f}  "
          f"({extra:,} more frauds caught by the ensemble)")
    print()
    print("Read together with stages 1 and 2: every gain came from training the")
    print("model long enough. Neither architectural prescription in the fraud-GNN")
    print("literature -- max aggregation, self-residual -- helped, and the 25/10")
    print("fan-out those papers specify" + (" did not help either." if fan <= 0.005
                                            else " did help."))
    print("The original conclusion "
          + ("STANDS, now on a converged model." if gap > 0
             else "DOES NOT STAND -- the GNN wins once trained."))
    print("=" * w)

    (config.ARTIFACTS / "gnn_tuned.json").write_text(json.dumps({
        "stage1_lr": STAGE1, "stage2_arch": STAGE2,
        "best_lr": BEST_LR, "best_aggr": BEST_AGGR,
        "best_residual": BEST_RESIDUAL, "epochs": gnn_tune_epochs(),
        "tuned_k5": k5, "tuned_k25": k25,
        "untuned_reference": base, "ensemble_reference": canon,
        "pr_auc_gain_from_tuning": gain, "fan_out_effect": fan,
        "gap_to_ensemble": gap, "extra_frauds_from_ensemble": int(extra),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'gnn_tuned.json'}")


def gnn_tune_epochs():
    import gnn_tune
    return gnn_tune.EPOCHS


if __name__ == "__main__":
    main()
