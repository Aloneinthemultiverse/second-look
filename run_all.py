"""Reproduce every result in this repository, in order.

    python run_all.py            # everything
    python run_all.py --quick    # skip the stages that retrain several models

Each stage writes a JSON artifact to artifacts/, so results can be inspected
without rerunning. Stages are independent apart from the noted dependency on
model.py, which saves the calibrated test scores that later stages reuse.

Determinism: every model is trained with seed=42 (config.SEED). The only
randomness elsewhere is the analyst-accuracy simulation in chow_capacity.py,
which uses a seeded generator.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = [
    ("check_data.py", "leakage screen -- run this first on any new data", False),
    ("data.py", "temporal split + feature audit", False),
    ("canonical.py", "THE HEADLINE NUMBER (ensemble, per-instance threshold)", False),
    ("model.py", "detector + calibration + threshold sweep (writes scores)", False),
    ("audit.py", "segmented calibration, degeneracy, error anatomy, drift", False),
    ("sensitivity.py", "threshold sensitivity to cost assumptions", False),
    ("alert_budget.py", "Precision@k under analyst capacity (Dal Pozzolo)", False),
    ("chow_band.py", "review band derived from costs (Chow)", False),
    ("chow_capacity.py", "Chow under a real analyst budget", False),
    ("playbook.py", "5-lookup playbook + entity graph + info filter", False),
    ("investigator.py", "TreeSHAP -> analyst briefs", False),
    ("review_band_metrics.py", "does the evidence add signal? (honest answer: no)", True),
    ("robustness.py", "defence-only adversarial audits", False),
    ("fallback.py", "counting-service outage fallback", False),
    ("loss_types.py", "one decision layer, four loss types", False),
    ("three_way.py", "allow/review/block on the SHIPPED ensemble", True),
    ("conformal.py", "adaptive conformal control of the block rate", False),
    ("conformal_full.py", "conformal at 1%, 2%, 5% targets", True),
    ("plots.py", "figures", False),
    ("fusion.py", "LightGBM + XGBoost + RandomForest fusion", True),
    ("verify_fusion.py", "is the fusion gain real? (3 seeds)", True),
    ("anomaly_layer.py", "IsolationForest blended into the fusion", True),
    ("verify_anomaly.py", "is the anomaly gain real? (3 seeds -- no)", True),
    ("high_value.py", "the expensive-fraud weakness", True),
    ("resampling.py", "SMOTEENN (negative: 322 fewer frauds caught)", True),
    ("pooling_value.py", "what is pooling data worth? (federation premise)", True),
    ("federated.py", "naive federated averaging (negative)", True),
    ("federated_ensemble.py", "calibrated federated bagging (also negative)", True),
    ("personalised_fl.py", "personalised FL with gated routing", True),
    ("gnn.py", "GraphSAGE on a time-respecting graph", True),
    ("gnn_tune.py", "GNN tuning: LR sweep + heterophily test (HOURS)", True),
    ("gnn_stage3.py", "GNN fan-out + test metrics (needs gnn_tune first)", True),
    ("hybrid.py", "ensemble + GNN blend", True),
    ("verify_hybrid.py", "is the hybrid gain real? (3 seeds)", True),
    ("calibration_compare.py", "none vs Platt vs isotonic", True),
    ("cost_sensitive.py", "global vs per-instance vs cost-weighted", True),
    ("restriction_cost.py", "cost of the inference-latency constraint", True),
    ("verify.py", "seeds, split points, inference latency", True),
    ("second_dataset.py", "does the method transfer? (ULB)", True),
    ("sensitivity_cross.py", "cross-dataset sensitivity", True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip stages that retrain multiple models")
    args = ap.parse_args()

    print("running tests first ...")
    if subprocess.run([sys.executable, "-m", "pytest", "test_core.py", "-q"]).returncode:
        print("tests failed -- stopping")
        return 1

    failed = []
    for script, desc, slow in STAGES:
        if args.quick and slow:
            print(f"\n--- SKIP {script}  ({desc})")
            continue
        print(f"\n{'=' * 70}\n>>> {script}  --  {desc}\n{'=' * 70}")
        t0 = time.perf_counter()
        rc = subprocess.run([sys.executable, script]).returncode
        print(f"[{script} finished in {time.perf_counter() - t0:.0f}s, rc={rc}]")
        if rc:
            failed.append(script)

    print("\n" + "=" * 70)
    print("FAILED: " + ", ".join(failed) if failed else "all stages completed")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
