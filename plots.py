"""Figures. Completes the Day 2 deliverable (the reliability diagram) and
produces the visuals the pitch needs.

Reads existing artifacts where possible so nothing is retrained. The reliability
diagram is measured on the TEST slice -- calibration was fitted on the separate
calibration slice, so plotting it on test is the honest check.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

import config
import pipeline

FIGDIR = config.ARTIFACTS / "figures"
FIGDIR.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def _load(name):
    return json.loads((config.ARTIFACTS / name).read_text(encoding="utf-8"))


def fig_reliability():
    """Day 2 deliverable: does the calibrator actually calibrate, on held-out data?"""
    cal = _load("calibration_compare.json")
    p = np.load(config.ARTIFACTS / "cal_test_scores.npy")
    y, _ = pipeline.load_test_arrays()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot([0, max(mean_p)], [0, max(mean_p)], "k--", lw=1, label="perfect")
    ax.plot(mean_p, frac, "o-", color="#1f77b4", label=f"Platt (ECE {cal['platt']['ece']:.4f})")
    ax.set_xlabel("mean predicted P(fraud)")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("Reliability on held-out test slice")
    ax.legend(frameon=False)

    names = ["none", "platt", "isotonic"]
    ece = [cal[n]["ece"] for n in names]
    pra = [cal[n]["pr_auc"] for n in names]
    x = np.arange(3)
    ax2.bar(x - 0.2, ece, 0.4, label="ECE (lower better)", color="#ff7f0e")
    ax2b = ax2.twinx()
    ax2b.bar(x + 0.2, pra, 0.4, label="PR-AUC (higher better)", color="#2ca02c")
    ax2b.grid(False)
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylabel("ECE"); ax2b.set_ylabel("PR-AUC")
    ax2.set_title("Isotonic calibrates worse AND loses ranking")
    for i, n in enumerate(names):
        ax2.annotate(f"{cal[n]['n_unique_scores']:,}\nlevels", (i, 0),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=7, color="#444")
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_calibration.png", bbox_inches="tight")
    plt.close(fig)


def fig_alert_budget():
    d = _load("alert_budget.json")
    ks = [r["k"] for r in d["budgets"]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ks, [r["card_precision_at_k"] for r in d["budgets"]], "o-",
            label="card Precision@k")
    ax.plot(ks, [r["txn_precision_at_k"] for r in d["budgets"]], "s--",
            label="transaction Precision@k")
    ax.plot(ks, [r["card_recall"] for r in d["budgets"]], "^-",
            label="card recall (coverage)", color="#888")
    ax.axhline(d["baseline_precision"], color="r", ls=":", lw=1,
               label=f"random ({d['baseline_precision']:.3f})")
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("analyst budget k (alerts per day)")
    ax.set_ylabel("precision / recall")
    ax.set_title("At realistic capacity the model is excellent at the top")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_alert_budget.png", bbox_inches="tight")
    plt.close(fig)


def fig_capacity_accuracy():
    """The most striking result: imperfect analysts cap useful capacity."""
    d = _load("chow_capacity.json")
    ks = [r["k"] for r in d["budgets"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for acc, style in (("1.0", "o-"), ("0.9", "s-"), ("0.8", "^-")):
        ax.plot(ks, [r["loss_by_accuracy"][acc] / 1e6 for r in d["budgets"]],
                style, label=f"analyst {float(acc):.0%} accurate")
    ax.axhline(d["do_nothing"] / 1e6, color="r", ls="--", lw=1,
               label="no fraud system at all")
    ax.set_xlabel("review capacity k (cases per day)")
    ax.set_ylabel("realised loss (Rs millions)")
    ax.set_title("At 80% analyst accuracy, reviewing more destroys value")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_capacity_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def fig_cost_vs_f1():
    d = _load("cost_sensitive.json")
    keys = [k for k in d["results"] if not k.startswith("A0")]
    losses = [d["results"][k]["rupee_loss"] / 1e6 for k in keys]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.barh(range(len(keys)), losses, color="#4c78a8")
    ax.axvline(d["do_nothing"] / 1e6, color="r", ls="--", lw=1, label="do nothing")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.split(" ", 1)[1] for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("realised loss (Rs millions)")
    ax.set_title("Decision strategies, thresholds chosen off-test")
    for b, l in zip(bars, losses):
        ax.text(b.get_width() - 1.5, b.get_y() + b.get_height() / 2,
                f"{l:.1f}M", va="center", ha="right", color="w", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_strategies.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for fn in (fig_reliability, fig_alert_budget, fig_capacity_accuracy,
               fig_cost_vs_f1):
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except FileNotFoundError as e:
            print(f"  skip {fn.__name__}: missing artifact ({e.filename})")
    print(f"\nfigures in {FIGDIR}")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}  {f.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
