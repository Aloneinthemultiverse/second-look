"""Tune the GraphSAGE arm properly, then re-test the finding.

gnn.py reported PR-AUC 0.2831 against the ensemble's 0.5220 and concluded the
GNN loses. That conclusion was reached at 60 epochs with the calibration PR-AUC
still climbing, K=5 neighbours, and a single learning rate. An undertrained
model losing is not evidence about the method.

This applies the configuration the literature actually uses:

  EPOCHS 200 with patience    Ethereum-HGNN (arXiv:2203.12363) and the open-set
                              GAD work (arXiv:2311.06835) both train 100-200.
  K = 25 fan-out              GraphFC (arXiv:2305.11377) and the heterophily
                              study (arXiv:2312.06441) use T1=25, T2=10.
  LR searched                 {0.05, 0.01, 0.005, 0.001}, the grid in the
                              node-classification study (arXiv:2206.09144).

And it tests the substantive hypothesis, not just knobs. Fraud graphs are
HETEROPHILOUS -- fraudsters deliberately transact alongside normal traffic to
camouflage. Mean aggregation is a low-pass filter: it smooths each node toward
its neighbourhood average, which under heterophily erases exactly the signal we
want. Max aggregation and a self-residual both preserve high-frequency
structure. If heterophily is the binding constraint, those should help and
tuning alone should not.

Every choice is made on the CALIBRATION slice. Test is touched once, at the end,
for the winner. Same features, same temporal split, same Platt calibration, same
per-instance tau* as every other arm.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import calibrate
import config
import data
import features
import gnn

EPOCHS = 200
PATIENCE = 30
EVAL_EVERY = 5
LR_GRID = [0.05, 0.01, 0.005, 0.001]
ARCHS = [("mean", False), ("mean", True), ("max", True)]
K_FINAL = 25


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden, aggr="mean", residual=False):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden, aggr=aggr)
        self.c2 = SAGEConv(hidden, hidden, aggr=aggr)
        self.out = torch.nn.Linear(hidden, 1)
        self.residual = residual

    def forward(self, x, edge_index):
        h = F.relu(self.c1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h2 = F.relu(self.c2(h, edge_index))
        h = h2 + h if self.residual else h2
        return self.out(h).squeeze(-1)


def train(g, y, i_tr, i_ca, lr, aggr, residual, tag):
    """Train to convergence. Model selection on the calibration slice only."""
    torch.manual_seed(config.SEED)
    net = SAGE(g.x.shape[1], gnn.HIDDEN, aggr, residual)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    pos_w = torch.tensor([(y[:i_tr] == 0).sum() / max((y[:i_tr] == 1).sum(), 1)])

    best, best_state, best_ep, stale = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        net.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            net(g.x, g.edge_index)[:i_tr], g.y[:i_tr], pos_weight=pos_w)
        loss.backward()
        opt.step()

        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            net.eval()
            with torch.no_grad():
                sc = torch.sigmoid(net(g.x, g.edge_index)).numpy()
            ap = average_precision_score(y[i_tr:i_ca], sc[i_tr:i_ca])
            if ap > best:
                best, best_ep, stale = ap, ep, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                stale += EVAL_EVERY
            if ep % 25 == 0:
                print(f"      ep {ep:>3}  loss {loss.item():.4f}  "
                      f"calib PR-AUC {ap:.4f}  best {best:.4f} @{best_ep}",
                      flush=True)
            if stale >= PATIENCE:
                print(f"      early stop at {ep} (no gain for {stale} epochs)",
                      flush=True)
                break

    print(f"    {tag}: calib PR-AUC {best:.4f} @ epoch {best_ep}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    return net, best, best_ep


def scores(net, g):
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(g.x, g.edge_index)).numpy()


def build_graph(df, Xn, y, k):
    gnn.K_NEIGHBOURS = k
    t0 = time.time()
    ei = gnn.build_edges(df)
    print(f"   {ei.shape[1]:,} edges in {time.time()-t0:.0f}s "
          f"({ei.shape[1]/len(df):.1f} per node)", flush=True)
    assert (ei[0] < ei[1]).all(), "an edge points forward in time"
    return Data(x=torch.tensor(Xn), y=torch.tensor(y),
                edge_index=torch.tensor(ei)), int(ei.shape[1])


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

    log = {"stage1_lr": [], "stage2_arch": [], "epochs": EPOCHS,
           "patience": PATIENCE, "lr_grid": LR_GRID}

    # ---- stage 1: learning rate, on the cheap K=5 graph -------------------
    print("\nbuilding K=5 graph for the search ...", flush=True)
    g5, e5 = build_graph(df, Xn, y, 5)

    print(f"\nSTAGE 1  learning rate ({EPOCHS} epochs, patience {PATIENCE}, "
          f"mean aggregation)", flush=True)
    for lr in LR_GRID:
        print(f"  lr {lr} ...", flush=True)
        _, ap, ep = train(g5, y, i_tr, i_ca, lr, "mean", False, f"lr={lr}")
        log["stage1_lr"].append({"lr": lr, "calib_pr_auc": ap, "best_epoch": ep})
    best_lr = max(log["stage1_lr"], key=lambda r: r["calib_pr_auc"])["lr"]
    print(f"  -> best lr {best_lr}", flush=True)

    # ---- stage 2: architecture, the heterophily test ----------------------
    print(f"\nSTAGE 2  aggregation and residual (lr {best_lr})", flush=True)
    print("  mean aggregation is a low-pass filter; under heterophily it should")
    print("  lose to max aggregation and to a self-residual.", flush=True)
    for aggr, res in ARCHS:
        tag = f"{aggr}{'+res' if res else ''}"
        print(f"  {tag} ...", flush=True)
        _, ap, ep = train(g5, y, i_tr, i_ca, best_lr, aggr, res, tag)
        log["stage2_arch"].append({"aggr": aggr, "residual": res,
                                   "calib_pr_auc": ap, "best_epoch": ep})
    b = max(log["stage2_arch"], key=lambda r: r["calib_pr_auc"])
    print(f"  -> best arch {b['aggr']}{'+res' if b['residual'] else ''}",
          flush=True)

    # ---- stage 3: winner at literature fan-out, measured on TEST ----------
    print(f"\nSTAGE 3  winner at K={K_FINAL} (literature fan-out), then TEST",
          flush=True)
    g25, e25 = build_graph(df, Xn, y, K_FINAL)
    net, ap_ca, ep = train(g25, y, i_tr, i_ca, best_lr, b["aggr"], b["residual"],
                           f"K={K_FINAL}")
    sc = scores(net, g25)
    p_te = calibrate.fit_calibrator(sc[i_tr:i_ca], y[i_tr:i_ca])(sc[i_ca:])
    tuned = gnn.evaluate(p_te, y_te, amt_te)

    # also score the winner on the K=5 graph, to separate fan-out from arch
    net5, _, _ = train(g5, y, i_tr, i_ca, best_lr, b["aggr"], b["residual"],
                       "K=5")
    s5 = scores(net5, g5)
    p5 = calibrate.fit_calibrator(s5[i_tr:i_ca], y[i_tr:i_ca])(s5[i_ca:])
    tuned5 = gnn.evaluate(p5, y_te, amt_te)

    base = json.loads((config.ARTIFACTS / "gnn.json").read_text())["graphsage"]
    canon = json.loads((config.ARTIFACTS / "canonical.json").read_text())

    w = 92
    print("\n" + "=" * w)
    print("DOES A PROPERLY TUNED GRAPHSAGE CHANGE THE FINDING?")
    print("=" * w)
    print(f"{'arm':<40}{'PR-AUC':>10}{'ROC-AUC':>10}{'caught':>14}{'recall':>9}")
    print("-" * w)
    rows = [("GraphSAGE untuned (60ep, K=5, mean)", base),
            ("GraphSAGE tuned  (K=5)", tuned5),
            (f"GraphSAGE tuned  (K={K_FINAL})", tuned)]
    for label, r in rows:
        print(f"{label:<40}{r['pr_auc']:>10.4f}{r['roc_auc']:>10.4f}"
              f"{r['frauds_caught']:>8,}/{r['frauds_total']:<5,}"
              f"{r['recall']:>9.3f}")
    print(f"{'ENSEMBLE (canonical, the shipped model)':<40}"
          f"{canon['pr_auc']:>10.4f}{canon['roc_auc']:>10.4f}"
          f"{canon['frauds_caught']:>8,}/{canon['frauds_total']:<5,}"
          f"{canon['recall_count']:>9.3f}")
    print("-" * w)
    gain = tuned["pr_auc"] - base["pr_auc"]
    gap = canon["pr_auc"] - tuned["pr_auc"]
    print(f"tuning recovered PR-AUC {gain:+.4f}")
    print(f"remaining gap to the ensemble {gap:+.4f}")
    print("verdict: " + ("GNN NOW WINS -- rewrite the finding" if gap < 0
                         else "GNN still loses, now on a tuned model"))
    print("=" * w)

    log.update({"best_lr": best_lr, "best_arch": b, "k_final": K_FINAL,
                "edges_k5": e5, f"edges_k{K_FINAL}": e25,
                "tuned_k5": tuned5, "tuned_final": tuned,
                "untuned_reference": base, "ensemble_reference": canon,
                "pr_auc_gain_from_tuning": gain, "gap_to_ensemble": gap})
    (config.ARTIFACTS / "gnn_tuned.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'gnn_tuned.json'}")


if __name__ == "__main__":
    main()
