"""GraphSAGE on a time-respecting transaction graph.

PayPal and others use graph structure in production, so the question is worth
answering with a real experiment rather than an argument from the literature.

GRAPH CONSTRUCTION -- and the part most papers get wrong.
Nodes are transactions. Edges connect transactions that share an identifier:
card1, DeviceInfo, or P_emaildomain. Crucially, each transaction is linked ONLY
to EARLIER transactions sharing that identifier, capped at K per relation.

Why that matters: the usual approach builds one static graph over the whole
dataset, so a transaction's neighbours include ones that happened after it.
Under a temporal split that leaks the future through the graph structure even
when the features are clean, and the reported lift is partly an artefact. Our
edges are strictly backward in time, so nothing a node sees postdates it.

MODEL: 2-layer GraphSAGE (mean aggregation) -- message passing, which is the
mechanism the GNN literature is actually about. Deeper nets over-smooth on
sparse transaction graphs.

FAIRNESS: same 77 online features, same temporal split, same Platt calibration
on the same calibration slice, same per-instance cost threshold as every other
arm in this project. The only thing that changes is the model.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import calibrate
import config
import data
import features

K_NEIGHBOURS = 5          # per relation, per node -- keeps the graph bounded
RELATIONS = ["card1", "DeviceInfo", "P_emaildomain"]
HIDDEN = 64
EPOCHS = 60
LR = 0.01


def per_instance_tau(amount):
    c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = amount + config.CHARGEBACK_FEE_INR
    return c_fp / (c_fp + c_fn)


def evaluate(p, y, amount):
    blocked = p >= per_instance_tau(amount)
    tp, fp, fn = blocked & (y == 1), blocked & (y == 0), (~blocked) & (y == 1)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "frauds_caught": int(tp.sum()), "frauds_total": int((y == 1).sum()),
        "recall": float(tp.sum() / max((y == 1).sum(), 1)),
        "precision": float(tp.sum() / max(blocked.sum(), 1)),
        "fpr": float(fp.sum() / max((y == 0).sum(), 1)),
        "rupee_loss": float((amount[fn] + config.CHARGEBACK_FEE_INR).sum()
                            + (amount[fp] * config.MERCHANT_MARGIN_RATE
                               + config.CUSTOMER_LTV_INR).sum()),
    }


def build_edges(df: pd.DataFrame) -> np.ndarray:
    """Backward-in-time edges: each node links to its K most recent predecessors.

    df must already be sorted by TransactionDT, so positional index == time
    order and 'earlier' is simply a smaller index.
    """
    src, dst = [], []
    for col in RELATIONS:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy()
        last: dict = {}
        for i, v in enumerate(vals):
            if pd.isna(v):
                continue
            prev = last.get(v)
            if prev:
                for j in prev[-K_NEIGHBOURS:]:
                    src.append(j); dst.append(i)      # past -> present only
            last.setdefault(v, []).append(i)
            if len(last[v]) > K_NEIGHBOURS:
                last[v] = last[v][-K_NEIGHBOURS:]
        print(f"   {col}: {len(src):,} cumulative edges")
    return np.array([src, dst], dtype=np.int64)


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.c1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.c2(h, edge_index))
        return self.out(h).squeeze(-1)


def main() -> None:
    print("loading ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    cols = audit["online"]
    df = raw.sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)
    i_tr = int(n * config.TRAIN_FRAC)
    i_ca = int(n * (config.TRAIN_FRAC + config.CALIB_FRAC))

    # numeric matrix: categoricals -> codes, standardised on TRAIN only
    X, _, amt, _ = features.build(df, cols)
    Xn = X.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c)
    Xn = Xn.astype("float32").replace([np.inf, -np.inf], np.nan)
    mu, sd = Xn.iloc[:i_tr].mean(), Xn.iloc[:i_tr].std().replace(0, 1)
    Xn = ((Xn - mu) / sd).fillna(0.0)
    y = df["isFraud"].to_numpy().astype("float32")

    print(f"building time-respecting graph (K={K_NEIGHBOURS} per relation) ...")
    t0 = time.time()
    edge_index = build_edges(df)
    print(f"   {edge_index.shape[1]:,} edges in {time.time()-t0:.0f}s "
          f"({edge_index.shape[1]/n:.1f} per node)")
    assert (edge_index[0] < edge_index[1]).all(), "an edge points forward in time"

    g = Data(x=torch.tensor(Xn.to_numpy()), y=torch.tensor(y),
             edge_index=torch.tensor(edge_index))
    tr_mask = torch.zeros(n, dtype=torch.bool); tr_mask[:i_tr] = True
    ca_mask = torch.zeros(n, dtype=torch.bool); ca_mask[i_tr:i_ca] = True
    te_mask = torch.zeros(n, dtype=torch.bool); te_mask[i_ca:] = True

    torch.manual_seed(config.SEED)
    net = SAGE(g.x.shape[1], HIDDEN)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)
    pos_w = torch.tensor([(y[:i_tr] == 0).sum() / max((y[:i_tr] == 1).sum(), 1)])
    print(f"training GraphSAGE ({EPOCHS} epochs, pos_weight {pos_w.item():.1f}) ...")

    best, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        net.train(); opt.zero_grad()
        out = net(g.x, g.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            out[tr_mask], g.y[tr_mask], pos_weight=pos_w)
        loss.backward(); opt.step()

        if ep % 5 == 0 or ep == EPOCHS:
            net.eval()
            with torch.no_grad():
                sc = torch.sigmoid(net(g.x, g.edge_index)).numpy()
            ap = average_precision_score(y[i_tr:i_ca], sc[i_tr:i_ca])
            if ap > best:
                best, best_state = ap, {k: v.clone()
                                        for k, v in net.state_dict().items()}
            print(f"   epoch {ep:>3}  loss {loss.item():.4f}  calib PR-AUC {ap:.4f}")

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        raw_scores = torch.sigmoid(net(g.x, g.edge_index)).numpy()

    cal = calibrate.fit_calibrator(raw_scores[i_tr:i_ca], y[i_tr:i_ca])
    p_te = cal(raw_scores[i_ca:])
    y_te, amt_te = y[i_ca:].astype(int), amt[i_ca:]

    res = evaluate(p_te, y_te, amt_te)

    w = 96
    print("\n" + "=" * w)
    print("GRAPHSAGE ON A TIME-RESPECTING TRANSACTION GRAPH")
    print("=" * w)
    print(f"nodes {n:,}   edges {edge_index.shape[1]:,}   "
          f"relations {', '.join(RELATIONS)}   K={K_NEIGHBOURS}")
    print("-" * w)
    print(f"{'model':<28}{'ROC-AUC':>10}{'PR-AUC':>10}{'caught':>16}"
          f"{'recall':>9}{'FPR':>8}{'rupee loss':>16}")
    print("-" * w)
    print(f"{'GraphSAGE (2-layer)':<28}{res['roc_auc']:>10.4f}{res['pr_auc']:>10.4f}"
          f"{res['frauds_caught']:>9,}/{res['frauds_total']:<6,}"
          f"{res['recall']:>9.3f}{res['fpr']:>8.2%}{res['rupee_loss']:>16,.0f}")
    ref = {"roc_auc": 0.8859, "pr_auc": 0.4988, "frauds_caught": 1638,
           "recall": 0.403, "fpr": 0.0090, "rupee_loss": 44456606}
    print(f"{'single LightGBM (ref)':<28}{ref['roc_auc']:>10.4f}{ref['pr_auc']:>10.4f}"
          f"{ref['frauds_caught']:>9,}/{res['frauds_total']:<6,}"
          f"{ref['recall']:>9.3f}{ref['fpr']:>8.2%}{ref['rupee_loss']:>16,.0f}")
    print("-" * w)
    print(f"vs LightGBM:  ROC-AUC {res['roc_auc']-ref['roc_auc']:+.4f}   "
          f"PR-AUC {res['pr_auc']-ref['pr_auc']:+.4f}   "
          f"rupees {ref['rupee_loss']-res['rupee_loss']:+,.0f}")
    print("=" * w)

    (config.ARTIFACTS / "gnn.json").write_text(
        json.dumps({"nodes": n, "edges": int(edge_index.shape[1]),
                    "k": K_NEIGHBOURS, "relations": RELATIONS,
                    "graphsage": res, "lightgbm_reference": ref}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'gnn.json'}")


if __name__ == "__main__":
    main()
