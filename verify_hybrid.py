"""Does the GNN add anything to the ensemble, despite losing on its own?

GraphSAGE scored PR-AUC 0.2827 against the booster ensemble's 0.5093. That alone
does not settle whether it belongs in the blend: a weaker model earns its place
if its ERRORS are decorrelated from the strong one's, because the blend then
covers cases the ensemble misses.

So the diagnostic that matters is not the GNN's standalone score, it is the
correlation between its predictions and the ensemble's. High correlation means
it has learned the same thing worse and adds nothing. Low correlation means it
sees something the trees do not, and is worth its weight.

Measured here:
  - Spearman and Pearson correlation between GNN and ensemble scores
  - correlation on the FRAUD subset specifically (agreement where it matters)
  - blends from 0% to 40% GNN weight, priced in rupees
  - whether the GNN ranks any frauds highly that the ensemble ranks low
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.data import Data

import calibrate
import config
import data
import features
from gnn import EPOCHS, HIDDEN, K_NEIGHBOURS, LR, SAGE, build_edges, evaluate
from leaderboard import LGB_P, XGB_P

BLENDS = [0.0, 0.05, 0.10, 0.20]
SEEDS = [42, 7, 2024]


def numeric(df):
    return df.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c).fillna(-999)


def run(seed: int) -> dict:
    print("loading ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    cols = audit["online"]
    df = raw.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    i_tr = int(n * config.TRAIN_FRAC)
    i_ca = int(n * (config.TRAIN_FRAC + config.CALIB_FRAC))

    X, y_all, amt, _ = features.build(df, cols)
    y_all = y_all.astype(int)
    X_tr, X_ca, X_te = X.iloc[:i_tr], X.iloc[i_tr:i_ca], X.iloc[i_ca:]
    y_tr, y_ca, y_te = y_all[:i_tr], y_all[i_tr:i_ca], y_all[i_ca:]
    amt_te = amt[i_ca:]
    N_tr, N_ca, N_te = numeric(X_tr), numeric(X_ca), numeric(X_te)

    # --- ensemble -----------------------------------------------------------
    print("training ensemble (LGB + XGB + RF) ...")
    m = lgb.train(dict(LGB_P, seed=seed, bagging_seed=seed), lgb.Dataset(X_tr, label=y_tr), num_boost_round=300)
    xm = xgb.train(dict(XGB_P, seed=seed), xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True),
                   num_boost_round=300)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1,
                                random_state=seed).fit(N_tr, y_tr)
    d_ca = xgb.DMatrix(X_ca, enable_categorical=True)
    d_te = xgb.DMatrix(X_te, enable_categorical=True)
    ens_ca = np.mean([m.predict(X_ca), xm.predict(d_ca),
                      rf.predict_proba(N_ca)[:, 1]], axis=0)
    ens_te = np.mean([m.predict(X_te), xm.predict(d_te),
                      rf.predict_proba(N_te)[:, 1]], axis=0)

    # --- GNN ----------------------------------------------------------------
    print(f"building graph (K={K_NEIGHBOURS}) ...")
    edge_index = build_edges(df)
    Xn = X.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c)
    Xn = Xn.astype("float32").replace([np.inf, -np.inf], np.nan)
    mu, sd = Xn.iloc[:i_tr].mean(), Xn.iloc[:i_tr].std().replace(0, 1)
    Xn = ((Xn - mu) / sd).fillna(0.0)
    g = Data(x=torch.tensor(Xn.to_numpy()),
             y=torch.tensor(y_all.astype("float32")),
             edge_index=torch.tensor(edge_index))
    tr_mask = torch.zeros(n, dtype=torch.bool); tr_mask[:i_tr] = True

    torch.manual_seed(seed)
    net = SAGE(g.x.shape[1], HIDDEN)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)
    pw = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                      dtype=torch.float32)
    print(f"training GraphSAGE ({EPOCHS} epochs) ...")
    for ep in range(1, EPOCHS + 1):
        net.train(); opt.zero_grad()
        out = net(g.x, g.edge_index)
        F.binary_cross_entropy_with_logits(out[tr_mask], g.y[tr_mask],
                                           pos_weight=pw).backward()
        opt.step()
        if ep % 20 == 0:
            print(f"   epoch {ep}")
    net.eval()
    with torch.no_grad():
        gnn_raw = torch.sigmoid(net(g.x, g.edge_index)).numpy()

    cal_e = calibrate.fit_calibrator(ens_ca, y_ca)
    cal_g = calibrate.fit_calibrator(gnn_raw[i_tr:i_ca], y_ca)
    pe, pg = cal_e(ens_te), cal_g(gnn_raw[i_ca:])

    # --- the diagnostic that decides it ------------------------------------
    sp = spearmanr(pe, pg).statistic
    pe_r = pearsonr(pe, pg).statistic
    fr = y_te == 1
    sp_f = spearmanr(pe[fr], pg[fr]).statistic

    w = 92
    print("\n" + "=" * w)
    print("DOES THE GNN SEE ANYTHING THE ENSEMBLE DOES NOT?")
    print("=" * w)
    print(f"Spearman correlation, all transactions   {sp:.4f}")
    print(f"Pearson correlation,  all transactions   {pe_r:.4f}")
    print(f"Spearman correlation, frauds only        {sp_f:.4f}")
    # frauds the ensemble ranks low but the GNN ranks high
    e_rank = pd.Series(pe).rank(pct=True).to_numpy()
    g_rank = pd.Series(pg).rank(pct=True).to_numpy()
    rescued = int(((e_rank < 0.90) & (g_rank > 0.99) & fr).sum())
    print(f"frauds the ensemble ranks below the 90th percentile but the GNN "
          f"puts above the 99th: {rescued}")
    print("-" * w)

    rows = {}
    for b in BLENDS:
        p = (1 - b) * pe + b * pg
        rows[b] = evaluate(p, y_te, amt_te)

    print(f"{'GNN weight':>12}{'ROC-AUC':>10}{'PR-AUC':>10}{'caught':>16}"
          f"{'recall':>9}{'FPR':>8}{'rupee loss':>16}")
    print("-" * w)
    for b, v in rows.items():
        tag = f"{b:.0%}" + ("  (ensemble)" if b == 0 else
                            "  (GNN only)" if b == 1 else "")
        print(f"{tag:>12}{v['roc_auc']:>10.4f}{v['pr_auc']:>10.4f}"
              f"{v['frauds_caught']:>9,}/{v['frauds_total']:<6,}"
              f"{v['recall']:>9.3f}{v['fpr']:>8.2%}{v['rupee_loss']:>16,.0f}")
    print("-" * w)

    base = rows[0.0]["rupee_loss"]
    best_b = min(rows, key=lambda k: rows[k]["rupee_loss"])
    print(f"best blend: {best_b:.0%} GNN  "
          f"(Rs {base - rows[best_b]['rupee_loss']:+,.0f} vs pure ensemble)")
    if best_b == 0.0:
        print("The GNN adds nothing at any weight. It is not complementary --")
        print("it learned the same structure the count features already encode,")
        print("and learned it worse.")
    return {"seed": seed, "spearman": float(sp), "rescued": rescued,
            "blends": {b: rows[b]["rupee_loss"] for b in BLENDS}}



def main() -> None:
    import statistics as st
    runs = [run(s_) for s_ in SEEDS]
    w = 84
    print()
    print("=" * w)
    print("IS THE HYBRID GAIN REAL?  (3 seeds)")
    print("=" * w)
    print(f"{'seed':<8}{'spearman':>11}{'rescued':>10}" +
          "".join(f"{b:>16.0%}" for b in BLENDS))
    print("-" * w)
    for r in runs:
        print(f"{r['seed']:<8}{r['spearman']:>11.4f}{r['rescued']:>10}" +
              "".join(f"{r['blends'][b]:>16,.0f}" for b in BLENDS))
    print("-" * w)
    print(f"{'gain vs 0%':<29}" + "".join(
        f"{st.mean([r['blends'][0.0] - r['blends'][b] for r in runs]):>16,.0f}"
        for b in BLENDS))
    print(f"{'sd of gain':<29}" + "".join(
        f"{st.stdev([r['blends'][0.0] - r['blends'][b] for r in runs]):>16,.0f}"
        for b in BLENDS))
    print("-" * w)
    for b in BLENDS[1:]:
        g = [runs[i]['blends'][0.0] - runs[i]['blends'][b] for i in range(len(runs))]
        pos = sum(x > 0 for x in g)
        mean, sd = st.mean(g), st.stdev(g)
        verdict = ("ROBUST" if pos == len(g) and sd < abs(mean) / 2 else
                   "directional" if pos == len(g) else "NOT ROBUST")
        print(f"  {b:.0%} GNN: positive in {pos}/{len(g)} seeds, "
              f"mean Rs {mean:+,.0f}, sd Rs {sd:,.0f} -> {verdict}")
    print("=" * w)
    json.dump({"seeds": runs}, open(config.ARTIFACTS / "verify_hybrid.json", "w"),
              indent=2, default=str)


if __name__ == "__main__":
    main()
