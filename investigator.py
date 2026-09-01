"""The investigator layer: TreeSHAP attribution -> analyst brief.

Design follows the pattern in arXiv:2607.19266 (2026), "Toward Auditable Fraud
Detection", which applies a bounded LLM investigation agent to cases the
classifier scores uncertainly, grounded in TreeSHAP attributions. The narrative
framing follows the XAIstories / SHAPstories result that SHAP values turned into
natural-language narratives are substantially more usable than raw attributions.

WHY THIS LAYER EXISTS AT ALL
Dal Pozzolo et al. show fraud operations run under a hard analyst budget. Our
own measurement: at k=200 alerts/day, 14.6% of fraudulent cards are never
reviewed. Analyst minutes are the scarce resource, so every reviewed case must
arrive readable. That is a capacity argument, not a reason to add an LLM for
its own sake.

THE BOUNDARY, AND IT IS NOT NEGOTIABLE
  - The LLM never produces a score. The number comes from the calibrated model.
  - The LLM never decides an action. Deterministic rules do that.
  - The LLM receives only pre-computed, filtered evidence -- never raw rows.
  - Its output is a brief for a human. Nothing downstream consumes it.

An LLM has no business scoring 77 tabular features; gradient boosting is
strictly better at that. It is good at turning a pile of attributions into two
sentences a human can act on. That is the whole job.

Runs without an API key: the deterministic template path always works, and the
LLM path activates when ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import calibrate
import config
import data
import features
import model

REVIEW_LO, REVIEW_HI = 0.20, 0.80   # the uncertain band
TOP_SIGNALS = 6
MAX_CASES = 25

# Plain-English names for the IEEE-CIS columns an analyst would see.
FEATURE_GLOSS = {
    "TransactionAmt": "transaction amount",
    "card1": "card identifier", "card2": "card issuer range",
    "card3": "card country", "card5": "card sub-type",
    "card4": "card network", "card6": "card type (credit/debit)",
    "addr1": "billing region", "addr2": "billing country",
    "dist1": "billing/shipping distance", "dist2": "secondary distance",
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "recipient email domain",
    "DeviceType": "device type", "DeviceInfo": "device model",
    "ProductCD": "product category",
}
for i in range(1, 15):
    FEATURE_GLOSS[f"C{i}"] = f"address/phone count signal C{i}"
for i in range(1, 10):
    FEATURE_GLOSS[f"M{i}"] = f"match flag M{i}"


def gloss(name: str) -> str:
    return FEATURE_GLOSS.get(name, name)


def build_evidence(booster, X_row: pd.DataFrame, feature_names, prob: float,
                   amount_inr: float) -> dict:
    """TreeSHAP attributions for one transaction, reduced to signed signals.

    Uses LightGBM's native pred_contrib (TreeSHAP) so there is no extra
    dependency and the attributions are exact, not sampled.
    """
    contrib = booster.predict(X_row, pred_contrib=True)[0]
    values, bias = contrib[:-1], contrib[-1]

    order = np.argsort(np.abs(values))[::-1][:TOP_SIGNALS]
    signals = []
    for i in order:
        if values[i] == 0:
            continue
        raw = X_row.iloc[0, i]
        # TransactionAmt is USD in the source data; everything else the analyst
        # sees is rupees. Convert so a single brief never mixes currencies.
        if feature_names[i] == "TransactionAmt" and not pd.isna(raw):
            shown = f"Rs {float(raw) * config.USD_TO_INR:,.0f}"
        else:
            shown = None if pd.isna(raw) else str(raw)
        signals.append({
            "feature": feature_names[i],
            "label": gloss(feature_names[i]),
            "value": shown,
            "contribution": float(values[i]),
            "direction": "raises risk" if values[i] > 0 else "lowers risk",
        })
    return {
        "probability": float(prob),
        "amount_inr": float(amount_inr),
        "baseline_log_odds": float(bias),
        "signals": signals,
    }


def template_brief(ev: dict) -> str:
    """Deterministic fallback. Always available, no API key needed."""
    up = [s for s in ev["signals"] if s["contribution"] > 0][:3]
    down = [s for s in ev["signals"] if s["contribution"] < 0][:2]
    parts = [f"Scored {ev['probability']:.1%} on a Rs {ev['amount_inr']:,.0f} "
             f"transaction."]
    if up:
        parts.append("Raising risk: " + ", ".join(
            f"{s['label']}" + (f" = {s['value']}" if s["value"] else "")
            for s in up) + ".")
    if down:
        parts.append("Lowering risk: " + ", ".join(s["label"] for s in down) + ".")
    parts.append("Model is uncertain; analyst judgement required.")
    return " ".join(parts)


PROMPT = """You are writing a short brief for a payments fraud analyst who has \
about 60 seconds per case.

A gradient-boosted model scored this transaction. You did NOT score it and you \
must not produce, restate as your own, or second-guess any probability. You are \
explaining the model's attributions so a human can decide.

Evidence (TreeSHAP attributions, already filtered):
{evidence}

Write at most 3 sentences:
1. What the model reacted to, in plain language.
2. Whether the signals agree or conflict with each other.
3. The single most useful thing the analyst should check next.

Rules: no invented facts beyond the evidence above. No probability estimates of \
your own. No recommendation to block or allow -- that decision is not yours."""


def llm_brief(ev: dict) -> str | None:
    """Optional LLM path. Returns None if unavailable, never raises."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{"role": "user",
                       "content": PROMPT.format(
                           evidence=json.dumps(ev["signals"], indent=2))}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:  # noqa: BLE001 -- degrade to template, never crash
        print(f"  (LLM unavailable: {type(exc).__name__}; using template)")
        return None


def main() -> None:
    print("training ...")
    raw = data.load_raw()
    audit = data.audit_leakage(raw)
    tr_df, ca_df, te_df = data.temporal_split(raw)
    cols = audit["online"]

    X_tr, y_tr, _, cats = features.build(tr_df, cols)
    X_ca, y_ca, _, _ = features.build(ca_df, cols, cats)
    X_te, y_te, amt_te, _ = features.build(te_df, cols, cats)

    booster = lgb.train(model.PARAMS, lgb.Dataset(X_tr, label=y_tr),
                        num_boost_round=model.NUM_ROUNDS,
                        valid_sets=[lgb.Dataset(X_ca, label=y_ca)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    p = calibrate.fit_calibrator(b_ca := booster.predict(X_ca), y_ca)(
        booster.predict(X_te))
    del b_ca

    band = (p >= REVIEW_LO) & (p < REVIEW_HI)
    idx = np.where(band)[0]
    print(f"review band [{REVIEW_LO}, {REVIEW_HI}): {len(idx):,} of {len(p):,} "
          f"transactions ({band.mean():.2%}), fraud rate {y_te[band].mean():.2%}")

    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"LLM path: {'enabled' if use_llm else 'not configured, using template'}")

    names = booster.feature_name()
    order = idx[np.argsort(amt_te[idx])[::-1][:MAX_CASES]]  # largest first

    briefs = []
    for i in order:
        ev = build_evidence(booster, X_te.iloc[[i]], names, p[i], amt_te[i])
        text = (llm_brief(ev) if use_llm else None) or template_brief(ev)
        briefs.append({"row": int(i), "probability": float(p[i]),
                       "amount_inr": float(amt_te[i]),
                       "actual_is_fraud": int(y_te[i]),
                       "top_signals": ev["signals"], "brief": text,
                       "source": "llm" if use_llm else "template"})

    w = 92
    print("\n" + "=" * w)
    print(f"INVESTIGATOR BRIEFS  (top {len(briefs)} review-band cases by amount)")
    print("=" * w)
    for b in briefs[:5]:
        print(f"\ncase row {b['row']}   P(fraud) {b['probability']:.3f}   "
              f"Rs {b['amount_inr']:,.0f}   actual: "
              f"{'FRAUD' if b['actual_is_fraud'] else 'genuine'}")
        print("  " + b["brief"])
    print("\n" + "-" * w)

    # Honest evaluation of the layer: does the ordering carry signal?
    band_rate = float(y_te[band].mean())
    sel_rate = float(np.mean([b["actual_is_fraud"] for b in briefs]))
    print(f"review-band fraud rate           {band_rate:.2%}")
    print(f"fraud rate in the {len(briefs)} briefed cases  {sel_rate:.2%}")
    print("These cases are selected by AMOUNT, not by score, so a higher fraud")
    print("rate here would be incidental. The layer's job is legibility, not")
    print("ranking -- it is not claimed to improve detection.")
    print("=" * w)

    (config.ARTIFACTS / "investigator_briefs.json").write_text(
        json.dumps({"review_band": [REVIEW_LO, REVIEW_HI],
                    "band_size": int(band.sum()),
                    "band_fraud_rate": band_rate,
                    "briefs": briefs}, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'investigator_briefs.json'}")


if __name__ == "__main__":
    main()
