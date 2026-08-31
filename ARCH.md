# Second Look — Architecture

A fraud triage system for card-not-present e-commerce payments.
Submission for the **Razorpay AI Buildathon 2026, Track 02 — AI Risk Manager**.

---

## The one-line claim

> Fraud scoring is a tabular problem with no unstructured input, so we used gradient
> boosting — not an LLM — and we **calibrated** it, because our threshold is chosen by
> minimising **expected rupee loss**, not F1. The LLM writes the analyst's brief.
> It never produces a number, and it never decides money movement.

---

## The problem

For card-not-present fraud, **the merchant eats the loss.** A stolen card buys ₹4,000 of
goods; six weeks later the real cardholder disputes it; the money is clawed back via
chargeback. The merchant loses the money, the goods, and a chargeback fee.

The naive answer is "block suspicious transactions." The trap is that blocking a
*genuine* customer — a **false decline** — costs the merchant the sale, the margin, and
that customer's future business. Across the industry, false declines cost more than fraud.

So the real problem is not detection. It is:

> **Where do you draw the line between blocking and allowing, when both mistakes cost
> money — but different amounts?**

Four things make it hard:

1. **Fraud is rare** (~3.5%). A model that predicts "never fraud" is 96.5% accurate and
   worthless. Accuracy is a lie; precision/recall and cost are what matter.
2. **Costs are asymmetric** and merchant-specific (margin, repeat rate, ticket size).
3. **Labels arrive late.** You learn a transaction was fraud 30–60 days later, when the
   chargeback lands. Any chargeback-derived feature cannot exist at checkout time.
4. **Fraud moves.** Patterns shift in weeks, so evaluation must be on *future* data.

---

## Data

**IEEE-CIS Fraud Detection** (Vesta Corporation, via Kaggle).

| | |
|---|---|
| Rows | 590,540 real card-not-present e-commerce transactions |
| Features | 394 (transaction + identity tables merged) |
| Label | `isFraud` — 20,663 positives |
| Base rate | ~3.5% |

**Stated limitation:** this is US card data, not Indian. We apply an Indian cost model to
a real transaction dataset. That assumption is disclosed here, in the README, and in the
pitch. It is a limitation, not a hidden one.

---

## Pipeline

```
Transaction
    |
    v
[1] ONLINE FEATURE SET  ......... ~50 features computable in <50ms
    |                             offline aggregates documented + EXCLUDED
    v
[2] DETECTOR  ................... LightGBM (fallback: sklearn HistGradientBoosting)
    |                             output = raw ranking score, NOT a probability
    v
[3] CALIBRATOR  ................. isotonic / Platt, fit on a temporal calibration split
    |                             output = P(fraud), an actual probability
    v
[4] DECISION ROUTER  ............ deterministic, thresholds minimise E[rupee loss]
    |                             ALLOW | REVIEW | BLOCK
    |
    +-- ALLOW / BLOCK ........... deterministic. LLM not consulted.
    |
    +-- REVIEW band
            |
            v
       [5] INVESTIGATOR  ........ fixed 5-lookup playbook (deterministic)
            |                     + entity graph (deterministic)
            |                     + LLM synthesis -> analyst brief (no numbers)
            v
       [6] HUMAN QUEUE  ......... always. The brief is attached; the human decides.
```

Every stage writes to an append-only JSON trace:
`txn_id -> features -> raw_score -> calibrated_P -> action -> lookups -> brief -> outcome`

---

## Layer detail

### [1] Online feature set

Only features computable at checkout inside a latency budget.

- **Static:** card type, email domain, device type, product code, amount
- **Online aggregates (cache-backed):** card velocity (count/sum, 1h & 24h), device
  first-seen and lifetime count, email domain age, amount z-score vs the card's recent
  history, billing/shipping match
- **Excluded:** long-window batch aggregates and anything derived from chargebacks
  (`D_*` fields known only post-facto). Documented in `EXCLUDED_FEATURES.md`.

Why this matters: most public work on this dataset uses all 394 features, including
aggregates that could never be computed in time at a real checkout. Restricting the
feature set is a deliberate production constraint, and the cost of that restriction is
reported.

### [2] Detector

Gradient boosting. **No LLM** — an LLM scoring 394 tabular features would be slower,
costlier, and worse than a booster. This is the deliberate "where we chose not to use
one" decision.

**Temporal split on `TransactionDT`:** first 70% train / next 10% calibration /
last 20% test. Never random — a random split trains on transactions that occurred after
the ones being tested, which leaks the future and inflates every score.

### [3] Calibrator

Raw booster output is a **ranking score, not a probability**. Multiplying it by a rupee
cost would be arithmetically meaningless.

Isotonic regression (fallback: Platt scaling) fit on the calibration split, which sits
temporally between train and test and is never seen by the model.

**Validation:** reliability diagram plotted on the **test** set — fitting and validating
on the same split proves nothing.

### [4] Decision router

Deterministic. Thresholds chosen to minimise expected rupee loss:

```
cost(block a genuine txn)  = lost_margin + customer_LTV
cost(allow a fraud txn)    = txn_amount + chargeback_fee

E[loss](T) = Σ over test set, given calibrated P and the action T implies
choose T* = argmin E[loss]
```

All cost parameters live in `config.py` and are varied ±30% in a sensitivity analysis.

**STEP_UP is cut.** A four-action policy requires a friction cost for challenging a
genuine customer, and we cannot defend a number for it on this dataset. Shipping three
actions we can price beats four we cannot. Documented as a scoping decision, not an
oversight.

### [5] Investigator — **not an agent**

No decisions. No probabilities. No choice of what to look up.

**Fixed playbook** (deterministic): card velocity · device history · email domain risk ·
amount anomaly vs card history · billing/shipping mismatch.

**Entity graph** (deterministic): links card ↔ device ↔ email. An information-value
pre-filter strips low-cardinality attributes before graph construction.

**LLM synthesis** — the only LLM in the pipeline. Input: structured playbook results and
the filtered graph. Output (JSON-mode enforced):

```json
{
  "key_signals": [{"signal": "...", "direction": "RISK|SAFE",
                   "weight": "HIGH|MED|LOW", "evidence": "..."}],
  "conflict_summary": "why the signals disagree, if they do",
  "recommendation": "ALLOW | BLOCK | NEEDS_HUMAN",
  "confidence": "HIGH | MEDIUM | LOW",
  "reasoning": "<= 200 tokens, written for a human analyst"
}
```

No `probability` field. The number comes from the calibrated model; the LLM explains the
ambiguity.

### [6] Adjudication

```
P >= T_block   -> BLOCK   (deterministic, LLM not consulted)
P <  T_allow   -> ALLOW   (deterministic, LLM not consulted)
otherwise      -> HUMAN QUEUE, always, with the brief attached
```

The LLM's recommendation is **advisory only**. It never triggers an action. Human
overrides are logged for retraining.

---

## Results we report

### The headline — three policies, one test set

| Policy | Rupee loss |
|---|---|
| Allow everything (no fraud system) | — |
| Block at the **F1-optimal** threshold | — |
| Block at the **rupee-optimal** threshold | — |

**The gap between rows 2 and 3 is the finding:** what optimising the metric everyone
reports actually costs this merchant.

### Supporting

- Precision / recall / PR-AUC on the temporally held-out test set
- Reliability diagram (test set), pre- and post-calibration
- Sensitivity of `T*` to ±30% variation in LTV, margin, chargeback fee
- **Review-band metrics:** how often the LLM's advisory recommendation agrees with
  `isFraud` on the cases the model could not resolve. If it adds no signal, we say so.
- Cost of the online-feature restriction: performance with the full 394 vs our ~50

---

## Guardrails

- LLM never emits a number and never triggers an action
- No auto-block above a configurable amount — those always go to a human
- Information-value pre-filter runs before the LLM sees any entity linkage
- **Strictly defense-only.** No generator, no simulator, no evasion testing, nothing
  offense-capable. Explicitly out of scope per the track rules.
- Full replayable decision trace for every transaction

---

## Prior art — stated, not hidden

Razorpay ships **Vulcan** (payments foundation model covering routing, fraud, RTO risk)
and **Agent Studio** agents including an RTO Shielder and a Dispute Auto-Responder. This
project overlaps their fraud-detection surface, and Track 02 lists "Fraud-spike detector"
and "Abuse-ring sentinel" as example directions.

We are not claiming novelty of problem. The contribution is methodological:
calibration, cost-optimal thresholds, an enforced inference-latency constraint, and
honest reporting of what the LLM does and does not do.

---

## Scope — 4 days (1–5 September)

| Day | Work |
|---|---|
| **1 (Sep 1)** | Data load, temporal split, online feature set, leakage audit, first booster |
| **2 (Sep 2)** | Calibration + reliability diagram, cost model, **three-policy baseline table** |
| **3 (Sep 3)** | Playbook lookups, entity graph, LLM brief, review-band metrics |
| **4 (Sep 4)** | Evaluation tables, README, 5-minute video |
| Sep 5 | Submit |

**Cut for time:** STEP_UP action, multi-model comparison, any UI beyond a CLI report,
hyperparameter search beyond sensible defaults.

**Priority if we run out of time:** layers 1–4 alone are a complete Track 02 submission —
they satisfy "measured precision and recall on a held-out test set" and "honest metrics
including false-positive cost". Layer 5 is the first thing to drop.

---

## What broke

*To be written from what actually happens. Not pre-scripted.*

---

## Honest risks

- **Crowding.** IEEE-CIS is one of the most-worked public datasets in existence. The
  model is not the differentiator; calibration, the cost model, the latency constraint,
  and the temporal discipline are.
- **US data, Indian cost model.** Disclosed above.
- **The LLM layer is small by design.** This is a tabular problem with no unstructured
  input. We would rather ship a small, honest LLM role than a large decorative one.
