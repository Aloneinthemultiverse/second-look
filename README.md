# Second Look

**Cost-optimal fraud triage for card-not-present payments.**
Razorpay AI Buildathon 2026 — Track 02, AI Risk Manager.

> Fraud scoring is a tabular problem with no unstructured input, so the detector is
> gradient boosting, not an LLM. The threshold is chosen by minimising expected
> rupee loss, not F1. The LLM writes the analyst's brief — it never produces a
> number and never decides whether money moves.

---

## The problem

For card-not-present fraud, **the merchant absorbs the loss.** A stolen card buys
₹4,000 of goods; six weeks later the cardholder disputes it and the money is
clawed back. The merchant loses the money, the goods, and a chargeback fee.

The obvious fix — block suspicious transactions — creates the harder problem.
Declining a *genuine* customer costs the sale, the margin, and that customer's
future business. Across the industry, false declines cost more than fraud.

So the real question is not "is this fraud?" It is:

> **Where do you draw the line, when both mistakes cost money — but different
> amounts, and the amounts differ per transaction?**

---

## Headline results

Temporally held-out test set: **118,109 transactions, 3.44% fraud, 42 days.**

| Metric | Value |
|---|---|
| PR-AUC | 0.5091 ± 0.0028 (3 seeds) |
| ROC-AUC | 0.8870 |
| **Card Precision@10/day** | **0.919** — 26.7× lift over base rate |
| Calibration error (ECE) | 0.0067 → **0.0032** after Platt |
| Single-row inference | p50 **7.1 ms**, p95 11.1 ms |
| Savings vs no fraud system | **23.98%** (honest), 24.82% best strategy |

**The single most important number is Card Precision@10 = 0.919.** PR-AUC of 0.51
makes this model look mediocre, but PR-AUC measures performance over the whole
stream — work no analyst team will ever do. Under a realistic alert budget
(Dal Pozzolo et al.), **92% of the top ten alerts per day are genuine fraud.**

---

## What I got wrong

This section is first on purpose. Five claims did not survive checking.

**1. "Optimising F1 instead of rupees costs ₹964,857." — Dead.**
Ran three seeds and three split points. The gap ranged from ₹0 to ₹710,488 —
standard deviation larger than the mean. Traced it to threshold-grid resolution,
not to anything real. The claim was one grid, one seed. Cut.

**2. My reported savings were inflated by ₹713,146.** I was sweeping the decision
threshold on the test set and then reporting results on that same test set —
fitting to my evaluation data. Moving threshold selection to the calibration
slice dropped honest savings from 25.20% to **23.98%**. See `cost_sensitive.py`,
arm `A0`, which is kept solely to quantify the bias.

**3. "The optimal threshold varies 7× across merchant economics."** True on
IEEE-CIS (3.4% fraud). Does **not** generalise: on ULB (0.13% fraud) two of four
merchant profiles are told to block nothing at all. It is a base-rate-conditional
finding, not a law.

**4. The investigation playbook adds no detection signal.** Measured, not assumed:
inside the review band, the model score alone ranks better (ROC-AUC 0.6735) than
score plus the five playbook lookups (0.6401). The booster already consumes the
count signals, device and email fields the playbook recomputes.

My first version of that experiment was broken — it median-imputed missing values,
deleting the fact that "device never seen before" is itself a signal, and used a
linear model on non-linear features. Fixing both lifted playbook-only from 0.4485
(below random) to 0.5621, and the conclusion held regardless. The layer's value is
legibility for an analyst under a budget, not detection, and that is now measured
rather than asserted. Whether an analyst *decides* better with a brief needs human
subjects and is not claimed here.

**5. "The model is weak on high-value fraud" — I measured that against the wrong
policy.** The audit reported caught fraud averaging Rs 11,062 against missed
Rs 14,899, a Rs 3,837 gap. But the audit used a *global* threshold, while the
shipped policy uses per-instance thresholds tau*(x) from Elkan/Bahnsen, which
block more readily on large amounts by construction. Re-measured correctly the
gap is **Rs 1,124** — 71% smaller — and value-recall is 39.2%, not 36.9%.

I then tried two documented remedies anyway (`high_value.py`): cost-proportionate
weighting with recalibration (Zadrozny et al. 2003) and a dedicated high-amount
model. **Both made things worse** on value-recall and rupee loss. Cost-weighting
did narrow the caught/missed gap furthest (Rs 587), confirming it redirects
attention to expensive fraud as advertised — it just costs more discrimination
(precision 0.608 -> 0.588) than the rebalancing is worth. Baseline retained.

I also claimed Platt calibration "provably preserves ranking exactly." In exact
arithmetic yes; in floating point, saturation tied 21 of 41,390 scores on ULB.
Isotonic tied 41,367. The mechanism holds, the wording was wrong.

---

## Architecture

```
transaction
    │
    ▼
[1] ONLINE FEATURES ......... 77 of 434 columns, computable at checkout
    │                          339 V_* excluded (latency), 15 D_* (time drift)
    ▼
[2] DETECTOR ................ LightGBM. No LLM — an LLM scoring 77 tabular
    │                          features would be slower, costlier and worse.
    ▼
[3] CALIBRATOR .............. Platt, fit on a temporal calibration slice.
    │                          Raw scores × rupees is meaningless arithmetic.
    ▼
[4] DECISION ................ ALLOW / REVIEW / BLOCK, per-instance thresholds
    │                          derived from costs (Elkan + Chow), bounded by
    │                          analyst capacity (Dal Pozzolo)
    ▼
[5] INVESTIGATOR ............ TreeSHAP attributions → analyst brief.
    │                          LLM writes prose. It never scores, never decides.
    ▼
[6] HUMAN ................... reviewed cases always end with a person
```

**The LLM boundary is enforced in code, not just described.** `investigator.py`
receives only pre-computed, filtered attributions, emits no numeric field, and
nothing downstream consumes its output.

---

## Research implemented

| Paper | What it gave this project |
|---|---|
| **Elkan (2001)**, *Foundations of Cost-Sensitive Learning* | Confirmed calibrate-then-threshold is correct; τ\* = C_FP/(C_FP+C_FN) |
| **Bahnsen et al.**, instance-dependent cost-sensitive learning | Costs vary per transaction → per-instance thresholds (+₹384,567) |
| **Chow (1970)**, classification with a reject option | The review band is *derived* from costs, not invented |
| **Dal Pozzolo et al. (TNNLS 2017)** | Alert budget, Card Precision@k — reframed what "good" means |
| **arXiv:2607.19266 (2026)** | Precedent for bounded LLM investigation on uncertain cases |

**Combining Chow with a capacity constraint produced a result none of them
state:** at 80% analyst accuracy, reviewing 500 cases/day is *worse than having no
fraud system at all* (₹60.9M vs ₹58.5M). Analyst accuracy caps useful review
capacity. Value per review falls from ₹14,221 at k=10 to ₹1,575 at k=500.

---

## Deployment-evidence checklist

Scored against the minimum checklist published in *Operational Evidence Gaps for
LLMs in Fraud Detection and Trust-and-Safety Workflows* (arXiv:2607.13078, 2026).

| Required evidence | Status |
|---|---|
| **Latency budget** | ✅ measured, p50 7.1 ms / p95 11.1 ms single-row (model scoring only; feature retrieval excluded) |
| **Cost per decision** | ✅ explicit rupee cost model, all parameters in `config.py`, varied ±30% |
| **Decision threshold** | ✅ per-instance τ\*(x) from costs; selected off-test, never on the evaluation slice |
| **Explanation integrity** | ✅ TreeSHAP attributions; the LLM emits no numeric field and triggers no action |
| **Adversarial pressure** | ❌ **not evaluated.** Track 02 is strictly defence-only, so no attack generation was built. This is a real gap in deployment evidence, declared rather than glossed. |

Four of five. The fifth is deliberately out of scope under the competition rules.

## Benchmark context

Combinatorial Fusion Analysis reports **ROC-AUC 0.9405** on IEEE-CIS
(arXiv:2606.10393, 2026) by fusing Random Forest, XGBoost and LightGBM.

This project reaches **0.8870**. The difference is explained, not excused:
that work fuses three models over all 394 features; this uses a single model over
the 77 features computable inside a checkout latency budget, on a temporal split,
with no hyperparameter tuning. `restriction_cost.py` measures what the feature
restriction alone costs: 0.031 PR-AUC and Rs 1.1M.

The goal here was never a leaderboard number. It was the decision layer on top of
whatever score you have.

---

## Verification

| Check | Result |
|---|---|
| 3 random seeds | PR-AUC 0.5061–0.5116 (sd 0.0028) |
| 3 temporal split points | PR-AUC 0.4999–0.5409 |
| 5 consecutive test periods | 0.4736–0.5702, no decay |
| **Second dataset (ULB, 0.13% fraud)** | ROC-AUC 0.9694 — method transfers |
| Calibration within amount buckets | worst bias 0.38% — cost model valid where amounts are large |
| Degeneracy | 111,512 distinct scores, 76/77 features used, top feature 10.1% gain |
| Feature drift train→test | 0 of 46 numeric features shifted > 0.5 sd |
| Unit tests | 11 passing |

---

## Reproducing

```bash
pip install -r requirements.txt
python fetch_data.py     # needs ~/.kaggle/kaggle.json + accepted competition rules
python run_all.py        # every result, in order (--quick to skip retraining stages)
```

Every stage writes JSON to `artifacts/`. All models train with `seed=42`.

---

## Limitations

Stated plainly, because they affect how the numbers should be read.

- **IEEE-CIS is US card data.** An Indian cost model is applied to it. Disclosed,
  not hidden.
- **The cost parameters are assumptions**, not measurements: ₹2,500 LTV, 25%
  margin, ₹1,200 chargeback fee, USD→INR at 88. `sensitivity.py` varies each ±30%.
- **The latency figure is model scoring only.** Feature retrieval from a cache is
  not included and would add to the budget.
- **Reviews are assumed to resolve correctly** in the base case. `chow_capacity.py`
  quantifies the damage at 90% and 80% analyst accuracy.
- **No hyperparameter tuning.** The booster runs on sensible defaults.
- **Not built:** automated retraining, drift monitoring, feedback ingestion from
  analyst decisions. Real production requirements, out of scope for the timeframe.

---

## What broke, and how I got out

I calibrated the model with isotonic regression because it is more flexible than
Platt. Then I noticed PR-AUC had dropped from 0.5095 to 0.4881 between the raw and
calibrated scores — for a transformation that should not change ranking at all.

Isotonic regression is a step function. It had collapsed **111,512 distinct scores
into 167 levels.** Scores tied inside a level have no ordering, so every ranking
metric degrades. I switched to Platt scaling, which is strictly monotonic and
provably preserves ranking, and it also calibrated better (ECE 0.0032 vs 0.0049).

I only found it because I measured PR-AUC on *both sides* of the calibrator instead
of assuming calibration is free.

The second one was worse, and mine. My savings figure was 25.20%. I was choosing
the decision threshold by sweeping it on the test set, then reporting the result on
that same test set — marking my own homework. Moving threshold selection to the
calibration slice cost **₹713,146 of savings that were never real.** The honest
number is 23.98%.

Both bugs produced *better* numbers, which is exactly why neither would have been
caught by looking at the results.
