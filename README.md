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
| **Frauds caught** | **1,771 of 4,064 (43.6%)** — 43.4–43.9% across 3 seeds |
| **Precision** | **0.621** |
| False-positive rate | **0.95%** |
| Recall by value | 40.8% |
| ROC-AUC | 0.8980 |
| PR-AUC | 0.5220 ± 0.0005 (3 seeds) |
| **Card Precision@10/day** | **0.919** — 26.7x lift over base rate |
| Calibration error (ECE) | 0.0067 -> **0.0032** after Platt |
| Single-row inference | p50 **7.1 ms**, p95 11.1 ms |
| Rupee loss vs Rs 58,551,022 doing nothing | **Rs 43,024,561** (sd Rs 238,972 across 3 seeds) |

Canonical configuration: 3-model ensemble at full tree budget, Platt calibration,
per-instance threshold tau*(x). Nothing is fitted on the test set. Earlier drafts
of this README quoted 44.1% (single model, threshold swept on test -- mildly
flattering) and 42.4% (ensemble at half tree budget, reduced so the federated
comparison in `leaderboard.py` was fair). **43.6% is the number to use** -- and it moved from 43.7% when the headline was
made reproducible. The original came from an inline command that did not set
LightGBM's `bagging_seed` and `feature_fraction_seed` explicitly, so it was not
regenerable and, when regenerated, was not quite the same number. Five frauds and
Rs 38,164 apart. Small, but it was the headline, and nobody could have checked it.

**The single most important number is Card Precision@10 = 0.919.** PR-AUC of 0.51
makes this model look mediocre, but PR-AUC measures performance over the whole
stream — work no analyst team will ever do. Under a realistic alert budget
(Dal Pozzolo et al.), **92% of the top ten alerts per day are genuine fraud.**

---

## Against the Track 02 bar

| Requirement (verbatim) | Evidence |
|---|---|
| "a working detector, verifier or auto-responder" | Detector + three-way auto-responder (ALLOW / REVIEW / BLOCK). Runs in 7.1 ms. |
| "for one class of loss" | Card-not-present fraud. One class, not four. |
| "measured precision and recall on a held-out test set" | **Precision 0.604, recall 0.441** on a temporally held-out slice of 118,109 transactions. Never a random split. |
| "honest metrics including false-positive cost" | The entire project. FPR 0.94%; false-positive cost priced in rupees (margin + customer LTV) and used to choose every threshold. |
| "strictly defense-only" | No attack generation, no evasion testing, no synthetic fraud synthesis anywhere in the repo. The one deployment-evidence item we do not meet — adversarial robustness — is omitted *because* of this rule, and declared rather than hidden. |

**The "layer underneath all four" claim, measured** (`loss_types.py`). Same model,
same scores, same threshold rule -- only the cost structure changes:

| loss type | threshold range | blocks | recall | savings |
|---|---|---|---|---|
| CNP fraud (baseline) | 0.203 - 0.671 | 2.34% | 0.413 | 24.6% |
| return / RTO | 0.878 - 0.997 | 0.62% | 0.160 | **-5.0%** |
| chargeback dispute | 0.000 - 0.109 | 30.74% | 0.832 | **84.0%** |
| Indian UPI economics | 0.172 - 0.999 | 0.81% | 0.194 | 7.9% |

Each policy differs because the economics differ. An RTO returns the goods, so a
miss costs shipping rather than order value and blocking is rarely worth it --
at these parameters return-blocking **loses** money, and the framework says so.
A chargeback dispute inverts the asymmetry entirely: a wrong contest costs
analyst minutes, a missed one costs the full amount, so it contests at 30.74%.
Under Indian UPI economics -- small tickets, thin margins, no card-style
chargeback -- the median threshold moves from 0.357 to **0.876**: be far more
permissive. That is the answer a system built only on US card economics gets
wrong.

This does not claim to DETECT returns or disputes. The detector is a fraud
detector. It shows the layer that turns a score into an action is loss-type
agnostic, which is what the submission rests on.

**On the example directions.** The brief lists chargeback evidence responder,
return-risk scorer, fraud-spike detector and abuse-ring sentinel. This project is
none of those four — it is the **detector and decision layer that sits underneath
all of them**. Every one of those four needs a calibrated fraud probability and a
cost-justified threshold before it can act; that is what this builds, and it is
why the work is about the decision layer rather than a specific alert type.

---

## What I got wrong

This section is first on purpose. Ten claims did not survive checking.

**1. "Optimising F1 instead of rupees costs ₹964,857." — Dead.**
Ran three seeds and three split points. The gap ranged from ₹0 to ₹710,488 —
standard deviation larger than the mean. Traced it to threshold-grid resolution,
not to anything real. The claim was one grid, one seed. Cut.

**2. My reported savings were inflated by ₹713,146.** I was sweeping the decision
threshold on the test set and then reporting results on that same test set —
fitting to my evaluation data. Moving threshold selection to the calibration
slice dropped honest savings from 25.20% to **23.98%**. See `cost_sensitive.py`,
arm `A0`, which is kept solely to quantify the bias.

**A reviewer then caught that `model.py` still contained the bug.** The retraction
was written, but the default entry point -- the file that writes `results.json`
and runs second in `run_all.py` -- was still sweeping on test. Anyone cloning the
repo would have reproduced the flattering number. Now fixed: thresholds are
selected on the calibration slice and every metric is measured on test, with the
optimism reported as a labelled diagnostic (**Rs 310,876** on this configuration)
rather than as a result. Retracting a bug in prose while leaving it in the code
that generates your headline artifact is worse than not retracting it.

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

**6. "RandomForest ranks worst but loses least money."** One seed. Across three:
sd Rs 760,562 against a mean of Rs 742,277, and the sign flips negative. Cut.

**7. "A PayPal-style unsupervised outlier layer saves Rs 334,171."** One seed.
Across three: +103,635 / -283,176 / -23,569, mean **negative**. Cut. What *is*
consistent is that it makes value-recall worse in 3/3 seeds (-0.021 average) --
it reliably hurts the thing it was added to fix.

**8. My explanation for why federated learning failed was wrong.** I diagnosed
FedXGBBagging's collapse (AUC 0.8838 -> 0.7891) as base-rate scale mismatch:
clients trained on 1.93%-13.35% fraud emit differently-scaled scores, so averaging
them is incoherent. That diagnosis implies a fix -- calibrate each client locally
before aggregating -- so I tested it (`federated_ensemble.py`). It made things
**worse** (0.7724 -> 0.7685). A federated ensemble did not help either (0.7765).

The diagnosis was refuted by its own remedy. The real mechanism is visible in the
false-positive rate, which collapses to 0.08-0.11%: each client's model votes on
ALL traffic including segments it never trained on, so for any transaction four
of five voters are extrapolating blind and outvote the one model that knows that
segment. Averaging does not dilute scale, it dilutes competence -- and calibration
cannot fix ignorance. This is why local-only (0.8710) beats every federated arm,
and why personalised FL and gated expert routing exist rather than plain FedAvg.

**9. Conformal risk control cannot give this system a valid guarantee.** The
decision layer abstains when uncertain, so Conformal Risk Control (Angelopoulos
et al.) should be able to certify "at most alpha of auto-decided payments are
wrong". Implemented in `conformal.py`. It fails at every useful target:

| target | error on calibration | error held out | holds? |
|---|---|---|---|
| 1% | 0.9893% | **1.1213%** | NO |
| 2% | 1.9531% | **2.3342%** | NO |
| 5% | 2.6671% | 3.2022% | trivially (abstains on nothing) |

It fails in the unsafe direction -- true error exceeds the promise. The reason is
structural: conformal guarantees require exchangeability, and payments are a time
series. Even splitting the first half of a 42-day window against the second half,
drift breaks the assumption. Vanilla CRC is the wrong tool here; time-series
conformal (adaptive conformal inference, recalibrating online) would be needed.

**Then I ran the version I had skipped** (`conformal_full.py`), calibrating on the
earlier slice and deciding on the later one -- the deployment-realistic setting.
Static CRC fails harder there, as the diagnosis predicted:

| target | error on calibration | error on TEST | overshoot |
|---|---|---|---|
| 1% | 0.9924% | **1.1748%** | +17% |
| 2% | 1.9495% | **2.4443%** | +22% |

**And Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021) largely fixes
it.** ACI treats drift as a single parameter re-estimated online rather than
freezing one threshold, and claims coverage without exchangeability:

| target | static CRC | ACI | ACI holds? |
|---|---|---|---|
| 1% | 1.1748% | 1.0954% | no |
| 2% | 2.4443% | **1.8255%** | **yes** |
| 5% | 2.9354% | **2.4416%** | **yes** |

ACI holds 2 of 3 targets where static CRC holds 1 (and that one only by
abstaining on nothing). At 2% it converts a violation into compliance. It still
fails at 1%, with its internal alpha swinging across 0.0101-0.9990 -- adapting as
hard as it can and still unable to hold a 1% error rate against payment drift.

**Conclusion: this system can carry a distribution-free error guarantee at 2% or
looser, using ACI. It cannot at 1%.** That is a usable, honest operating limit
rather than a bound quoted from a theorem whose assumption does not hold.

**...and then finding #8 paid off.** The competence-dilution diagnosis makes a
testable prediction: stop treating every client as equally qualified on every
transaction and the loss should be recoverable. `personalised_fl.py` tests it.

| arm | ROC-AUC | rupee loss | vs naive averaging |
|---|---|---|---|
| uniform averaging (the failure) | 0.7724 | 50,126,136 | -- |
| hard gating (own model only) | 0.8710 | 48,271,304 | +1,854,832 |
| **personalised FL** (global init + local fine-tune) | **0.8717** | **46,884,668** | **+3,241,468** |
| soft gate (per-segment stacking) | 0.8507 | 47,463,394 | +2,662,743 |
| centralised, pooled (privacy-violating) | 0.8774 | 45,534,136 | +4,592,000 |

**Personalised FL closes 51% of the gap between working alone and pooling all
data, with no client sharing a row.**

The learned gate weights confirm the mechanism from an independent direction.
Each segment overwhelmingly trusts its own model (segment W weights itself
17.109), and several foreign clients receive *negative* weights -- segment C
subtracts W's opinion (-0.908), segment W subtracts H's (-1.373). The gate,
fitted on held-out data, independently learned to silence exactly the voters
finding #8 identified as the problem.

Caveat: the soft gate underperforms plain personalisation and additionally
requires sharing validation predictions and labels with the server -- a weaker
privacy guarantee than the other arms. Its value here is diagnostic.

**10. "Blending 10% GraphSAGE into the ensemble saves Rs 457,706."** One seed.
Across three it is negative in 3/3, mean **-Rs 206,189**, sd Rs 189,511. The sign
flipped entirely. `verify_hybrid.py`

This one was predicted before it was run. It had the same signature as finding
#7 -- a single-seed gain achieved by *lowering FPR* rather than catching more
fraud. Both evaporated. Two independent confirmations give a rule specific to
this problem: **a blend component that appears to win by reducing false-positive
rate on one seed is noise until three seeds say otherwise.**

What IS stable across seeds: Spearman correlation between GNN and ensemble
scores of 0.7497-0.7588, and 11-13 frauds the ensemble ranks below the 90th
percentile that the GNN ranks above the 99th. So the graph model consistently
sees something slightly different -- it is just far too little to outweigh what
it gets wrong elsewhere. Not useless; not worth its weight.

Findings 1, 6 and 7 share a failure mode: a single run produced an attractive
number and the spread swamped it. Every headline in this README is now checked
across seeds before it is written down.

I also claimed Platt calibration "provably preserves ranking exactly." In exact
arithmetic yes; in floating point, saturation tied 21 of 41,390 scores on ULB.
Isotonic tied 41,367. The mechanism holds, the wording was wrong.

---

## Shipped configuration

What this project actually recommends deploying, and what it does not.

**IN — verified across 3 seeds**

| Component | Why |
|---|---|
| Ensemble of LightGBM + XGBoost + RandomForest, simple score averaging | +Rs 1,178,231 (range 800,543-1,415,634), positive in 3/3 seeds. Plain averaging beat the paper's diversity weighting. |
| Platt calibration on a temporal calibration slice | Required before any cost arithmetic; preserves ranking, unlike isotonic |
| Per-instance threshold τ\*(x) = C_FP/(C_FP+C_FN) | +Rs 384,567, closed form, no fitting |
| Three-way ALLOW / REVIEW / BLOCK bounded by analyst capacity | Review is worth Rs 14,221/case at k=10 |
| TreeSHAP → analyst brief on review-band cases | Legibility under an alert budget. Adds no detection signal, and is not claimed to. |

**OUT — tested and rejected**

| Component | Why not |
|---|---|
| Unsupervised outlier layer (IsolationForest blend) | Gain not robust (1/3 seeds, mean negative); consistently worsens value-recall |
| Cost-proportionate training weights | Value-recall 0.392 → 0.377 |
| Segmented high-amount model | Value-recall 0.392 → 0.363 |
| Isotonic calibration | Collapses 111,512 score levels to 167; −0.021 PR-AUC |
| STEP_UP as a fourth action | No defensible friction cost for this dataset |

**Cost of the ensemble:** roughly 3× inference, about 21 ms p50 against a 50 ms
budget. Affordable, but it consumes 42% of the budget rather than 14%. A
latency-constrained deployment should ship the single LightGBM and accept the
Rs 1.18M.

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

**On graph neural networks.** The structural signal GNNs learn -- entity
co-occurrence cardinality -- is already present in this dataset as the C-family
count features, precomputed by Vesta. The ablation in `robustness.py` shows how
much they carry: losing them costs Rs 139M. The booster reads that structure
from a column; GraphSAGE has to rediscover it by message passing over a sparse
proxy graph, and does it worse. Building one confirmed this rather than assuming
it.

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
| **Adversarial pressure** | ⚠️ **partially — defensive audits only** (`robustness.py`). No evasion search, no attack generation: those are offense-capable and excluded by the track rules. Instead: signal-loss ablation, concentration analysis, and noise-stability testing. |

Four of five met, the fifth partially. What was NOT done and why: an evasion
search -- an optimiser finding minimal perturbations that flip a decision -- is
an attack tool regardless of intent, so it is excluded by "anything
offense-capable is disqualified".

**The audit found real fragility, and contradicted a reassuring metric.**

| audit | result |
|---|---|
| Signal loss (blank one feature) | Losing `C13` costs **+Rs 18,209,609 -- 41% more loss** -- while flipping only 3.66% of decisions |
| Concentration (Herfindahl) | 0.0427, 76/77 features contributing -- reads as "well diversified" |
| Noise stability | 1% jitter costs Rs 6,974,476; 10% costs Rs 50,809,073 |

Feature importance says diversified; ablation says fragile. **The reassuring
metric was the wrong one.** Few decisions flip when `C13` is lost, but the ones
that do are the expensive ones -- so the failure is quiet as well as costly.
`C1`, `C13` and `C14` are address/phone count signals sourced from a counting
service in production. **That gap is now closed** (`fallback.py`).

The full outage is far worse than the single-feature audit suggested. Losing the
whole C-family (C1-C14) costs **Rs 139,160,941** -- loss quadruples from Rs 44.1M
to Rs 183.3M. And the failure mode is perverse: the primary model blocks **24,633
payments instead of 2,759** at a **19.41% false-positive rate**, declining one in
five genuine customers -- while *recall rises* from 0.413 to 0.615. On a
fraud-caught dashboard an outage looks like the model improving.

| scenario | blocked | recall | FPR | rupee loss |
|---|---|---|---|---|
| counting service healthy | 2,759 | 0.413 | 0.95% | 44,124,593 |
| service down, no fallback | 24,633 | 0.615 | **19.41%** | **183,285,534** |
| service down, fallback engaged | 1,990 | 0.248 | 0.86% | 50,196,139 |

**The fallback recovers 96% of the outage cost** (Rs 139.2M -> Rs 6.1M). It is a
weaker detector -- recall 0.248 -- but it is honest about what it cannot see, and
holds FPR at 0.86%. Degrading quietly beats degrading loudly.

Routing rule: if any C-family feature is unavailable, score with the fallback
model **and its own calibrator and threshold**. Never score a partially-blank
vector with the primary.

## Industry context

| Source | Their number | Ours |
|---|---|---|
| Adyen 2025 fraud report | **5% of fraud-linked identities = 58% of fraudulent value** | Independent confirmation that fraud loss is value-concentrated, which is why this project reports value-recall alongside count-recall |
| PayPal | false-positive rate **below 5%** | **0.94%** — five times more conservative about declining genuine customers |
| Stripe Radar | fraud reduced **32%**, trained on 70 trillion data points | **24.8%** savings, trained on 413k transactions and 77 features |

Not a like-for-like comparison — different metrics, incomparable data scale — but
it places the results in a real operating range rather than a vacuum.

## Benchmark context, and the fusion test

Combinatorial Fusion Analysis reports **ROC-AUC 0.9405** on IEEE-CIS
(arXiv:2606.10393, 2026) by fusing Random Forest, XGBoost and LightGBM. We
reproduced that method on our own constrained setup (`fusion.py`) to find out
whether our gap was the feature restriction or the single-model choice:

| model | ROC-AUC | PR-AUC | card P@10 | rupee loss | vs LightGBM |
|---|---|---|---|---|---|
| LightGBM | 0.8870 | 0.5095 | 0.919 | 44,124,593 | — |
| XGBoost | 0.8864 | 0.4990 | 0.921 | 44,761,738 | −637,146 |
| RandomForest | 0.8857 | 0.4891 | 0.919 | 43,291,465 | **+833,128** |
| **fusion (average score)** | **0.8988** | **0.5258** | **0.929** | **43,236,713** | **+887,879** |
| fusion (diversity-weighted) | 0.8976 | 0.5250 | 0.929 | 43,470,134 | +654,459 |

Three results worth stating:

1. **Fusion is the largest single improvement in this project — Rs 887,879**, bigger
   than the per-instance threshold gain (Rs 384,567). Simple score averaging beat the
   paper's diversity weighting.
2. **RandomForest has the worst ROC-AUC of the three single models and the best rupee
   loss.** The model that ranks worst loses least money — this project's central claim,
   arrived at accidentally.
3. **It quantifies the SOTA gap.** 0.9405 − 0.8870 = 0.0535, of which fusion recovers
   0.0118. So roughly **78% of the gap is the online-feature restriction**, not the
   single-model choice. Adopting fusion costs ~3× inference (about 21 ms p50), which
   still fits a 50 ms budget but consumes most of it.

This project reaches **0.8870** single-model, **0.8988** fused. The difference is explained, not excused:
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

## The dataset

**IEEE-CIS Fraud Detection** (Vesta Corporation, Kaggle 2019).

| | |
|---|---|
| Transactions | 590,540 card-not-present |
| Fraud rate | 3.5% (20,663 positives) |
| Columns | 434 present, **77 used** |
| Currency | USD, converted at a fixed rate so every figure is in rupees |
| Split | temporal 70 / 10 / 20, sorted by `TransactionDT`, never shuffled |

**Why 357 columns are dropped.** `data.audit_leakage()` classifies every column
and the exclusions are reported rather than implied:

| Class | Count | Reason |
|---|---|---|
| online | 77 | available at checkout, when the decision is actually made |
| `V_*` | 339 | Vesta's engineered features -- not computable inline |
| `D_*` | 15 | day-offsets that grow with `TransactionDT`, so they drift |

That restriction costs accuracy. `restriction_cost.py` measures how much, rather
than leaving it as a claim.

---

## Reproducing

```bash
pip install -r requirements.txt
python fetch_data.py     # needs ~/.kaggle/kaggle.json + accepted competition rules
python canonical.py      # the headline number on its own
python canonical.py --seeds   # ...and its spread across 3 seeds
python three_way.py      # allow / review / block breakdown on the ensemble
python run_all.py        # every stage, in order (--quick skips the slow ones)
```

Every stage writes JSON to `artifacts/`. All models train with `seed=42`.

`requirements.txt` is grouped by purpose: the core block is all you need for the
shipped system. The graph experiments pull in PyTorch and PyTorch Geometric, and
they only reproduce a **negative** result -- skip them unless you want to check it.

---

## Running it on your own payments

Four steps, about five minutes. Full detail in **[ADAPTING.md](ADAPTING.md)**.

### 1. Give your CSV four column names

Rename these four. Everything else is discovered automatically -- there is no
feature list to edit, and you can have as many extra columns as you like.

| Your column becomes | What it holds |
|---|---|
| `isFraud` | 0 or 1 |
| `TransactionDT` | any increasing number -- epoch seconds, a date, even row order. Only the ORDER matters. |
| `TransactionAmt` | the amount |
| `TransactionID` | a unique id |

So a file like this works:

```csv
TransactionID,TransactionDT,TransactionAmt,isFraud,channel,device,city
1,1704067200,2499.00,0,app,android,Chennai
2,1704067340,89999.00,1,web,windows,Delhi
3,1704067410,349.50,0,pos,terminal,Mumbai
```

Put it in `data/` and point `config.py` at it:

```python
TRANSACTION_CSV = DATA_DIR / "your_payments.csv"
```

### 2. Screen it before you train. Do not skip this.

```bash
python check_data.py data/your_payments.csv
```

**This is the step people skip and regret.** The pipeline will happily train on
anything you give it -- including a column that does not exist yet at the moment
you have to decide. That kind of column does not cause an error. It just makes
your model look excellent.

Screening nine public datasets found four with one. A status field where
"Declined" is 59.6% fraud (you only know a payment was declined *after*
something declined it). A card-state field with three values at exactly
**100.00%** fraud. A reason field filled in only for frauds. One of those
datasets scores **AUC 1.0000** straight out of the box, and thousands of people
have downloaded it.

`check_data.py` catches all four and exits non-zero on anything critical. It
passes clean on IEEE-CIS. Every flag is a question, and the question is always
the same: **at the instant you must decide, do you actually have that value?**

### 3. Put in your costs

This is the part that decides every rupee figure the repo produces.

```python
# config.py
USD_TO_INR           = 1.0      # 1.0 if your amounts are ALREADY in your currency
MERCHANT_MARGIN_RATE = 0.25     # your gross margin
CUSTOMER_LTV_INR     = 2500.0   # what losing a good customer actually costs you
CHARGEBACK_FEE_INR   = 1200.0   # your acquirer's penalty per chargeback
```

**`USD_TO_INR = 1.0` matters more than it looks.** IEEE-CIS is in dollars, so
the default multiplies every amount by 88. Leave it at 88 on rupee data and
every cost figure is inflated 88x.

Get the other three from your finance team, not from us -- ours are assumptions,
and they are the weakest numbers in this repo. `python sensitivity.py` shows how
far the conclusions move when they are wrong by +/-30%.

### 4. Run it

```bash
python canonical.py --seeds   # the headline, across 3 seeds
python three_way.py           # allow / review / block breakdown
python run_all.py --quick     # everything except the slow stages
```

---

**What transfers, and what does not.** The decision layer transfers unchanged --
`chow_band.py`, `three_way.py`, `loss_types.py`, `conformal.py`,
`sensitivity.py`, `calibrate.py`. It works on top of any scorer. The detector's
weights do not transfer; retrain them on your traffic. Swap in a better model
than ours and every result here still holds, and gets more valuable.

**One thing to check yourself.** `data.audit_leakage()` excludes IEEE-CIS's
`V_*` and `D_*` columns by prefix. On your data those prefixes match nothing, so
*every* column is treated as usable. That is a safe default only because step 2
exists -- run it.

**Reproducibility note, because this was wrong.** The headline figure was
originally produced by an inline terminal command and never committed -- no
script in the repo generated it, and seven analysis scripts were missing from
`run_all.py`, while the README claimed it ran "every result". Both fixed:
`canonical.py` now produces the headline and every script is wired in.

**Which numbers are seed-verified and which are not.** Verified across three
seeds: **the headline itself** (`canonical.py --seeds` -- recall sd 0.3%, PR-AUC
sd 0.0005, rupee loss sd Rs 238,972), the ensemble gain, and every finding that
was RETRACTED. Single
seed: the Rs 139M outage cost, the fallback's 96% recovery, the robustness
ablations, and all four loss-type policies. Those are single-run and are labelled
as such rather than being quietly presented alongside verified ones.

---

## The three-way decision, measured

`three_way.py`. Two-way ALLOW/BLOCK is the headline policy; adding a human lane
is what the analyst budget actually buys. Both computed on the same ensemble
scores, so nothing is mixed across runs.

**Test set: 118,109 transactions, 4,064 frauds.**

| Lane | Transactions | Frauds in it | Share of all fraud |
|---|---|---|---|
| ALLOW | 81,716 (69.2%) | 615 | 15.1% |
| **REVIEW** | 35,916 (30.4%) | **3,009** | **74.0%** |
| BLOCK | 477 (0.4%) | 440 | 10.8% |

**Fraud reached (review + block): 3,449 of 4,064 = 84.9%**, against 43.6% for
blocking alone. Auto-block precision is **92.2%** -- 37 genuine customers touched
in the whole test set.

The review lane is 2.4x denser in fraud than the traffic it holds. Sweeping the
review cost moves the whole structure:

| Review cost | Fraud reached | Reviews/day | Auto-block precision | Realised loss |
|---|---|---|---|---|
| Rs 25 | **96.8%** | 1,861 | 100.0% | Rs 2.76M |
| Rs 150 | 84.9% | 855 | 92.2% | Rs 9.61M |
| Rs 400 | 74.8% | 458 | 88.2% | Rs 16.2M |
| Rs 1,000 | 60.9% | 210 | 80.7% | Rs 24.4M |

**Do not quote the Rs 9.61M bare.** It charges reviews at cost and assumes
analysts resolve every case correctly. `chow_capacity.py` models 80% and 90%
accuracy -- at 80%, a large queue starts *losing* money.

---

## Nine datasets, one pipeline

`transfer.py`, `try_baf.py`, `indian_all.py` -> `artifacts/transfer.json`.

Same code every time: temporal split, identical LightGBM parameters, Platt
calibration, the same cost rule with LTV set to 50% of each dataset's own median
amount. Only the loader changes. Nothing is tuned per dataset.

**Real data**

| Dataset | Rows | Fraud % | AUC | ALLOW | REVIEW | BLOCK | Caught | Conc |
|---|---|---|---|---|---|---|---|---|
| IEEE-CIS *(shipped)* | 590,540 | 3.44% | 0.8870 | 84,992 | 32,619 | 497 | 3,329/4,064 | 2.6x |
| ULB credit card | 284,807 | 0.13% | 0.9375 | 56,458 | 454 | 50 | 60/75 | 18.4x |
| Elliptic (Bitcoin) | 203,769 | 9.76% | **0.9623** | -- | -- | -- | -- | -- |
| NeurIPS BAF | 1,000,000 | 1.40% | 0.8892 | 159,689 | 40,311 | 0 | 2,191/2,799 | 3.9x |

**Simulated -- scores inflated, generator rules are learnable**

| Dataset | Rows | AUC | Caught | Conc |
|---|---|---|---|---|
| Sparkov | 1,296,675 | 0.9888 | 1,336/1,538 | 30.4x |
| PaySim | 6,362,620 | 0.9258 | 3,238/4,254 | 251.8x |
| Indian financial fraud | 245,091 | 0.8513 | 1,367/1,681 | 2.2x |

**No signal -- negative controls, kept in deliberately**

| Dataset | Rows | AUC | Conc |
|---|---|---|---|
| Indian bank transactions | 200,000 | 0.4896 | **1.0x** |
| Indian banking 2019-24 | 217,428 | 0.5252 | 1.3x |

**Concentration is the column that matters** -- how much denser the review lane
is in fraud than the traffic it holds. At **1.0x** the model found nothing and
the frauds in that lane are there by volume alone, which is why the bottom two
rows appear to "catch" 1,346 and 115 frauds while detecting precisely zero.
**Never sum the caught column across tiers.**

Elliptic shows dashes because its features are fully anonymised with no amount
column -- the cost layer needs a value to work with, so only ranking quality is
reported (PR-AUC 0.7774, 15.3x lift). NeurIPS BAF is account-opening fraud, not
transaction fraud, and is labelled as such.

**Four popular datasets were rejected before testing.** Three (`rupakroy`,
`jainilcoder`, `chitwanmanchanda`) are byte-identical 186 MB re-uploads of
PaySim; one (`whenamancodes`) is ULB again; and
`nelgiriyewithana/credit-card-fraud-2023` has been rebalanced to **exactly
50.0000%** fraud, which makes every rate computed on it meaningless. A fifth,
`skullagos/upi-transactions-2024`, was dropped after catching **0 of 96**.

---

## Is there a real Indian dataset? No.

`indian_all.py`, `clean_indian.py`. Five tested; all fail, in two distinct ways.

| Dataset | AUC | Failure mode |
|---|---|---|
| marusagar/bank-transaction-fraud | 0.4896 | pure noise |
| skullagos/upi-transactions-2024 | 0.4951 | noise |
| belbino/indian-banking-2019-24 | 0.5252 | noise |
| saurabhbadole/CIBIL | 0.9999 | **target leakage** -- grade derived from its own features |
| jatinkhandelwal/indian-financial-fraud | 0.9754 -> 0.8513 | **3 leaks**; usable after cleaning |

Proof the first is generated: the fraud rate is 5.04% inside *every* category
(spread <= 1.1%); mean amount Rs 49,552 genuine vs Rs 49,278 fraud; and
**Customer_ID is 100% unique -- 200,000 customers across 200,000 transactions,
so no customer ever transacts twice.** Real payment streams do not look like
that.

Why none exist: RBI's 2018 localisation directive and the DPDP Act 2023 make
releasing real payment data legally hazardous, NPCI holds UPI data with no
research mandate, and fraud features are defensive secrets.

**The nuance that matters.** NeurIPS BAF is *also* synthetic -- Feedzai generated
it with a CTGAN from real bank applications under differential privacy. It scores
0.889. The difference is that BAF was generated *from* real data while the Indian
ones were generated *from nothing*. **Synthetic is not the problem;
synthetic-from-nothing is.**

One real Indian dataset does exist but is not fraud: L&T vehicle loan default
(`try_lt.py`, AUC **0.6489**, in line with published benchmarks). It is loan
default, so the defensible claim is *"no real Indian **transaction-fraud**
dataset"* -- not the broader version.

---

## The GNN, retrained to the literature's own specification

`gnn_tune.py`, `gnn_stage3.py` -> `artifacts/gnn_tuned.json`.

The original GraphSAGE run stopped at 60 epochs **with calibration PR-AUC still
climbing**, so "the GNN loses" was not yet a claim about the method. This answers
the objection properly.

**Learning rate** (200 epochs, patience 30): 0.05 -> 0.1235 (diverged);
0.01 -> 0.3903 (still improving at the cap); **0.005 -> 0.3945** (converged at
epoch 185, and the literature default); 0.001 -> 0.3467.

**The heterophily hypothesis, stated in advance and refuted.**
[arXiv:2312.06441](https://arxiv.org/html/2312.06441v3) argues mean aggregation
is a low-pass filter that erases signal under heterophily, and that max
aggregation or a self-residual should recover it:

| Architecture | calib PR-AUC | vs baseline |
|---|---|---|
| **mean** | **0.3945** | -- |
| mean + residual | 0.3866 | -0.0079 |
| max + residual | 0.3628 | -0.0317 |

**Both prescribed fixes made it worse.**

| Arm | PR-AUC | ROC-AUC | Frauds caught |
|---|---|---|---|
| GraphSAGE untuned (60ep, K=5) | 0.2831 | 0.8174 | 554/4,064 |
| GraphSAGE tuned (K=5) | 0.3112 | 0.8255 | 703/4,064 |
| GraphSAGE tuned (K=25) | 0.3210 | 0.8308 | 717/4,064 |
| **ENSEMBLE (shipped)** | **0.5220** | **0.8980** | **1,771/4,064** |

Tuning was worth doing -- **+163 frauds, +29%** -- and the original run genuinely
was undertrained. But **every gain came from training longer.** Neither
architectural prescription helped, and the K=25 fan-out flips sign between slices
(+0.0098 on test, -0.0016 on calibration), so it is noise rather than an effect.
The original conclusion stands on a converged model: **1,054 fewer frauds than
the ensemble.** Single seed (42), and labelled as such.

---

## Limitations

Stated plainly, because they affect how the numbers should be read.

- **IEEE-CIS is US card data, and the gap is a THREAT MODEL gap, not just a
  currency one.** RBI mandates additional-factor authentication on domestic card
  payments, so the specific fraud this dataset contains -- stolen-card
  card-not-present checkout -- is structurally suppressed in India. Domestic
  Indian fraud skews to social engineering: OTP phishing, UPI collect-request
  scams, refund abuse, account takeover. None of that is in this data.
  Where the work does transfer: cross-border card payments, where Indian AFA
  does not apply, and where Razorpay's own Vulcan announcement highlights
  international card fraud. The decision layer -- calibration, per-instance
  cost thresholds, alert-budget evaluation -- is threat-model agnostic and
  applies to any scorer. The detector is not.
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
