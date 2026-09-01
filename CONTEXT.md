# Context transfer — Second Look

Paste this into a new session or another model to resume work with full context.
Everything below is verified against artifacts in this repo, not recalled.

---

## Situation

I am a student applying to the **Razorpay AI Buildathon 2026**, an internship
hiring programme (₹75,000/month, 6 or 12 months, in-person Bangalore).
**Applications close 5 September 2026. Today is 1 September. Four days left.**

The submission needs three things:
1. A **public GitHub repo**
2. A **5-minute pitch video**
3. A written answer to **"what broke, and how you got out"**

Judging criteria as published:
- **Problem taste** — did you pick something that actually matters
- **Build quality** — does it run, is it structured, would you trust it
- **AI judgment** — the right tool in the right place, *and where you chose NOT to use one*
- **Failure recovery** — what broke, and what you did about it

**Track 02 — AI Risk Manager**, verbatim brief:
> "Stop the merchant losing money to fraud, returns and chargebacks. Build a
> working detector, verifier or auto-responder for one class of loss, with
> measured precision and recall on a held-out test set. The bar: Honest metrics
> including false-positive cost. Strictly defense-only: anything offense-capable
> is disqualified."

Note Track 02 is the only track that does **not** say "build an agent" — it says
detector/verifier/auto-responder. Its rationale says it exists to surface
"risk and ML minded builders the others miss."

---

## The project: "Second Look"

Cost-optimal fraud triage for card-not-present payments.

**One line:** fraud scoring is a tabular problem with no unstructured input, so
the detector is gradient boosting, not an LLM. The threshold is chosen by
minimising expected rupee loss, not F1. The LLM writes the analyst's brief — it
never produces a number and never decides whether money moves.

**The problem framing:** for CNP fraud the *merchant* absorbs the loss. Blocking
suspicious payments creates the harder problem — a false decline costs the sale,
the margin, and the customer's future business. So the question is not "is this
fraud" but "where do you draw the line when both mistakes cost money and the
amounts differ per transaction."

---

## Data

**IEEE-CIS Fraud Detection** (Vesta, Kaggle) — 590,540 real transactions,
434 columns after merging transaction+identity, **3.4990% fraud** (20,663 positives).

**Temporal split on `TransactionDT`, never random** (70/10/20):

| slice | rows | fraud |
|---|---|---|
| train | 413,378 | 3.5169% |
| calibration | 59,053 | 3.4901% |
| test | 118,109 | 3.4409% (42 days) |

**Feature audit:** 77 online features used. Excluded: 339 `V_*` (offline
aggregates, not computable in a checkout latency budget) and 15 `D_*` (day-offset
timedeltas that grow with `TransactionDT`, so a temporal split lets a model learn
"large D ⇒ later period" — this is temporal drift, NOT chargeback leakage; an
earlier version of the repo wrongly called it leakage).

**Second dataset for replication:** ULB creditcardfraud, 284,807 rows, 0.1317%
fraud. Same pipeline unchanged → ROC-AUC 0.9694, PR-AUC 0.7302.

---

## Verified results (all on the temporally held-out test set)

| Metric | Value |
|---|---|
| PR-AUC | 0.5091 ± 0.0028 (3 seeds) |
| ROC-AUC | 0.8870 |
| **Card Precision@10/day** | **0.919** (26.7× lift) |
| Calibration error (ECE) | 0.0067 → 0.0032 after Platt |
| Single-row inference | p50 7.1 ms, p95 11.1 ms (model scoring only) |
| Recall (count) | 0.441 |
| Recall (value) | 0.392 |
| False-positive rate | 0.94–1.04% |
| Savings vs no fraud system | **23.98%** honest |

**The key reframe:** PR-AUC 0.51 looks mediocre but measures the whole stream —
work no analyst team will ever do. At a realistic alert budget (Dal Pozzolo),
Card Precision@10 = 0.919.

**The recall trap** (important, counter-intuitive):

| target recall | block rate | rupee loss |
|---|---|---|
| 44% (optimal) | 2.51% | 43,760,271 |
| 65% | 7.77% | 66,640,920 |
| 95% | 61.80% | 420,615,744 |
| *do nothing* | 0% | *58,551,022* |

At 95% recall you block 62% of traffic and lose **7× more than having no fraud
system at all**. 44% is where the money stops, not a limitation.

**Chow + capacity finding (novel — no paper states it):** at 80% analyst accuracy,
reviewing 500 cases/day is *worse than no fraud system* (₹60.9M vs ₹58.5M).
Analyst accuracy caps useful review capacity. Value per review falls from
₹14,221 at k=10 to ₹1,575 at k=500.

---

## Shipped configuration

**IN — verified across 3 seeds**
- Ensemble LightGBM + XGBoost + RandomForest, **simple score averaging**
  (+₹1,178,231, range 800,543–1,415,634, positive 3/3). Plain averaging beat the
  paper's diversity weighting.
- Platt calibration on the temporal calibration slice
- Per-instance threshold τ*(x) = C_FP/(C_FP+C_FN) (+₹384,567, closed form)
- Three-way ALLOW / REVIEW / BLOCK bounded by analyst capacity
- TreeSHAP → LLM analyst brief on review-band cases (legibility only)

**OUT — tested and rejected, with numbers**
- Unsupervised IsolationForest blend (not robust; worsens value-recall 3/3 seeds)
- Cost-proportionate training weights (value-recall 0.392 → 0.377)
- Segmented high-amount model (0.392 → 0.363)
- Isotonic calibration (collapses 111,512 score levels → 167; −0.021 PR-AUC)
- STEP_UP fourth action (no defensible friction cost)

**Ensemble costs ~3× inference (~21ms of a 50ms budget).** A latency-constrained
deployment should ship single LightGBM and forgo the ₹1.18M.

---

## Papers implemented

| Paper | Contribution |
|---|---|
| Elkan (2001), *Foundations of Cost-Sensitive Learning* | τ* = C_FP/(C_FP+C_FN); calibrate-then-threshold validated |
| Bahnsen et al., instance-dependent cost-sensitive learning | Per-instance thresholds |
| Chow (1970), classification with reject option | Review band derived from costs |
| Dal Pozzolo et al. (TNNLS 2017) | Alert budget, Card Precision@k |
| Zadrozny et al. (2003) | Cost-proportionate weighting (tested, rejected) |
| arXiv:2607.19266 (2026) | Precedent for bounded LLM investigation on uncertain cases |
| arXiv:2606.10393 (2026) | Ensemble fusion (adopted); their SOTA is ROC-AUC 0.9405 |
| arXiv:2607.13078 (2026) | Deployment-evidence checklist — we meet 4/5 |

---

## THE SEVEN RETRACTIONS (the core of the submission)

Findings that did **not** survive verification. This is the first section of the
README, deliberately.

1. **"Optimising F1 costs ₹964,857"** — across seeds the gap ranged ₹0–710,488,
   sd > mean. Traced to threshold-grid resolution. Cut.
2. **Savings inflated by ₹713,146** — I was sweeping the decision threshold on
   the test set and reporting on that same test set. Marking my own homework.
   Honest savings: 25.20% → **23.98%**.
3. **"Threshold varies 7× across merchants"** — true on IEEE-CIS, does not
   generalise; on ULB two of four profiles block nothing. Base-rate conditional.
4. **Investigation playbook adds no detection signal** — score alone 0.6735 vs
   score+playbook 0.6401. (First version of that experiment was itself broken —
   median imputation deleted the "device never seen" signal; fixing it lifted
   playbook-only 0.4485 → 0.5621 and the conclusion held.)
5. **"Model is weak on high-value fraud"** — measured against the wrong policy.
   Under the shipped per-instance threshold the caught/missed gap is ₹1,124, not
   ₹3,837 (71% smaller).
6. **"RandomForest ranks worst but loses least money"** — sd ₹760,562 vs mean
   ₹742,277, sign flips. Cut.
7. **"PayPal-style anomaly layer saves ₹334,171"** — +103,635 / −283,176 /
   −23,569 across seeds, mean negative. Cut.

**Pattern:** 1, 6 and 7 share a failure mode — a single run produced an
attractive number and the spread swamped it. **Every headline is now verified
across seeds before it is written down.**

---

## Industry context

| Source | Their number | Ours |
|---|---|---|
| Adyen 2025 | 5% of fraud-linked identities = 58% of fraudulent value | Confirms value-concentration; why we report value-recall |
| PayPal | FPR below 5% | **0.94%** — 5× more conservative |
| Stripe Radar | 32% fraud reduction, 70T data points | 24.8% savings, 413k transactions |

Razorpay's own Vulcan (4B payments, 3T data points) is far better at detection.
**We do not claim otherwise.** Positioning: this is the *decision layer* that sits
on top of any scorer — swap in a better score and every result still holds.

---

## Repo state

**27 commits. 11 tests passing. 4 figures. ~15 JSON artifacts.**

Key files: `data.py` (temporal split, feature audit) · `features.py` ·
`model.py` (detector + sweep) · `calibrate.py` (Platt vs isotonic, with the
measured reason) · `cost_sensitive.py` (global vs per-instance vs cost-weighted)
· `alert_budget.py` (Precision@k) · `chow_band.py` + `chow_capacity.py` ·
`fusion.py` + `verify_fusion.py` · `anomaly_layer.py` + `verify_anomaly.py` ·
`playbook.py` (5 lookups + entity graph with information-value filter) ·
`investigator.py` (TreeSHAP → brief) · `review_band_metrics.py` · `audit.py` ·
`verify.py` (seeds, splits, latency) · `second_dataset.py` · `high_value.py` ·
`plots.py` · `test_core.py` · `run_all.py`

Docs: `README.md` (authoritative) · `ARCH.md` (day-1 plan, marked superseded) ·
`VIDEO_SCRIPT.md` (timed 5-min script)

**Notable engineering detail:** the entity graph filters links by information
value (−log₂ P(value)). gmail.com is 46% of rows = 1.37 bits and is dropped, so
"these two cases share an email domain" can never become false evidence. The
filter is principled, not a hardcoded blocklist.

---

## ⚠️ WHAT IS NOT DONE

1. **No GitHub remote.** The repo is local only. The form needs a public URL.
   `gh repo create second-look --public --source=. --push`
2. **No video recorded.** Script exists at `VIDEO_SCRIPT.md`.

Nothing else is blocking. **These two are the entire remaining risk.**

---

## Working principles established (keep these)

- **Verify across seeds before believing any number.** Seven findings died this way.
- **Never select a threshold on the test set.** That bug cost ₹713,146 of fake savings.
- **Report negative results.** Four rejected approaches are documented with numbers.
- **State limitations up front** — US data with an Indian cost model; invented
  cost parameters (sensitivity-tested ±30%); latency excludes feature retrieval;
  no hyperparameter tuning.
- **Do not claim to beat Razorpay's systems.** Position as the decision layer.
- **The LLM never emits a number and never triggers an action.** Enforced in code.

---

## If you are continuing this work

The model is past the bar; more modelling has near-zero marginal value with four
days left. The highest-value remaining actions are, in order: push to GitHub,
record the video, and make sure the video frames seven retractions as **rigour,
not weakness** (lead with Card Precision@10 = 0.919, reach the retractions at
~3:00, close on limitations).
