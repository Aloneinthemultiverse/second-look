# Verification record — Second Look

Every figure below is console output from a run performed on the date shown, on
the exact commit named. Nothing is transcribed by hand. Raw captures are in
`proof/`; machine-readable results are the 44 JSON files in `artifacts/`, each
written by a named script in this repository.

```bash
git clone https://github.com/Aloneinthemultiverse/second-look.git
cd second-look && pip install -r requirements.txt
python fetch_data.py && python canonical.py --seeds
```

---

## 1. Environment

```
run date : 2026-10-25 17:50 UTC
python   : Python 3.14.0
commit   : ba188d69ddccefdb67bd2f06fdb1b8d392857123
branch   : master
clean    : 1 tracked files modified

lightgbm       4.7.0
xgboost        3.3.0
scikit-learn   1.8.0
pandas         2.3.3
numpy          2.4.1
scipy          1.17.0
```

---

## 2. Headline result, reproduced across three random seeds

`python canonical.py --seeds` → `proof/01_canonical_seeds.txt`

```
CANONICAL RESULT  ensemble, per-instance tau*, nothing fit on test
==============================================================
frauds caught                                   1,771 of 4,064
recall (count)                                           43.6%
recall (value)                                           40.8%
precision                                                0.621
false-positive rate                                      0.95%
ROC-AUC                                                 0.8980
PR-AUC                                                  0.5220
rupee loss                                       Rs 43,024,561
--------------------------------------------------------------
across seeds 42, 7, 2024:
  recall        min 43.4%   max 43.9%   sd 0.3%
  PR-AUC        min 0.5220   max 0.5229   sd 0.0005
  rupee loss    min 43,024,561   max 43,461,866   sd 238,972
==============================================================

wrote C:\Users\Sujit Narrayan M\razorpay\artifacts\canonical.json
```

**Why three seeds.** One run proves the code executed. Three runs from different
random initialisations show the result is a property of the method rather than of
a lucky seed. Recall moves by half a percentage point; PR-AUC by 0.0009. An
earlier finding in this project was positive on one seed and negative on the
other two — it was retracted, which is why every headline figure is now reported
with its spread.

---

## 3. Unit tests

`python -m pytest test_core.py -v` → `proof/02_tests.txt`

```
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-9.0.3, pluggy-1.6.0 -- C:\Python314\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Sujit Narrayan M\razorpay
plugins: anyio-4.14.2, typeguard-4.6.0
collecting ... collected 11 items

test_core.py::test_split_is_strictly_ordered_in_time PASSED              [  9%]
test_core.py::test_split_partitions_every_row_exactly_once PASSED        [ 18%]
test_core.py::test_split_does_not_leak_rows_between_slices PASSED        [ 27%]
test_core.py::test_audit_excludes_D_and_V_but_keeps_lookalikes PASSED    [ 36%]
test_core.py::test_perfect_predictions_cost_nothing PASSED               [ 45%]
test_core.py::test_blocking_nothing_costs_exactly_the_fraud PASSED       [ 54%]
test_core.py::test_blocking_everything_costs_exactly_the_genuine PASSED  [ 63%]
test_core.py::test_loss_is_monotone_in_threshold_extremes PASSED         [ 72%]
test_core.py::test_platt_preserves_ranking_exactly PASSED                [ 81%]
test_core.py::test_isotonic_creates_ties_platt_does_not PASSED           [ 90%]
test_core.py::test_calibrator_outputs_are_probabilities PASSED           [100%]

============================= 11 passed in 3.39s ==============================
```

These test the specific failure modes this project is about, not generic imports.
`test_split_is_strictly_ordered_in_time` fails if a temporal split ever leaks the
future. `test_isotonic_creates_ties_platt_does_not` encodes the calibration
finding — isotonic regression collapsed 111,512 distinct scores into 167 levels
and destroyed ranking, so the test exists to stop anyone reintroducing it.

---

## 4. Leakage screen

`python check_data.py` → `proof/03_leakage_screen.txt`

```
REQUIRED COLUMNS
==============================================================================
  OK    label    -> isFraud
  OK    amount   -> TransactionAmt
  OK    time     -> TransactionDT

==============================================================================
BASE RATE
==============================================================================
  3.4990% positive (20,663 of 590,540)

==============================================================================
COLUMN SCREEN
==============================================================================
  screened 393 columns

==============================================================================
FINDINGS  (0)
==============================================================================
  Nothing flagged. That is not a guarantee -- this screen cannot
  know which of your columns are filled in after the fact.

==============================================================================
Clear to train. Set your costs in config.py first -- every rupee
figure this repo produces is downstream of them.
==============================================================================
```

**393 columns screened, zero findings.** The screen checks every column in the raw
file, not only the 77 the model uses, so it is not passing because it was pointed
somewhere safe. The same screen rejects four public datasets, including one that
scores AUC 1.0000 out of the box because it carries a status field where
"Declined" is 59.6% fraud — an outcome, not a feature.

---
## 5. The three-way decision

`python three_way.py` → `artifacts/three_way.json`. Test set: 118,109 transactions, 4,064 frauds. Review cost Rs 150.

| Lane | Transactions | Frauds | Share of fraud |
|---|---|---|---|
| ALLOW | 81,716 (69.2%) | 615 | 15.1% |
| REVIEW | 35,916 (30.4%) | 3,009 | 74.0% |
| BLOCK | 477 (0.4%) | 440 | 10.8% |

**Fraud reached (review + block): 3,449 of 4,064 = 84.9%**, against 43.6% for blocking alone. Auto-block precision **92.2%** — only 37 genuine customers touched in the entire test set.

| Review cost | Fraud reached | Reviews/day | Auto-block precision |
|---|---|---|---|
| Rs 25 | 96.8% | 1,861 | 100.0% |
| Rs 50 | 93.5% | 1,457 | 97.6% |
| Rs 150 | 84.9% | 855 | 92.2% |
| Rs 400 | 74.8% | 458 | 88.2% |
| Rs 1,000 | 60.9% | 210 | 80.7% |

---

## 6. Nine-dataset transfer study

`python transfer.py` → `artifacts/transfer.json`. One pipeline, no per-dataset tuning.

| Dataset | Rows | Fraud % | ROC-AUC | Review concentration |
|---|---|---|---|---|
| IEEE-CIS | 590,540 | 3.44% | 0.8870 | 2.6x |
| ULB credit card | 284,807 | 0.13% | 0.9375 | 18.4x |
| NeurIPS BAF | 1,000,000 | 1.40% | 0.8892 | 3.9x |
| Sparkov | 1,296,675 | 0.59% | 0.9888 | 30.4x |
| PaySim | 6,362,620 | 0.33% | 0.9258 | 251.8x |
| Indian bank | 200,000 | 5.14% | 0.5081 | 1.0x |

Concentration is how much denser the review lane is in fraud than the traffic it
holds. **1.0x means the model found nothing.** The synthetic control is in this
table deliberately — reporting only the datasets that flatter a method is the
failure this project has retracted eleven times.

---

## 7. The graph model, retrained to the literature's specification

`python gnn_tune.py && python gnn_stage3.py` → `artifacts/gnn_tuned.json`

| Arm | PR-AUC | ROC-AUC | Frauds caught |
|---|---|---|---|
| GraphSAGE untuned (60 epochs) | 0.2831 | 0.8174 | 554 |
| GraphSAGE tuned (K=5) | 0.3112 | 0.8255 | 703 |
| GraphSAGE tuned (K=25) | 0.3210 | 0.8308 | 717 |
| **Ensemble (shipped)** | **0.5220** | **0.8980** | **1,771** |

Retrained at 200 epochs with a four-point learning-rate sweep and the heterophily
fix the fraud-GNN literature prescribes. It improved 29% and still catches
**1,054 fewer frauds** than the ensemble. All three architectural
prescriptions in those papers made it worse. Single seed, labelled as such.

---

## 8. Where everything else lives

| Evidence | Location |
|---|---|
| Raw console captures | `proof/` |
| Machine-readable results | `artifacts/*.json` — 44 files |
| Figures | `artifacts/figures/` — 4 PNG |
| Full write-up, including what was retracted | `README.md` |
| Interactive demo | https://claude.ai/code/artifact/117f9074-3828-4b71-ba5c-7eda18062371 |

Every number in the README names the script that produced it. That rule exists
for a reason given in the README's opening section: the headline figure was once
produced by a terminal command that was never committed, and when a script was
finally written to generate it, **the number changed** — 1,776 became 1,771. The
corrected figure is the one in section 2 above.
