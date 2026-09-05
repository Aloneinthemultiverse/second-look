# Running this on your own data

The detector is trained on IEEE-CIS, but the thing worth reusing is the
**decision layer** — the part that turns a score into an allow / review / block
action using your economics. That layer is dataset-agnostic. This is how to
point it at your own payments.

---

## What dataset this repo uses

**IEEE-CIS Fraud Detection** (Vesta Corporation, Kaggle competition, 2019).

- 590,540 card-not-present transactions
- 3.5% fraud rate
- 434 columns, of which we use **77**
- US card data, amounts in USD

`fetch_data.py` downloads it. You need a Kaggle API token at `~/.kaggle/kaggle.json`
and you must accept the competition rules first, or Kaggle returns 403.

**Why the other 357 columns are dropped** — `data.audit_leakage()` classifies
every column:

| Class | Count | Why excluded |
|---|---|---|
| online | 77 | usable at checkout time |
| `V_*` | 339 | Vesta's engineered features; unavailable at decision time |
| `D_*` | 15 | day-offsets that grow with `TransactionDT`, so they drift |

---

## The minimum your data needs

Only **four columns** are structurally required:

| Column | Meaning | Notes |
|---|---|---|
| `isFraud` | 0 or 1 | the label |
| `TransactionDT` | any increasing number | seconds, epoch, row order — only the ORDER matters |
| `TransactionAmt` | the amount | see currency note below |
| `TransactionID` | unique id | only needed if you also supply an identity file |

**Everything else is discovered automatically.** Add as many feature columns as
you like — `audit_leakage()` sweeps whatever it finds, and `features.build()`
casts string columns to pandas `category` so LightGBM splits on them natively.
There is no feature list to edit.

---

## Step 0 -- screen your data before anything else

```bash
python check_data.py path/to/your.csv
```

Everything below assumes your columns are honest. **They usually are not**, and
the failure is silent: a leaked column does not error, it just makes the model
look excellent. Of nine public datasets screened during this project, four had a
column that does not exist at decision time, and one scored a perfect AUC 1.0000
straight out of the box.

`check_data.py` exits non-zero if it finds anything critical. Do not skip it
because your data "came from the warehouse" -- warehouse tables are exactly where
post-settlement fields live.

---

## Three steps

### 1. Put your CSV where the code looks

```python
# config.py
TRANSACTION_CSV = DATA_DIR / "your_payments.csv"
IDENTITY_CSV    = DATA_DIR / "your_identity.csv"   # optional; delete the file if none
```

### 2. Fix the currency

IEEE-CIS is in USD, so `features.build()` multiplies every amount by
`USD_TO_INR`. **If your data is already in rupees this will inflate every amount
88×** and every cost figure with it.

```python
# config.py
USD_TO_INR = 1.0    # data already in the target currency
```

### 3. Put in your real costs

This is the part that matters, and the part where our numbers are weakest —
ours are assumptions, yours don't have to be.

```python
# config.py
MERCHANT_MARGIN_RATE = 0.25     # your gross margin
CUSTOMER_LTV_INR     = 2500.0   # what a lost customer actually costs you
CHARGEBACK_FEE_INR   = 1200.0   # your acquirer's penalty per chargeback
```

**Get these from your finance team, not from us.** Every rupee figure this repo
produces is downstream of these three numbers. Run `python sensitivity.py` to
see how much the conclusions move when they're wrong by ±30%.

Then:

```bash
pip install -r requirements.txt
python canonical.py --seeds      # the headline, across 3 seeds
python three_way.py              # allow / review / block breakdown
```

---

## What carries over, and what doesn't

**Carries over unchanged** — this is the reusable part:

- `chow_band.py` — the allow/review/block rule, derived from costs
- `three_way.py` — the breakdown at five review-cost levels
- `loss_types.py` — repoint the same layer at returns, chargebacks, UPI
- `conformal.py` — holds the block rate steady as data drifts
- `sensitivity.py` — how much your conclusions depend on your cost guesses
- `calibrate.py` — Platt scaling, with the measured reason it isn't isotonic

**Needs attention on a new dataset:**

- **The `V_*` / `D_*` exclusion rules are IEEE-CIS specific.** On your data those
  prefixes won't match, so *every* column is classified online. That's a safe
  default, but **you must check it yourself**: any column computed after the
  transaction settled (a chargeback flag, a manual review outcome, a refund
  field) will leak the answer and produce a beautiful, meaningless model. Edit
  `DRIFT_PREFIXES` / `LATENCY_PREFIXES` in `data.py`, or drop those columns
  before loading.
- `AUTO_BLOCK_LIMIT_INR` — the ceiling above which nothing is auto-blocked.
  Set it to your own risk appetite.
- Split fractions (70/10/20) assume enough data for a calibration slice to be
  meaningful. Under ~50,000 rows, widen `CALIB_FRAC`.

---

## The one thing not to change

**The split stays temporal.** `data.temporal_split()` sorts by `TransactionDT`
and asserts the three slices don't overlap in time:

```python
assert train["TransactionDT"].max() <= calib["TransactionDT"].min()
assert calib["TransactionDT"].max() <= test["TransactionDT"].min()
```

A random split trains on transactions that happened *after* the ones it's tested
on. Every score comes out higher and every one of them is a lie — in production
you only ever have the past. Those asserts are there on purpose. If one fires on
your data, your timestamps are wrong; don't delete the assert.

---

## Honest scope

You are getting a **decision layer** that works with any scorer, and a detector
trained on somebody else's payments. The detector's specific weights will not
transfer to your traffic — retrain it. The cost arithmetic, the calibration
discipline, the temporal split, and the three-way policy all transfer directly,
and those are the parts that took the longest to get right.
