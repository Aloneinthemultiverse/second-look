"""Five-minute live demo. Runs in about ten seconds.

    python demo.py

Training the ensemble takes four minutes, which is four minutes of a progress
bar in front of an audience. This loads the calibrated scores that
`three_way.py` already saved and does only the decision arithmetic, which is
instant. The scores are the real ones -- `artifacts/ensemble_test_scores.npy`,
written by the same run that produced the headline -- so nothing here is
staged. If that file is missing, the script says so rather than inventing
anything.

Five acts, in the order they make sense to a person who has not seen the
project:

  1. the data, and why a random split would be a lie
  2. one transaction, three possible actions, and what each costs
  3. the same arithmetic over the whole test set
  4. why chasing recall bankrupts the merchant
  5. the human lane, and what it buys
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

W = 74
ART = Path(__file__).parent / "artifacts"


def rule(ch="="):
    print(ch * W)


def head(n, title):
    print()
    rule()
    print(f"  {n}. {title}")
    rule()


def rupee(x):
    return f"Rs {x:,.0f}"


def pause(seconds=0.6):
    """Let a line land before the next one. Demos read better with a beat."""
    sys.stdout.flush()
    time.sleep(seconds)


def main() -> None:
    import config

    need = [ART / "ensemble_test_scores.npy", ART / "test_arrays.npz"]
    missing = [p for p in need if not p.exists()]
    if missing:
        print("Cannot run the fast demo -- these are missing:")
        for m in missing:
            print(f"  {m}")
        print("\nRun `python three_way.py` once to generate them (about 5 min),")
        print("or run `python canonical.py` for the headline the slow way.")
        raise SystemExit(1)

    p = np.load(ART / "ensemble_test_scores.npy")
    d = np.load(ART / "test_arrays.npz")
    y, amt = d["y"].astype(int), d["amount"].astype(float)
    n, F = len(y), int(y.sum())

    print()
    rule()
    print("  SECOND LOOK -- cost-optimal fraud triage".center(W))
    print("  every payment gets its own threshold".center(W))
    rule()

    # ---- 1. the data ----------------------------------------------------
    head(1, "The data")
    print(f"  held-out transactions   {n:>12,}")
    print(f"  of which fraud          {F:>12,}   ({y.mean():.2%})")
    print(f"  median amount           {rupee(np.median(amt)):>12}")
    print()
    print("  Split by TIME, never randomly. A random split trains on")
    print("  transactions that happened AFTER the ones it is tested on.")
    print("  Every score comes out higher and every one of them is a lie.")
    pause()

    # ---- 2. one transaction ---------------------------------------------
    head(2, "Every payment gets its own threshold")
    c_fp = lambda a: a * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
    c_fn = lambda a: a + config.CHARGEBACK_FEE_INR
    review = 150.0

    print("  Blocking a genuine customer costs margin plus their future value.")
    print("  Allowing a fraud costs the amount plus a chargeback fee. Those are")
    print("  different numbers on every transaction, so the cutoff is too:")
    print()
    print(f"  {'amount':>11}{'cost if wrongly':>18}{'cost if wrongly':>18}"
          f"{'cutoff':>10}")
    print(f"  {'':>11}{'BLOCKED':>18}{'ALLOWED':>18}{'tau*(x)':>10}")
    print("  " + "-" * (W - 4))
    for a in (200.0, 2_000.0, 6_028.0, 20_000.0, 100_000.0):
        tau = c_fp(a) / (c_fp(a) + c_fn(a))
        note = "  <- median" if a == 6_028.0 else ""
        print(f"  {rupee(a):>11}{c_fp(a):>18,.0f}{c_fn(a):>18,.0f}"
              f"{tau:>10.3f}{note}")
    print("  " + "-" * (W - 4))
    print()
    print("  On a Rs 200 payment the model has to be 65% sure before blocking")
    print("  is worth it -- below that you lose more by turning away someone")
    print("  real. On Rs 100,000 it only needs 21%. Same model, same scores;")
    print("  the economics decide. That is Elkan (2001).")
    print()
    print("  Most systems pick ONE cutoff and apply it to all of this.")
    pause()

    # ---- 3. the whole test set ------------------------------------------
    head(3, "The same arithmetic, over all of it")
    cfp, cfn = c_fp(amt), c_fn(amt)
    blocked = p >= cfp / (cfp + cfn)
    tp = int((blocked & (y == 1)).sum())
    fp = int((blocked & (y == 0)).sum())
    print(f"  frauds caught           {tp:>12,}  of {F:,}   ({tp / F:.1%})")
    print(f"  precision               {tp / max(blocked.sum(), 1):>12.3f}"
          "   of everything blocked, this share was fraud")
    print(f"  genuine customers hit   {fp:>12,}   ({fp / (y == 0).sum():.2%} "
          "of honest traffic)")
    pause()

    # ---- 4. the recall trap ---------------------------------------------
    head(4, "Why not just catch more?")
    print(f"  {'recall target':>15}{'must block':>14}{'% of traffic':>15}"
          f"{'rupee loss':>18}")
    print("  " + "-" * (W - 4))
    order = np.argsort(-p)
    do_nothing = float((amt[y == 1] + config.CHARGEBACK_FEE_INR).sum())
    for target in (0.436, 0.65, 0.85, 0.95):
        k = int(np.searchsorted(np.cumsum(y[order]), target * F)) + 1
        sel = np.zeros(n, bool)
        sel[order[:k]] = True
        loss = float((amt[(~sel) & (y == 1)] + config.CHARGEBACK_FEE_INR).sum()
                     + (amt[sel & (y == 0)] * config.MERCHANT_MARGIN_RATE
                        + config.CUSTOMER_LTV_INR).sum())
        mark = "  <- ours" if abs(target - 0.436) < 1e-6 else ""
        print(f"  {target:>14.0%}{k:>14,}{k / n:>15.1%}{loss:>18,.0f}{mark}")
    print("  " + "-" * (W - 4))
    print(f"  doing nothing at all{do_nothing:>27,.0f}")
    print()
    print("  Chasing 95% recall costs multiples of having no fraud system.")
    print("  43.6% is not a limitation. It is where the money stops.")
    pause()

    # ---- 5. the human lane ----------------------------------------------
    head(5, "So add a third option: ask a person")
    cost = np.stack([p * cfn, np.full_like(p, review), (1 - p) * cfp])
    act = cost.argmin(axis=0)
    print(f"  {'lane':>10}{'transactions':>16}{'share':>9}{'frauds':>10}"
          f"{'of all fraud':>15}")
    print("  " + "-" * (W - 4))
    for i, nm in enumerate(("ALLOW", "REVIEW", "BLOCK")):
        m = act == i
        f = int((m & (y == 1)).sum())
        print(f"  {nm:>10}{int(m.sum()):>16,}{m.mean():>9.1%}{f:>10,}"
              f"{f / F:>15.1%}")
    print("  " + "-" * (W - 4))
    rev = act == 1
    reached = int(((act > 0) & (y == 1)).sum())
    conc = (int((rev & (y == 1)).sum()) / F) / max(rev.mean(), 1e-9)
    blk = act == 2
    print(f"  fraud reached (review+block)  {reached:>6,} of {F:,}"
          f"   ({reached / F:.1%})   vs {tp / F:.1%} by blocking alone")
    print(f"  review lane is {conc:.1f}x denser in fraud than the traffic it holds")
    if blk.sum():
        print(f"  auto-block precision          "
              f"{int((blk & (y == 1)).sum()) / blk.sum():>6.1%}"
              f"   ({int((blk & (y == 0)).sum())} genuine customers touched)")
    print()
    rule()
    print("  The machine handles the easy cases. A human handles the hard ones.")
    print("  The machine decides which is which -- using money, not a guess.")
    rule()
    print()
    print("  Full write-up and what was retracted:  README.md")
    print("  Verification record from a real run:   RESULTS.md")
    print("  Interactive version:")
    print("  https://claude.ai/code/artifact/117f9074-3828-4b71-ba5c-7eda18062371")
    print()


def model_quality() -> None:
    """The predictive half: does the model discriminate, and do its numbers mean
    anything?

    The rest of the demo takes P(fraud) as given and reasons about what to do
    with it. That is the decision layer. This act is about the model that
    produces the probability in the first place -- separation, ranking, and
    above all CALIBRATION, because the cost arithmetic is only valid if a
    predicted 0.30 really does default 30% of the time.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = np.load(ART / "ensemble_test_scores.npy")
    d = np.load(ART / "test_arrays.npz")
    y = d["y"].astype(int)
    n, F = len(y), int(y.sum())

    print()
    rule()
    print("  THE MODEL -- how good is the prediction itself?".center(W))
    rule()

    print()
    print("  SEPARATION   does it tell the two classes apart?")
    print("  " + "-" * (W - 4))
    print(f"  mean score, genuine     {p[y == 0].mean():>10.4f}")
    print(f"  mean score, fraud       {p[y == 1].mean():>10.4f}"
          f"   {p[y == 1].mean() / p[y == 0].mean():.0f}x higher")
    print(f"  median score, genuine   {np.median(p[y == 0]):>10.4f}")
    print(f"  median score, fraud     {np.median(p[y == 1]):>10.4f}")

    print()
    print("  RANKING      if you sort by score, where does the fraud land?")
    print("  " + "-" * (W - 4))
    print(f"  {'decile':>10}{'transactions':>15}{'frauds':>10}{'fraud rate':>13}"
          f"{'lift':>9}")
    order = np.argsort(-p)
    base = y.mean()
    for i in range(10):
        lo, hi = i * n // 10, (i + 1) * n // 10
        seg = y[order[lo:hi]]
        r = seg.mean()
        print(f"  {'top ' + str(i + 1) if i == 0 else i + 1:>10}"
              f"{len(seg):>15,}{int(seg.sum()):>10,}{r:>13.2%}"
              f"{r / base:>8.1f}x")
    print("  " + "-" * (W - 4))
    top = y[order[:n // 10]]
    print(f"  The top 10% of scores holds {top.sum() / F:.1%} of all fraud.")

    print()
    print("  CALIBRATION  when it says 30%, does 30% of it turn out to be fraud?")
    print("  " + "-" * (W - 4))
    print(f"  {'predicted band':>18}{'count':>11}{'predicted':>13}{'actual':>11}"
          f"{'gap':>9}")
    edges = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.01]
    worst = 0.0
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 50:
            continue
        pred, act = p[m].mean(), y[m].mean()
        worst = max(worst, abs(pred - act))
        print(f"  {f'{lo:.2f} - {min(hi, 1.0):.2f}':>18}{int(m.sum()):>11,}"
              f"{pred:>13.3f}{act:>11.3f}{act - pred:>+9.3f}")
    print("  " + "-" * (W - 4))
    print(f"  Largest gap between predicted and actual: {worst:.3f}")
    print()
    print("  This is the act that licenses everything else. Multiplying a score")
    print("  by a rupee cost is only meaningful if the score is a probability.")
    print("  Platt scaling, fitted on a slice held out in time -- not isotonic,")
    print("  which collapsed 111,512 distinct scores into 167 levels and")
    print("  destroyed the ranking above.")

    print()
    print("  AGGREGATE")
    print("  " + "-" * (W - 4))
    print(f"  ROC-AUC                 {roc_auc_score(y, p):>10.4f}")
    print(f"  PR-AUC                  {average_precision_score(y, p):>10.4f}")
    print(f"  PR-AUC if guessing      {base:>10.4f}"
          f"   so {average_precision_score(y, p) / base:.0f}x better than chance")
    print()
    for k in (10, 50, 200):
        sel = y[order[:k]]
        print(f"  precision @ top {k:<4}    {sel.mean():>10.1%}"
              f"   an analyst opening {k} cases a day")
    print()
    rule()
    print("  Prediction gets you the probability. The next part decides what")
    print("  to do with it -- run `python demo.py` for that.")
    rule()
    print()


def interactive() -> None:
    """Type an amount and a risk, get the decision and the arithmetic behind it.

    This is the same three-way rule the shipped system uses -- `chow_band.py`,
    unchanged. Nothing is looked up or faked: the costs are recomputed from
    config.py for whatever amount you type.
    """
    import config

    print()
    rule()
    print("  TRY IT -- type an amount and a fraud risk".center(W))
    rule()
    print(f"  Costs from config.py: margin {config.MERCHANT_MARGIN_RATE:.0%}, "
          f"lost customer {rupee(config.CUSTOMER_LTV_INR)},")
    print(f"  chargeback fee {rupee(config.CHARGEBACK_FEE_INR)}.")
    print()
    print("  One more number is yours to set: what an analyst review costs.")
    print("  It decides how much work the humans get. Cheap reviews and the")
    print("  system almost never auto-blocks; expensive ones and it has to.")
    try:
        raw = input("\n  analyst review cost in Rs [150] > ").strip().replace(",", "")
        review = float(raw) if raw else 150.0
    except (EOFError, KeyboardInterrupt, ValueError):
        review = 150.0
    print(f"  using {rupee(review)} per review.")
    print("\n  Blank amount, or Ctrl+C, to quit.")

    while True:
        print()
        try:
            raw = input("  amount in Rs            > ").strip().replace(",", "")
            if not raw:
                print("\n  done.\n")
                return
            amount = float(raw)
            if amount <= 0:
                print("  amount must be positive.")
                continue
            raw = input("  fraud risk 0-100 %      > ").strip().rstrip("%")
            prob = float(raw) / 100.0
            if not 0.0 <= prob <= 1.0:
                print("  risk must be between 0 and 100.")
                continue
        except (EOFError, KeyboardInterrupt):
            print("\n  done.\n")
            return
        except ValueError:
            print("  that wasn't a number -- try again.")
            continue

        c_fp = amount * config.MERCHANT_MARGIN_RATE + config.CUSTOMER_LTV_INR
        c_fn = amount + config.CHARGEBACK_FEE_INR
        tau = c_fp / (c_fp + c_fn)
        costs = {"ALLOW": prob * c_fn, "REVIEW": review,
                 "BLOCK": (1 - prob) * c_fp}
        best = min(costs, key=costs.get)

        print()
        print(f"  {'action':<9}{'expected cost':>16}   reasoning")
        print("  " + "-" * (W - 4))
        reasons = {
            "ALLOW": f"{prob:.0%} chance of losing {rupee(c_fn)}",
            "REVIEW": "an analyst's time, whatever the amount",
            "BLOCK": f"{1-prob:.0%} chance of losing {rupee(c_fp)}",
        }
        for k in ("ALLOW", "REVIEW", "BLOCK"):
            mark = " <--" if k == best else "    "
            print(f"  {k:<9}{costs[k]:>16,.0f}{mark} {reasons[k]}")
        print("  " + "-" * (W - 4))
        print(f"  DECISION: {best}   (cheapest at {rupee(costs[best])})")
        print()
        print(f"  Two-way cutoff for this amount: tau* = {tau:.3f}")
        if prob >= tau:
            print(f"  Risk {prob:.0%} is above it, so a block-only system would "
                  "decline this.")
        else:
            print(f"  Risk {prob:.0%} is below it, so a block-only system would "
                  "let it through.")
        if best == "REVIEW":
            print("  The third lane is cheaper than either -- that is what the")
            print("  human queue is for.")


if __name__ == "__main__":
    if "--try" in sys.argv or "-i" in sys.argv:
        interactive()
    elif "--model" in sys.argv or "-m" in sys.argv:
        model_quality()
    else:
        main()
        print("  The model behind it:            python demo.py --model")
        print("  Try your own numbers:           python demo.py --try")
        print()
