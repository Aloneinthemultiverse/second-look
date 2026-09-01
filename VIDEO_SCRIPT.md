# 5-minute pitch script — Second Look

Screen recording. Talk over terminal output and the four figures in
`artifacts/figures/`. No slides needed. Read it roughly, don't memorise it —
sounding like a person beats sounding polished.

---

## 0:00 — 0:40 · The problem

> "When a stolen card buys four thousand rupees of goods online, the merchant
> loses the money, the goods, and a chargeback fee. Not the bank. The merchant.
>
> So block suspicious payments — except that declining a real customer costs the
> sale, the margin, and everything they'd have bought later. Across the industry,
> false declines cost more than fraud does.
>
> So the question isn't 'is this fraud'. It's 'where do you draw the line, when
> both mistakes cost money, and the amounts are different every time'."

---

## 0:40 — 1:25 · What it does, and the number that matters

*[show terminal: alert_budget.py output]*

> "I built a fraud triage system on the IEEE-CIS dataset — 590,000 real
> transactions, split by time, never randomly.
>
> The headline metric everyone reports is PR-AUC. Mine is 0.51, which looks
> mediocre. But PR-AUC measures performance over the entire stream — work no
> analyst team will ever do.
>
> Dal Pozzolo's work says measure at the analyst's actual budget. At ten alerts
> a day, **card precision is 0.919**. Nine of every ten cases a human actually
> opens is real fraud. Twenty-seven times better than chance.
>
> The aggregate metric was hiding a good model."

---

## 1:25 — 2:15 · The recall trap

*[show the recall/cost table]*

> "My recall is 44%. That sounds bad, so let me show you what buying more costs.
>
> At 65% recall I lose an extra 23 million rupees. At 95% recall I'd block 62%
> of all traffic and lose 420 million — **seven times worse than having no fraud
> system at all**.
>
> Forty-four percent isn't a limitation. It's where the money stops. Anyone
> quoting high recall on a fraud model without the block rate next to it is
> quoting a number that would bankrupt the merchant.
>
> And with the review layer at realistic capacity, we reach 85% of fraudulent
> cards. They're not missed — they go to a human instead of being auto-blocked."

---

## 2:15 — 3:00 · Where the AI is, and where it isn't

*[show figure 01_calibration.png]*

> "This is a tabular problem with no unstructured input, so the detector is
> gradient boosting. An LLM scoring 77 numeric features would be slower, more
> expensive and worse. I didn't use one, deliberately.
>
> The LLM writes the analyst's brief from SHAP attributions. It never produces a
> number and it never decides whether money moves — that's enforced in code, not
> just described.
>
> And I measured whether the investigation layer adds detection signal. **It
> doesn't.** Score alone ranks better. So I claim it for legibility, which is
> what an analyst under a budget actually needs, and nothing more."

---

## 3:00 — 4:10 · What broke  ← *the most important minute*

> "Two things broke, and both of them made my numbers look better, which is why
> neither would have been caught by looking at results.
>
> **First.** I calibrated with isotonic regression, then noticed PR-AUC had
> dropped 0.02 — for a transformation that shouldn't change ranking at all.
> Isotonic is a step function. It had collapsed 111,000 distinct scores into
> 167 levels, and tied scores have no order. I switched to Platt scaling, which
> is strictly monotonic. I only found it because I measured on both sides of the
> calibrator instead of assuming calibration is free.
>
> **Second, and worse.** I reported 25.2% savings. Then I realised I was choosing
> my decision threshold by sweeping it on the test set — and then reporting the
> result on that same test set. I was marking my own homework. Moving threshold
> selection to a separate slice cost **713,000 rupees of savings that were never
> real**. The honest number is 23.98%.
>
> That made me distrust every single-run number I had. So I re-ran everything
> across seeds and split points. **Seven findings didn't survive.** My headline
> result about F1 versus cost had a standard deviation larger than its mean. A
> PayPal-style anomaly layer that looked worth 334,000 rupees was positive in
> one seed out of three.
>
> They're all in the README, in a section called 'What I got wrong', which is
> the first thing you read."

---

## 4:10 — 4:45 · Limitations, said out loud

> "The data is US card data and I apply an Indian cost model to it — that's
> disclosed, not hidden. My cost parameters are assumptions, so I vary all of
> them by thirty percent. The latency number is model scoring only; feature
> retrieval would add to it. And the model is measurably weaker on expensive
> fraud — I tried four published fixes and three made it worse."

---

## 4:45 — 5:00 · Close

> "This isn't a better fraud detector than Razorpay's. It couldn't be — Vulcan
> sees four billion payments and I see a public dataset.
>
> It's the decision layer that sits on top of any scorer. Swap my model for a
> better one and every result here still holds, and gets more valuable.
>
> What I'd want you to take from it is that I spent as long trying to break my
> own findings as I spent producing them."

---

## Recording notes

- **Screen record the terminal.** Real output beats slides for this audience.
- **Figures worth showing:** `03_capacity_accuracy.png` (analysts at 80% accuracy
  reviewing 500 cases/day is worse than no system) is the most striking image.
- Don't rush the "what broke" minute. That's what they said they read first.
- Don't apologise for the 44% or the 0.51. Explain them — both have good answers.
- Under-run rather than over-run. 4:30 delivered calmly beats 5:30 rushed.
