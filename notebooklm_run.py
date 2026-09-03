"""Create a NotebookLM notebook from the Track 02 brief and interrogate it.

The MCP server is registered but only loads on a Claude Code restart, so this
drives the same client API directly. Sources are the verbatim Track 02 brief and
the project README, so answers are grounded in what was actually written rather
than in the model's recollection.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

from notebooklm import NotebookLMClient

TITLE = "Razorpay Buildathon - Track 02 (Second Look)"

BRIEF = """RAZORPAY AI BUILDATHON 2026 - TRACK 02: AI RISK MANAGER

Headline: Stop the merchant losing money to fraud, returns and chargebacks.

The task: Build a working detector, verifier or auto-responder for one class of
loss, with measured precision and recall on a held-out test set.

Why now: AI-enabled fraud is hitting Indian BFSI while returns and chargebacks
quietly eat margin. This track surfaces the risk and ML minded builders the
others miss.

Example directions:
- Chargeback evidence responder
- Return-risk scorer
- Fraud-spike detector
- Abuse-ring sentinel

THE BAR: Honest metrics including false-positive cost. Strictly defense-only:
anything offense-capable is disqualified.

JUDGING CRITERIA (apply across all tracks):
- Problem taste: did you pick something that actually matters
- Build quality: does it run, is it structured, would you trust it
- AI judgment: the right tool in the right place, and where you chose NOT to
  use one
- Failure recovery: what broke, and what you did about it

SUBMISSION: public GitHub repo, 5-minute pitch video, and a written answer to
"what broke, and how you got out". Applications close 5 September 2026.
"""

QUESTIONS = [
    "What exactly does Track 02 require a submission to contain? List every "
    "explicit requirement as a checklist.",
    "Does the Second Look project satisfy every stated requirement of the "
    "Track 02 bar? Go requirement by requirement and cite the evidence.",
    "The brief lists four example directions. Second Look is none of them - it "
    "is a detector plus a cost-based decision layer. Is that a problem for the "
    "'problem taste' criterion? Argue both sides.",
    "Judging includes 'AI judgment: the right tool in the right place, and "
    "where you chose NOT to use one'. How well does Second Look answer BOTH "
    "halves of that criterion?",
    "What is the single weakest part of this submission, and what would a "
    "sceptical Razorpay engineer attack first?",
]


async def main() -> None:
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")

    client = await NotebookLMClient.from_storage()
    async with client:
        existing = [n for n in await client.notebooks.list() if n.title == TITLE]
        if existing:
            nb = existing[0]
            print(f"reusing notebook {nb.id}")
        else:
            print("creating notebook ...")
            nb = await client.notebooks.create(TITLE)
            await client.sources.add_text(nb.id, "Track 02 brief (verbatim)", BRIEF)
            await client.sources.add_text(nb.id, "Second Look README",
                                          readme[:100_000])
            print("waiting for sources to index ...")
            await asyncio.sleep(20)

        for i, q in enumerate(QUESTIONS, 1):
            print("\n" + "=" * 78)
            print(f"Q{i}. {q}")
            print("=" * 78)
            try:
                r = await client.chat.ask(nb.id, q)
                print(r.answer)
                if getattr(r, "sources", None):
                    print(f"\n[grounded in: {', '.join(s.title for s in r.sources)}]")
            except Exception as e:  # noqa: BLE001
                print(f"[failed: {type(e).__name__}: {e}]")
            await asyncio.sleep(3)

        print(f"\nnotebook id: {nb.id}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
