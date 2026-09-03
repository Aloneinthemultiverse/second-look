"""Research GNN fraud detection via NotebookLM, grounded in actual papers.

Adds recent arXiv work as sources, plus this project's README so the answers are
about OUR data and constraints rather than generic advice, then asks whether a
graph approach is worth building here.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

from notebooklm import NotebookLMClient

TITLE = "GNN fraud detection - applicability to Second Look"

PAPERS = [
    "https://arxiv.org/abs/2411.05815",   # GNNs for Financial Fraud Detection: A Review
    "https://arxiv.org/abs/2504.08183",   # Heterogeneous GNN with Graph Attention
    "https://arxiv.org/abs/2503.22681",   # detectGNN
    "https://arxiv.org/abs/2504.02275",   # RGCN in the fraud workflow
    "https://arxiv.org/abs/2605.12782",   # Graph-based fraud, calibrated risk scoring
]

CONTEXT = """OUR SYSTEM (for applicability questions)

Dataset: IEEE-CIS Fraud Detection. 590,540 card-not-present transactions,
3.5% fraud, split TEMPORALLY (70% train / 10% calibration / 20% test).

Features actually used: 77 online features computable inside a checkout latency
budget. We measured single-row inference at p50 7.1 ms / p95 11.1 ms. We
deliberately EXCLUDED 339 V_* offline aggregates because they cannot be computed
in time at a real checkout, and 15 D_* day-offset columns because they leak time
under a temporal split.

Available graph-forming identifiers in the data: card1 (card identifier), card2,
DeviceInfo, P_emaildomain and R_emaildomain, addr1/addr2 (billing region).
There is NO explicit merchant id and NO explicit user id.

Current model: ensemble of LightGBM + XGBoost + RandomForest, Platt calibrated,
with per-transaction cost-based thresholds. ROC-AUC 0.8976, PR-AUC 0.5224.

Hard constraints: inference must fit a checkout latency budget; two days of
development time remain; single developer.
"""

QUESTIONS = [
    "Across these papers, what graph structure is actually used for credit card "
    "fraud - what are the nodes, what are the edges, and how is the graph "
    "constructed from raw transaction tables?",
    "What concrete performance gains do these papers report for GNNs over "
    "gradient-boosted trees on tabular fraud data? Give the numbers and the "
    "baselines they were compared against.",
    "What is the INFERENCE LATENCY story for GNNs in fraud detection? Our "
    "budget is under 50ms per transaction at checkout. Do these papers address "
    "real-time serving, neighbour sampling cost, or do they assume batch "
    "scoring?",
    "Given our system description - 77 online features, temporal split, card1 / "
    "DeviceInfo / email domain as the only graph-forming identifiers, no "
    "merchant or user id, 7ms current inference, and two days of development "
    "time - is building a GNN realistic and worth it? Be specific about what we "
    "would have to give up.",
    "If we could only take ONE idea from the GNN literature and implement it as "
    "a feature for an existing gradient-boosted model rather than as a full "
    "graph network, what single graph-derived feature would give the most value "
    "for the least engineering?",
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
            nb = await client.notebooks.create(TITLE)
            print(f"created notebook {nb.id}")
            for url in PAPERS:
                try:
                    await client.sources.add_url(nb.id, url)
                    print(f"  + {url}")
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {url} -> {type(e).__name__}: {e}")
                await asyncio.sleep(2)
            await client.sources.add_text(nb.id, "Second Look - system context",
                                          CONTEXT)
            await client.sources.add_text(nb.id, "Second Look README",
                                          readme[:80_000])
            print("indexing ...")
            await asyncio.sleep(45)

        for i, q in enumerate(QUESTIONS, 1):
            print("\n" + "=" * 78)
            print(f"Q{i}. {q}")
            print("=" * 78)
            try:
                r = await client.chat.ask(nb.id, q)
                print(r.answer)
            except Exception as e:  # noqa: BLE001
                print(f"[failed: {type(e).__name__}: {e}]")
            await asyncio.sleep(3)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
