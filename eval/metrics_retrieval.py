"""Retrieval metrics against evidence quotes (no LLM involved).

Each item has one or more facts, each with one or more quotes (the evidence and its
alternatives). A retrieved chunk covers a fact when it contains any of the fact's quotes.
An item with several facts (multi-hop) is only fully recalled when every fact is covered,
and a chunk covering a fact already covered higher up earns nothing, so duplicated
passages can't inflate the scores.
"""

import math

from eval.golden import normalize


def covered(chunks: list[str], facts: list[list[str]]) -> list[set[int]]:
    """For each ranked chunk, the indexes of the facts it covers."""
    fs = [[normalize(q) for q in quotes] for quotes in facts]
    out = []
    for chunk in chunks:
        text = normalize(chunk)
        out.append({i for i, quotes in enumerate(fs) if any(q in text for q in quotes)})
    return out


def score(chunks: list[str], facts: list[list[str]], ks=(1, 3, 5, 10, 20)) -> dict[str, float]:
    """Recall@k, Hit@k, MRR and nDCG@10 for one ranked list.

    `facts`: one list of alternative quotes per fact."""
    per_chunk = covered(chunks, facts)
    seen: set[int] = set()
    gains = []  # quotes first covered at each rank
    for found in per_chunk:
        new = found - seen
        gains.append(len(new))
        seen |= new
    out: dict[str, float] = {}
    for k in ks:
        top = set().union(*per_chunk[:k]) if per_chunk[:k] else set()
        out[f"recall@{k}"] = len(top) / len(facts)
        out[f"hit@{k}"] = float(bool(top))
    first = next((rank for rank, g in enumerate(gains, start=1) if g), None)
    out["mrr"] = 1 / first if first else 0.0
    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains[:10], start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(facts), 10) + 1))
    out["ndcg@10"] = dcg / ideal
    out["first_relevant_rank"] = first or 0
    return out
