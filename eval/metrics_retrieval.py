"""Retrieval metrics against evidence quotes (no LLM involved).

Each item has one or more facts, each with one or more passages (the evidence and its
alternatives): a URL and a verbatim quote. A retrieved chunk covers a fact when it comes
from a passage's page and contains its quote. The page check matters: boilerplate such as
"Feature state: `Kubernetes v1.36 [stable]`" appears on many pages, and a quote alone would
count all of them. Pages are chunking-independent, so labels still survive re-chunking.
An item with several facts (multi-hop) is only fully recalled when every fact is covered,
and a chunk covering a fact already covered higher up earns nothing, so duplicated
passages can't inflate the scores.
"""

import math
from urllib.parse import urlsplit

from eval.golden import normalize

# Paths shared by many records (one per glossary term / feature gate): the fragment
# names the record, so it is part of the page identity.
KEYED_PATHS = (
    "/docs/reference/glossary/",
    "/docs/reference/command-line-tools-reference/feature-gates/",
)

Chunk = tuple[str, str]  # (url, text)
Fact = list[tuple[str, str]]  # alternative (url, quote) passages


def page(url: str) -> tuple[str, str]:
    """URL -> (path with query, record fragment or ""), ignoring scheme and host."""
    parts = urlsplit(url)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    return path, parts.fragment if parts.path in KEYED_PATHS else ""


def covered(chunks: list[Chunk], facts: list[Fact]) -> list[set[int]]:
    """For each ranked chunk, the indexes of the facts it covers."""
    fs = [[(page(url), normalize(quote)) for url, quote in fact] for fact in facts]
    out = []
    for url, text in chunks:
        where, text = page(url), normalize(text)
        out.append(
            {i for i, fact in enumerate(fs) if any(p == where and q in text for p, q in fact)}
        )
    return out


def score(chunks: list[Chunk], facts: list[Fact], ks=(1, 3, 5, 10, 20)) -> dict[str, float]:
    """Recall@k, Hit@k, MRR and nDCG@10 for one ranked list of (url, text) chunks.

    `facts`: one list of alternative (url, quote) passages per fact."""
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
