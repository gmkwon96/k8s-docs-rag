"""Rerank retrieved chunks with a Voyage cross-encoder.

rerank-3 / rerank-3-lite are the Voyage rerankers whose first 200M tokens per account are
free (rerank-2.5 has none), so only those are allowed. Billing counts
`query tokens x documents + document tokens`; every call goes through the Voyage token
ledger with that estimate first, and scores are cached by (model, query, texts) so reruns
cost nothing.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import voyageai

from rag.embedders import ESTIMATE_MARGIN
from rag.retrieve import Hit
from rag.settings import get_settings
from rag.usage import USAGE_DIR, Ledger

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "rerank" / "cache.jsonl"
FREE_TIER_MODELS = ("rerank-3", "rerank-3-lite")


@dataclass
class VoyageReranker:
    model: str = "rerank-3"
    cache_path: Path = CACHE_PATH

    def __post_init__(self):
        if self.model not in FREE_TIER_MODELS:
            raise ValueError(f"{self.model} has no free tokens; use one of {FREE_TIER_MODELS}")
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                row = json.loads(line)
                self._cache[row["key"]] = row["scores"]

    @cached_property
    def ledger(self) -> Ledger:
        return Ledger(USAGE_DIR / "voyage.jsonl", get_settings().voyage_token_budget)

    @cached_property
    def client(self) -> voyageai.Client:
        return voyageai.Client(api_key=get_settings().voyage_api_key, max_retries=5)

    def key(self, query: str, texts: list[str]) -> str:
        blob = json.dumps([self.model, query, texts])
        return hashlib.sha256(blob.encode()).hexdigest()

    def estimate_tokens(self, query: str, texts: list[str]) -> int:
        from ingest.chunk import count_tokens

        q = count_tokens(query)
        return int((q * len(texts) + sum(count_tokens(t) for t in texts)) * ESTIMATE_MARGIN)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        key = self.key(query, texts)
        if key not in self._cache:
            self.ledger.check(self.estimate_tokens(query, texts))
            result = self.client.rerank(query, texts, model=self.model, truncation=False)
            self.ledger.record(self.model, result.total_tokens, "rerank")
            by_index = {r.index: r.relevance_score for r in result.results}
            scores = [by_index[i] for i in range(len(texts))]
            self._cache[key] = scores
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "scores": scores}) + "\n")
        return self._cache[key]

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        scores = self.scores(query, [h.text for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))[:k]
        return [replace(hits[i], score=scores[i]) for i in order]
