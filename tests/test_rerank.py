from types import SimpleNamespace

import pytest

from rag.rerank import VoyageReranker
from rag.retrieve import Hit
from rag.usage import BudgetExceeded, Ledger


def hit(i, text):
    return Hit(i, 0.5, text, "1.37", f"c{i}", f"u{i}", "T", ["T"], [])


class FakeClient:
    def __init__(self, scores):
        self.scores, self.calls = scores, 0

    def rerank(self, query, documents, model, truncation):
        self.calls += 1
        results = [SimpleNamespace(index=i, relevance_score=s) for i, s in enumerate(self.scores)]
        return SimpleNamespace(results=results[::-1], total_tokens=123)


def reranker(tmp_path, scores, budget=10_000):
    r = VoyageReranker("rerank-3", cache_path=tmp_path / "cache.jsonl")
    r.__dict__["client"] = FakeClient(scores)
    r.__dict__["ledger"] = Ledger(tmp_path / "usage.jsonl", budget)
    return r


def test_only_free_tier_models():
    with pytest.raises(ValueError, match="no free tokens"):
        VoyageReranker("rerank-2.5")


def test_rerank_reorders_and_keeps_top_k(tmp_path):
    r = reranker(tmp_path, [0.1, 0.9, 0.5])
    out = r.rerank("q", [hit(1, "a"), hit(2, "b"), hit(3, "c")], k=2)
    assert [h.content_id for h in out] == [2, 3]
    assert out[0].score == 0.9
    assert r.ledger.total() == 123


def test_scores_are_cached_across_instances(tmp_path):
    r = reranker(tmp_path, [0.1, 0.9])
    r.rerank("q", [hit(1, "a"), hit(2, "b")], k=2)
    again = reranker(tmp_path, [0.0, 0.0])
    again.rerank("q", [hit(1, "a"), hit(2, "b")], k=2)
    assert again.client.calls == 0


def test_budget_is_checked_before_calling(tmp_path):
    r = reranker(tmp_path, [0.1], budget=1)
    with pytest.raises(BudgetExceeded):
        r.rerank("a long enough query", [hit(1, "a document with several tokens")], k=1)
    assert r.client.calls == 0
