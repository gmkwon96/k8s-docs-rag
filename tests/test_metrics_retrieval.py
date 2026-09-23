import math

import pytest

from eval.metrics_retrieval import score
from eval.run_retrieval import summarize


def test_single_quote_found_at_rank_2():
    s = score(["nothing", "the default is\n30 seconds.", "other"], [["default is 30 seconds"]])
    assert s["hit@1"] == 0 and s["hit@3"] == 1
    assert s["recall@1"] == 0 and s["recall@3"] == 1
    assert s["mrr"] == 0.5
    assert s["ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert s["first_relevant_rank"] == 2


def test_not_found():
    s = score(["a", "b"], [["missing quote"]])
    assert (s["recall@20"], s["hit@20"], s["mrr"], s["ndcg@10"], s["first_relevant_rank"]) == (
        0,
        0,
        0,
        0,
        0,
    )


def test_multihop_partial_recall():
    s = score(
        ["quote one here", "x", "quote two here"], [["quote one"], ["quote two"], ["quote three"]]
    )
    assert s["recall@1"] == pytest.approx(1 / 3)
    assert s["recall@3"] == pytest.approx(2 / 3)
    assert s["hit@1"] == 1


def test_duplicate_chunks_earn_nothing_twice():
    once = score(["quote one", "filler"], [["quote one"], ["quote two"]])
    twice = score(["quote one", "quote one"], [["quote one"], ["quote two"]])
    assert twice["ndcg@10"] == once["ndcg@10"]
    assert twice["recall@3"] == 0.5


def test_any_alternative_covers_a_fact():
    s = score(["x", "stated another way"], [["the original quote", "stated another way"]])
    assert s["recall@3"] == 1 and s["mrr"] == 0.5


def test_perfect_ranking_has_ndcg_1():
    s = score(["quote one", "quote two"], [["quote one"], ["quote two"]])
    assert s["ndcg@10"] == pytest.approx(1.0)


def test_summary_by_category():
    rows = [
        {"category": "fact", "metrics": score(["q1"], [["q1"]])},
        {"category": "fact", "metrics": score(["x"], [["q1"]])},
        {"category": "howto", "metrics": score(["x", "q1"], [["q1"]])},
    ]
    s = summarize(rows)
    assert s["n"] == 3
    assert s["overall"]["hit@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["by_category"]["fact"]["n"] == 2
    assert s["by_category"]["howto"]["mrr"] == 0.5
