import math

import pytest

from eval.metrics_retrieval import page, score
from eval.run_retrieval import summarize

BASE = "https://kubernetes.io"


def chunk(text, path="/docs/p/", anchor="a"):
    return (f"{BASE}{path}#{anchor}", text)


def fact(*quotes, path="/docs/p/"):
    return [(f"{path}#x", q) for q in quotes]


def test_single_fact_found_at_rank_2():
    s = score(
        [chunk("nothing"), chunk("the default is\n30 seconds."), chunk("other")],
        [fact("default is 30 seconds")],
    )
    assert s["hit@1"] == 0 and s["hit@3"] == 1
    assert s["recall@1"] == 0 and s["recall@3"] == 1
    assert s["mrr"] == 0.5
    assert s["ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert s["first_relevant_rank"] == 2


def test_not_found():
    s = score([chunk("a"), chunk("b")], [fact("missing quote")])
    assert (s["recall@20"], s["hit@20"], s["mrr"], s["ndcg@10"]) == (0, 0, 0, 0)
    assert s["first_relevant_rank"] == 0


def test_quote_on_another_page_does_not_count():
    boilerplate = "Feature state: `Kubernetes v1.36 [stable]` enabled by default"
    s = score([chunk(boilerplate, path="/docs/other/")], [fact(boilerplate, path="/docs/p/")])
    assert s["hit@20"] == 0
    s = score([chunk(boilerplate, path="/docs/p/", anchor="any-section")], [fact(boilerplate)])
    assert s["hit@1"] == 1  # same page, any section: chunk boundaries don't matter


def test_versioned_hosts_and_record_fragments():
    gates = "/docs/reference/command-line-tools-reference/feature-gates/"
    assert page(f"https://v1-35.docs.kubernetes.io{gates}#A") == page(f"{gates}#A")
    assert page(f"{gates}#A") != page(f"{gates}#B")
    assert page("/docs/p/#one") == page("/docs/p/#two")


def test_multihop_partial_recall():
    s = score(
        [chunk("quote one here"), chunk("x"), chunk("quote two here")],
        [fact("quote one"), fact("quote two"), fact("quote three")],
    )
    assert s["recall@1"] == pytest.approx(1 / 3)
    assert s["recall@3"] == pytest.approx(2 / 3)
    assert s["hit@1"] == 1


def test_duplicate_chunks_earn_nothing_twice():
    facts = [fact("quote one"), fact("quote two")]
    once = score([chunk("quote one"), chunk("filler")], facts)
    twice = score([chunk("quote one"), chunk("quote one")], facts)
    assert twice["ndcg@10"] == once["ndcg@10"]
    assert twice["recall@3"] == 0.5


def test_any_alternative_covers_a_fact():
    alt = [("/docs/p/#x", "the original quote"), ("/docs/q/#y", "stated another way")]
    s = score([chunk("x"), chunk("stated another way", path="/docs/q/")], [alt])
    assert s["recall@3"] == 1 and s["mrr"] == 0.5


def test_perfect_ranking_has_ndcg_1():
    s = score([chunk("quote one"), chunk("quote two")], [fact("quote one"), fact("quote two")])
    assert s["ndcg@10"] == pytest.approx(1.0)


def test_summary_by_category():
    rows = [
        {"category": "fact", "metrics": score([chunk("q1 text")], [fact("q1 text")])},
        {"category": "fact", "metrics": score([chunk("x")], [fact("q1 text")])},
        {"category": "howto", "metrics": score([chunk("x"), chunk("q1 text")], [fact("q1 text")])},
    ]
    s = summarize(rows)
    assert s["n"] == 3
    assert s["overall"]["hit@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["by_category"]["fact"]["n"] == 2
    assert s["by_category"]["howto"]["mrr"] == 0.5
