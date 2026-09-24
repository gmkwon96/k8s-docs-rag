import pytest

from ingest.embed import store
from ingest.load import load
from rag.retrieve import vector_search
from tests.test_load import conn, content, occurrence, test_db_url  # noqa: F401  (fixtures)


def test_vector_search_filters_by_version_and_returns_that_versions_url(conn):  # noqa: F811
    load(
        conn,
        "heading-plain",
        [
            content("a", "alpha", ["1.36", "1.37"]),
            content("b", "beta", ["1.37"]),
            content("c", "gamma", ["1.36"]),
        ],
    )
    ids = dict(conn.execute("SELECT key, id FROM contents").fetchall())
    store(conn, "fake-2d", [(ids["a"], [1.0, 0.0]), (ids["b"], [0.9, 0.1]), (ids["c"], [0.0, 1.0])])

    hits = vector_search(conn, [1.0, 0.0], "1.36", k=5, model="fake-2d", dim=2)
    assert [h.text for h in hits] == ["alpha", "gamma"]  # "beta" is 1.37 only
    assert hits[0].url == "https://kubernetes.io/docs/a/" and hits[0].version == "1.36"
    assert hits[0].score > hits[1].score


def test_keyword_search_matches_any_word_and_filters_version(conn):  # noqa: F811
    from rag.retrieve import keyword_search

    load(
        conn,
        "heading-plain",
        [
            content("a", "Roll back a Deployment with kubectl rollout undo", ["1.37"]),
            content("b", "Deployment strategies and rolling updates", ["1.37"]),
            content("c", "Roll back a Deployment", ["1.36"]),
            content("d", "Unrelated text about volumes", ["1.37"]),
        ],
    )
    hits = keyword_search(conn, "How do I undo a Deployment rollout?", "1.37", k=5)
    assert [h.text for h in hits][:1] == ["Roll back a Deployment with kubectl rollout undo"]
    assert "Unrelated text about volumes" not in [h.text for h in hits]
    assert all(h.version == "1.37" for h in hits)


def test_rrf_rewards_agreement():
    from rag.retrieve import Hit, rrf

    def h(cid):
        return Hit(cid, 0.0, f"t{cid}", "1.37", f"c{cid}", f"u{cid}", "T", ["T"], [])

    fused = rrf([[h(1), h(2), h(3)], [h(3), h(4), h(1)]], k=4)
    assert [x.content_id for x in fused][:2] == [1, 3]  # in both lists
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_retrieve_reranks_candidates_down_to_k(conn):  # noqa: F811
    from rag.retrieve import RetrievalConfig, retrieve

    load(conn, "heading-plain", [content(key, f"text {key}", ["1.37"]) for key in "abc"])
    ids = dict(conn.execute("SELECT key, id FROM contents").fetchall())
    store(conn, "fake-2d", [(ids["a"], [1.0, 0.0]), (ids["b"], [0.9, 0.1]), (ids["c"], [0.0, 1.0])])

    class Reverse:
        def rerank(self, question, hits, k):
            assert len(hits) == 3  # all candidates, not just k
            return list(reversed(hits))[:k]

    config = RetrievalConfig(rerank="rerank-3", candidates=3, k=2)
    hits = retrieve(
        conn, "q", [1.0, 0.0], "1.37", config, model="fake-2d", dim=2, reranker=Reverse()
    )
    assert [h.text for h in hits] == ["text c", "text b"]
