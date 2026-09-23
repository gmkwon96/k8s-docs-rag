from dataclasses import dataclass, field

import pytest

from ingest.embed import Cache, batches, decode_vector, embed_index, encode_vector
from ingest.load import load
from tests.test_load import conn, content, test_db_url  # noqa: F401  (fixtures)


@dataclass
class FakeEmbedder:
    name: str = "fake"
    dim: int = 3
    max_batch_texts: int = 2
    max_batch_tokens: int = 1000
    calls: list = field(default_factory=list)

    def embed(self, texts, input_type):
        self.calls.append((list(texts), input_type))
        # distinct per text: the last character's code point ("text a" -> 97)
        vectors = [[float(ord(t[-1])), 1.0, 0.5] for t in texts]
        return vectors, sum(len(t.split()) for t in texts)


def test_vector_encoding_roundtrips_float32():
    v = [0.25, -1.5, 3.0]
    assert decode_vector(encode_vector(v)) == v


def test_batches_respect_text_and_token_limits():
    rows = [(i, f"k{i}", "t", n) for i, n in enumerate([100, 100, 100, 500, 10])]
    sizes = [[r[0] for r in b] for b in batches(rows, max_texts=2, max_tokens=300)]
    assert sizes == [[0, 1], [2], [3], [4]]


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "m.jsonl"
    Cache(path).add([("a", [1.0, 2.0])])
    assert Cache(path).get("a") == [1.0, 2.0]
    assert Cache(path).get("b") is None


def test_embed_index_stores_vectors_and_builds_hnsw_index(conn, tmp_path):  # noqa: F811
    load(conn, "idx", [content(k, f"text {k}", ["1.37"]) for k in "abc"])
    embedder = FakeEmbedder()
    stats = embed_index(conn, embedder, "idx", Cache(tmp_path / "fake.jsonl"), log=lambda _: None)

    assert (stats["embedded"], stats["from_cache"], stats["total"]) == (3, 0, 3)
    assert [len(texts) for texts, _ in embedder.calls] == [2, 1]
    assert all(kind == "document" for _, kind in embedder.calls)
    [(n,)] = conn.execute(
        "SELECT count(*) FROM pg_indexes WHERE indexname = 'embeddings_hnsw_fake_3'"
    ).fetchall()
    assert n == 1
    # nearest neighbour among this model's vectors
    [(key,)] = conn.execute(
        """
        SELECT c.key FROM embeddings e JOIN contents c ON c.id = e.content_id
        WHERE e.model = 'fake'
        ORDER BY e.embedding::vector(3) <-> '[97,1,0.5]'::vector(3) LIMIT 1
        """
    ).fetchall()
    assert key == "a"


def test_second_run_embeds_nothing(conn, tmp_path):  # noqa: F811
    load(conn, "idx", [content("a", "text a", ["1.37"])])
    cache = Cache(tmp_path / "fake.jsonl")
    embed_index(conn, FakeEmbedder(), "idx", cache, log=lambda _: None)
    again = FakeEmbedder()
    stats = embed_index(conn, again, "idx", cache, log=lambda _: None)
    assert (stats["embedded"], stats["from_cache"], again.calls) == (0, 0, [])


def test_vectors_come_from_cache_after_the_database_is_rebuilt(conn, tmp_path):  # noqa: F811
    load(conn, "idx", [content("a", "text a", ["1.37"])])
    cache_path = tmp_path / "fake.jsonl"
    embed_index(conn, FakeEmbedder(), "idx", Cache(cache_path), log=lambda _: None)
    conn.execute("TRUNCATE contents, occurrences, embeddings RESTART IDENTITY")
    conn.commit()
    load(conn, "idx", [content("a", "text a", ["1.37"]), content("b", "text b", ["1.37"])])

    fresh = FakeEmbedder()
    stats = embed_index(conn, fresh, "idx", Cache(cache_path), log=lambda _: None)
    assert (stats["from_cache"], stats["embedded"], stats["total"]) == (1, 1, 2)
    assert [texts for texts, _ in fresh.calls] == [["text b"]]


def test_wrong_vector_shape_is_rejected(conn, tmp_path):  # noqa: F811
    load(conn, "idx", [content("a", "text a", ["1.37"])])
    with pytest.raises(RuntimeError, match="unexpected embedding shape"):
        embed_index(conn, FakeEmbedder(dim=4), "idx", Cache(tmp_path / "f.jsonl"), log=print)


def test_ledger_refuses_requests_over_budget(tmp_path):
    from rag.usage import BudgetExceeded, Ledger

    ledger = Ledger(tmp_path / "usage.jsonl", budget=100)
    ledger.record("voyage-4", 60, "document")
    ledger.check(40)
    with pytest.raises(BudgetExceeded, match="60 tokens used"):
        ledger.check(41)


def test_voyage_embedder_checks_budget_before_calling_api(tmp_path, monkeypatch):
    import rag.embedders as embedders
    from rag.usage import BudgetExceeded, Ledger

    embedder = embedders.VoyageEmbedder("t", "voyage-4", 1024)
    embedder.__dict__["ledger"] = Ledger(tmp_path / "usage.jsonl", budget=5)
    embedder.__dict__["client"] = None  # any API call would fail
    with pytest.raises(BudgetExceeded):
        embedder.embed(["a text that is clearly more than five tokens long"], "document")
