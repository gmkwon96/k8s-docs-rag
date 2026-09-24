"""Retrieval over one index, filtered to one Kubernetes version.

- vector_search: embedding similarity through the model's HNSW index
- keyword_search: Postgres full-text search over the `tsv` column. The question's lexemes
  are OR-ed (an AND of every word in a natural-language question matches almost nothing)
  and ranked with ts_rank normalized by document length; this is not BM25.
- hybrid_search: Reciprocal Rank Fusion of both candidate lists
"""

from dataclasses import dataclass

import psycopg


@dataclass
class Hit:
    content_id: int
    score: float  # cosine similarity
    text: str
    version: str
    chunk_id: str
    url: str
    title: str
    heading_path: list[str]
    feature_states: list[dict]


OCCURRENCE_JOIN = """
    CROSS JOIN LATERAL (
        SELECT * FROM occurrences o
        WHERE o.content_id = c.id AND o.version = %(version)s
        ORDER BY o.chunk_id LIMIT 1
    ) o
"""
HIT_COLUMNS = "c.text, o.version, o.chunk_id, o.url, o.title, o.heading_path, o.feature_states"
RRF_K = 60


def vector(values: list[float]) -> str:
    return "[" + ",".join(map(repr, values)) + "]"


def vector_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    version: str,
    *,
    k: int = 8,
    model: str = "voyage-4",
    dim: int = 1024,
    index_name: str = "heading-plain",
) -> list[Hit]:
    """Top-k contents present in `version`, with that version's URL and metadata.

    Uses the model's partial HNSW index (same cast and model filter). Iterative scan keeps
    returning candidates when the version filter rejects some, so we still get k hits.
    """
    q = vector(query_vector)
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(
            f"""
            SELECT c.id, 1 - (e.embedding::vector({dim}) <=> %(q)s::vector({dim})) AS score,
                   c.text, o.version, o.chunk_id, o.url, o.title, o.heading_path,
                   o.feature_states
            FROM embeddings e
            JOIN contents c ON c.id = e.content_id
            CROSS JOIN LATERAL (
                SELECT * FROM occurrences o
                WHERE o.content_id = c.id AND o.version = %(version)s
                ORDER BY o.chunk_id LIMIT 1
            ) o
            WHERE e.model = %(model)s AND c.index_name = %(index)s
              AND c.versions @> ARRAY[%(version)s]
            ORDER BY e.embedding::vector({dim}) <=> %(q)s::vector({dim})
            LIMIT %(k)s
            """,
            {"q": q, "version": version, "model": model, "index": index_name, "k": k},
        ).fetchall()
    return [Hit(*row) for row in rows]


def keyword_search(
    conn: psycopg.Connection,
    question: str,
    version: str,
    *,
    k: int = 8,
    index_name: str = "heading-plain",
) -> list[Hit]:
    rows = conn.execute(
        f"""
        WITH q AS (
            SELECT to_tsquery('english', coalesce(string_agg(DISTINCT lexeme, ' | '), ''))
                AS query
            FROM unnest(to_tsvector('english', %(question)s))
        )
        SELECT c.id, ts_rank(c.tsv, q.query, 1) AS score, {HIT_COLUMNS}
        FROM contents c CROSS JOIN q
        {OCCURRENCE_JOIN}
        WHERE c.index_name = %(index)s AND c.versions @> ARRAY[%(version)s]
          AND c.tsv @@ q.query
        ORDER BY score DESC, c.id
        LIMIT %(k)s
        """,
        {"question": question, "version": version, "index": index_name, "k": k},
    ).fetchall()
    return [Hit(*row) for row in rows]


def rrf(rankings: list[list[Hit]], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum of 1 / (rrf_k + rank) over the lists."""
    scores: dict[int, float] = {}
    first: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.content_id] = scores.get(hit.content_id, 0.0) + 1 / (rrf_k + rank)
            first.setdefault(hit.content_id, hit)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    return [Hit(**{**first[cid].__dict__, "score": scores[cid]}) for cid in order]


def hybrid_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    question: str,
    version: str,
    *,
    k: int = 8,
    candidates: int = 50,
    model: str = "voyage-4",
    dim: int = 1024,
    index_name: str = "heading-plain",
) -> list[Hit]:
    dense = vector_search(
        conn, query_vector, version, k=candidates, model=model, dim=dim, index_name=index_name
    )
    sparse = keyword_search(conn, question, version, k=candidates, index_name=index_name)
    return rrf([dense, sparse], k)


@dataclass(frozen=True)
class RetrievalConfig:
    """How the pipeline retrieves. Defaults are the best dev setting (experiment E5)."""

    method: str = "vector"  # vector | keyword | hybrid
    rerank: str | None = "rerank-3"
    candidates: int = 50  # hits fed to the reranker
    k: int = 8  # chunks given to the model


BASELINE = RetrievalConfig(rerank=None)  # milestone 2-3 setting: vector top k
DEFAULT = RetrievalConfig()


def retrieve(
    conn,
    question: str,
    query_vector,
    version: str,
    config: RetrievalConfig = DEFAULT,
    *,
    model: str = "voyage-4",
    dim: int = 1024,
    reranker=None,
) -> list[Hit]:
    depth = config.candidates if config.rerank else config.k
    if config.method == "vector":
        hits = vector_search(conn, query_vector, version, k=depth, model=model, dim=dim)
    elif config.method == "keyword":
        hits = keyword_search(conn, question, version, k=depth)
    else:
        hits = hybrid_search(conn, query_vector, question, version, k=depth, model=model)
    if config.rerank:
        if reranker is None:
            from rag.rerank import VoyageReranker

            reranker = VoyageReranker(config.rerank)
        hits = reranker.rerank(question, hits, config.k)
    return hits
