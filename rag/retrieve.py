"""Vector search over one index, filtered to one Kubernetes version."""

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
