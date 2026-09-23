"""Embed an index's contents with one model and store the vectors in Postgres.

Only contents without a vector for the model are embedded. Every vector is also appended to
a local cache (data/embeddings/<model>.jsonl, keyed by the sha256 of the embedded text), so
rebuilding the database or reloading an index never pays for the same text twice.
After loading, a partial HNSW index for the model is created if missing.

Usage:
    uv run python -m ingest.embed                          # voyage-4 on heading-plain
    uv run python -m ingest.embed --model voyage-4 --index-name heading-breadcrumb
"""

import argparse
import base64
import json
import re
import time
from array import array
from pathlib import Path

import psycopg
from psycopg import sql

from rag.embedders import EMBEDDERS
from rag.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "embeddings"


# ---------------------------------------------------------------------------- cache


def encode_vector(vector: list[float]) -> str:
    return base64.b64encode(array("f", vector).tobytes()).decode()


def decode_vector(data: str) -> list[float]:
    return array("f", base64.b64decode(data)).tolist()


class Cache:
    """Append-only JSONL of {key, embedding}; float32, base64-encoded."""

    def __init__(self, path: Path):
        self.path = path
        self.vectors: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.vectors[row["key"]] = row["embedding"]

    def get(self, key: str) -> list[float] | None:
        data = self.vectors.get(key)
        return decode_vector(data) if data is not None else None

    def add(self, items: list[tuple[str, list[float]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for key, vector in items:
                data = encode_vector(vector)
                self.vectors[key] = data
                f.write(json.dumps({"key": key, "embedding": data}) + "\n")


# ---------------------------------------------------------------------------- batching


def batches(rows: list[tuple], max_texts: int, max_tokens: int):
    """Group (id, key, text, n_tokens) rows under both request limits."""
    batch, tokens = [], 0
    for row in rows:
        n = row[3]
        if batch and (len(batch) >= max_texts or tokens + n > max_tokens):
            yield batch
            batch, tokens = [], 0
        batch.append(row)
        tokens += n
    if batch:
        yield batch


# ---------------------------------------------------------------------------- database


def missing_rows(conn: psycopg.Connection, index_name: str, model: str) -> list[tuple]:
    return conn.execute(
        """
        SELECT c.id, c.key, c.embed_text, c.n_tokens FROM contents c
        WHERE c.index_name = %s
        AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.content_id = c.id AND e.model = %s)
        ORDER BY c.id
        """,
        [index_name, model],
    ).fetchall()


def store(conn: psycopg.Connection, model: str, items: list[tuple[int, list[float]]]) -> None:
    with conn.transaction(), conn.cursor() as cur:
        with cur.copy("COPY embeddings (content_id, model, embedding) FROM STDIN") as copy:
            for content_id, vector in items:
                copy.write_row([content_id, model, "[" + ",".join(map(repr, vector)) + "]"])


def hnsw_index_name(model: str, dim: int) -> str:
    return f"embeddings_hnsw_{re.sub(r'[^a-z0-9]+', '_', model.lower())}_{dim}"


def ensure_hnsw_index(conn: psycopg.Connection, model: str, dim: int) -> str:
    """Partial HNSW index for one model: queries must use the same cast and model filter."""
    name = hnsw_index_name(model, dim)
    conn.execute(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {name} ON embeddings "
            "USING hnsw ((embedding::vector({dim})) vector_cosine_ops) WHERE model = {model}"
        ).format(
            name=sql.Identifier(name),
            dim=sql.Literal(dim),
            model=sql.Literal(model),
        )
    )
    conn.commit()
    return name


# ---------------------------------------------------------------------------- driver


def embed_index(conn, embedder, index_name: str, cache: Cache, log=print) -> dict:
    rows = missing_rows(conn, index_name, embedder.name)
    cached = [(r[0], v) for r in rows if (v := cache.get(r[1])) is not None]
    store(conn, embedder.name, cached)
    todo = [r for r in rows if r[1] not in cache.vectors]
    if todo and hasattr(embedder, "ledger"):
        # Fail before the first request if the whole run could exceed the budget.
        estimate = embedder.estimate_tokens([r[2] for r in todo])
        embedder.ledger.check(estimate)
        log(
            f"  {len(todo):,} texts to embed, ~{estimate:,} tokens estimated; "
            f"{embedder.ledger.total():,} used of budget {embedder.ledger.budget:,}"
        )
    billed, start = 0, time.monotonic()
    for i, batch in enumerate(
        batches(todo, embedder.max_batch_texts, embedder.max_batch_tokens), start=1
    ):
        vectors, tokens = embedder.embed([r[2] for r in batch], "document")
        if len(vectors) != len(batch) or any(len(v) != embedder.dim for v in vectors):
            raise RuntimeError(f"unexpected embedding shape from {embedder.name}")
        cache.add([(r[1], v) for r, v in zip(batch, vectors, strict=True)])
        store(conn, embedder.name, [(r[0], v) for r, v in zip(batch, vectors, strict=True)])
        billed += tokens
        log(f"  batch {i}: {len(batch)} texts, {tokens:,} tokens")
    index = ensure_hnsw_index(conn, embedder.name, embedder.dim)
    total = conn.execute(
        """
        SELECT count(*) FROM embeddings e JOIN contents c ON c.id = e.content_id
        WHERE c.index_name = %s AND e.model = %s
        """,
        [index_name, embedder.name],
    ).fetchone()[0]
    return {
        "from_cache": len(cached),
        "embedded": len(todo),
        "billed_tokens": billed,
        "seconds": round(time.monotonic() - start, 1),
        "total": total,
        "hnsw_index": index,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=sorted(EMBEDDERS), default="voyage-4")
    parser.add_argument("--index-name", default="heading-plain")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)

    embedder = EMBEDDERS[args.model]
    cache = Cache(args.cache_dir / f"{embedder.name}.jsonl")
    with psycopg.connect(args.database_url or get_settings().database_url) as conn:
        stats = embed_index(conn, embedder, args.index_name, cache)
    print(
        f"{args.index_name} x {embedder.name}: {stats['embedded']:,} embedded "
        f"({stats['billed_tokens']:,} tokens, {stats['seconds']}s), "
        f"{stats['from_cache']:,} from cache, {stats['total']:,} total, index {stats['hnsw_index']}"
    )


if __name__ == "__main__":
    main()
