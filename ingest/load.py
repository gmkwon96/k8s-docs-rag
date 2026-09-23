"""Load deduplicated contents and their occurrences into Postgres.

Reads data/index/<embed-text mode>/contents.jsonl (from ingest.dedupe) into the tables in
db/schema.sql under one index name (default "heading-<mode>").

Reloading an index is an in-place sync: texts that are gone are deleted, new ones are
inserted, and unchanged ones keep their row id, so their embeddings survive. Occurrences
are replaced wholesale. Everything runs in one transaction.

Usage:
    uv run python -m ingest.load
    uv run python -m ingest.load --embed-text breadcrumb
"""

import argparse
import json
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from ingest.dedupe import EMBED_TEXT_MODES, INDEX_DIR
from rag.settings import get_settings

CONTENT_COLUMNS = ("key", "kind", "text", "embed_text", "n_tokens", "versions")
OCCURRENCE_COLUMNS = (
    "version",
    "chunk_id",
    "record_id",
    "url",
    "title",
    "heading_path",
    "anchor",
    "feature_states",
    "content_type",
    "source_path",
    "commit",
)


def default_index_name(mode: str) -> str:
    return f"heading-{mode}"


def load(conn: psycopg.Connection, index_name: str, contents: list[dict]) -> dict:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE new_contents (
                key text, kind text, text text, embed_text text, n_tokens int, versions text[]
            ) ON COMMIT DROP
            """
        )
        with cur.copy(f"COPY new_contents ({', '.join(CONTENT_COLUMNS)}) FROM STDIN") as copy:
            for c in contents:
                copy.write_row([c[col] for col in CONTENT_COLUMNS])

        cur.execute(
            """
            DELETE FROM contents c WHERE c.index_name = %s
            AND NOT EXISTS (SELECT 1 FROM new_contents n WHERE n.key = c.key)
            """,
            [index_name],
        )
        deleted = cur.rowcount
        cur.execute(
            """
            INSERT INTO contents (index_name, key, kind, text, embed_text, n_tokens, versions)
            SELECT %s, key, kind, text, embed_text, n_tokens, versions FROM new_contents
            ON CONFLICT (index_name, key) DO UPDATE SET
                kind = EXCLUDED.kind, text = EXCLUDED.text, n_tokens = EXCLUDED.n_tokens,
                versions = EXCLUDED.versions
            RETURNING (xmax = 0) AS inserted
            """,
            [index_name],
        )
        inserted = sum(row[0] for row in cur.fetchall())

        cur.execute(
            """
            DELETE FROM occurrences o USING contents c
            WHERE o.content_id = c.id AND c.index_name = %s
            """,
            [index_name],
        )
        cur.execute(
            """
            CREATE TEMP TABLE new_occurrences (
                key text, version text, chunk_id text, record_id text, url text, title text,
                heading_path text[], anchor text, feature_states jsonb, content_type text,
                source_path text, commit text
            ) ON COMMIT DROP
            """
        )
        columns = ", ".join(("key", *OCCURRENCE_COLUMNS))
        with cur.copy(f"COPY new_occurrences ({columns}) FROM STDIN") as copy:
            for c in contents:
                for o in c["occurrences"]:
                    row = [o[col] for col in OCCURRENCE_COLUMNS]
                    row[OCCURRENCE_COLUMNS.index("feature_states")] = Jsonb(o["feature_states"])
                    copy.write_row([c["key"], *row])
        cur.execute(
            f"""
            INSERT INTO occurrences (content_id, {", ".join(OCCURRENCE_COLUMNS)})
            SELECT c.id, {", ".join(f"n.{col}" for col in OCCURRENCE_COLUMNS)}
            FROM new_occurrences n JOIN contents c ON c.index_name = %s AND c.key = n.key
            """,
            [index_name],
        )
        occurrences = cur.rowcount

    return {
        "index_name": index_name,
        "contents": len(contents),
        "inserted": inserted,
        "kept": len(contents) - inserted,
        "deleted": deleted,
        "occurrences": occurrences,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embed-text", choices=EMBED_TEXT_MODES, default="plain")
    parser.add_argument("--index-name", help="default: heading-<embed-text>")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)

    path = args.index_dir / args.embed_text / "contents.jsonl"
    contents = [json.loads(line) for line in path.read_text().splitlines()]
    index_name = args.index_name or default_index_name(args.embed_text)
    with psycopg.connect(args.database_url or get_settings().database_url) as conn:
        stats = load(conn, index_name, contents)
    print(
        f"{stats['index_name']}: {stats['contents']:,} contents "
        f"({stats['inserted']:,} new, {stats['kept']:,} kept, {stats['deleted']:,} deleted), "
        f"{stats['occurrences']:,} occurrences"
    )


if __name__ == "__main__":
    main()
