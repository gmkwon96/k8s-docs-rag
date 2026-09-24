"""Write the static demo's data: example answers with the chunks they were generated from,
and the eval results with bootstrap intervals, as JSON under web/public/data/.

The static site (web, STATIC_EXPORT=1) reads these instead of calling the API, so it runs
on any static host with no Claude or Voyage spend. The chunks are the ones the eval run
retrieved for each answer (by chunk id), looked up in Postgres: no API calls at all.

Usage:
    uv run python -m scripts.export_demo
"""

import json
from pathlib import Path

import psycopg

from api.main import EXAMPLES_RUN, RESULTS_DIR, eval_results
from rag.query import indexed_versions
from rag.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "web" / "public" / "data"
MAX_CHUNK_CHARS = 700  # the panel shows a few lines; keep the download small


def chunks(conn: psycopg.Connection, chunk_ids: list[str]) -> list[dict]:
    rows = conn.execute(
        """
        SELECT o.chunk_id, c.id, left(c.text, %s), o.version, o.url, o.title, o.heading_path,
               o.feature_states
        FROM occurrences o JOIN contents c ON c.id = o.content_id
        WHERE o.chunk_id = ANY(%s) AND c.index_name = 'heading-plain'
        """,
        [MAX_CHUNK_CHARS, chunk_ids],
    ).fetchall()
    by_id = {r[0]: r for r in rows}
    missing = [i for i in chunk_ids if i not in by_id]
    if missing:
        raise RuntimeError(f"chunks not in the database: {missing[:3]}")
    keys = ("chunk_id", "content_id", "text", "version", "url", "title", "heading_path")
    return [
        {**dict(zip(keys, by_id[i][:7], strict=True)), "feature_states": by_id[i][7], "score": None}
        for i in chunk_ids
    ]


def main() -> None:
    path = RESULTS_DIR / "generation" / f"{EXAMPLES_RUN}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keep = ("id", "category", "version", "question", "answer", "found", "sources")
    with psycopg.connect(get_settings().database_url) as conn:
        examples = [{**{k: r[k] for k in keep}, "hits": chunks(conn, r["retrieved"])} for r in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {"run": EXAMPLES_RUN, "versions": list(indexed_versions()), "examples": examples}
    (OUT_DIR / "examples.json").write_text(json.dumps(data, ensure_ascii=False))
    (OUT_DIR / "eval-results.json").write_text(json.dumps(eval_results(), ensure_ascii=False))
    for p in sorted(OUT_DIR.glob("*.json")):
        print(f"{p.relative_to(ROOT)}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
