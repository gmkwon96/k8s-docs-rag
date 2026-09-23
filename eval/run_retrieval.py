"""Score retrieval on a golden-set split: no LLM calls, only query embeddings.

For every item with evidence, embed the question (cached by text, so reruns cost nothing),
search its version, and compute eval.metrics_retrieval on the top k chunks. Writes one
JSONL row per item and a summary (overall and per category) to eval/results/retrieval/.

Usage:
    uv run python -m eval.run_retrieval                     # dev split, baseline config
    uv run python -m eval.run_retrieval --name e4-hybrid --k 20
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import psycopg

from eval.golden import load_split
from eval.metrics_retrieval import score
from ingest.embed import CACHE_DIR, Cache
from rag.embedders import EMBEDDERS
from rag.retrieve import vector_search
from rag.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results" / "retrieval"
HEADLINE = ("recall@5", "recall@10", "hit@5", "mrr", "ndcg@10")


def query_vectors(embedder, questions: list[str], cache: Cache) -> list[list[float]]:
    """Query embeddings, from the cache when possible; one batched call for the rest."""
    keys = [hashlib.sha256(f"query\n{q}".encode()).hexdigest() for q in questions]
    missing = sorted({(k, q) for k, q in zip(keys, questions, strict=True) if cache.get(k) is None})
    if missing:
        vectors, _ = embedder.embed([q for _, q in missing], "query")
        cache.add([(k, v) for (k, _), v in zip(missing, vectors, strict=True)])
    return [cache.get(k) for k in keys]


def summarize(rows: list[dict]) -> dict:
    def mean(rs, m):
        return round(statistics.mean(r["metrics"][m] for r in rs), 4) if rs else None

    metric_names = list(rows[0]["metrics"]) if rows else []
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {
        "n": len(rows),
        "overall": {m: mean(rows, m) for m in metric_names if m != "first_relevant_rank"},
        "by_category": {
            c: {"n": len(rs), **{m: mean(rs, m) for m in HEADLINE}}
            for c, rs in sorted(by_cat.items())
        },
    }


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def run(conn, embedder, items, *, k: int, index_name: str, cache: Cache) -> list[dict]:
    items = [i for i in items if i.evidence]
    vectors = query_vectors(embedder, [i.question for i in items], cache)
    rows = []
    for item, vector in zip(items, vectors, strict=True):
        t0 = time.monotonic()
        hits = vector_search(
            conn, vector, item.version, k=k, model=embedder.name, index_name=index_name
        )
        latency = time.monotonic() - t0
        metrics = score([h.text for h in hits], [e.quotes() for e in item.evidence])
        rows.append(
            {
                "id": item.id,
                "category": item.category,
                "version": item.version,
                "metrics": metrics,
                "retrieved": [{"chunk_id": h.chunk_id, "score": round(h.score, 4)} for h in hits],
                "latency_ms": round(latency * 1000, 1),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--name", default="baseline", help="run name for the output files")
    parser.add_argument("--model", default="voyage-4", choices=sorted(EMBEDDERS))
    parser.add_argument("--index-name", default="heading-plain")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.split == "test":
        print("note: the test split is scored once per milestone; make sure this run is one")

    embedder = EMBEDDERS[args.model]
    cache = Cache(CACHE_DIR / f"{embedder.name}-query.jsonl")
    items = load_split(args.split)
    with psycopg.connect(get_settings().database_url) as conn:
        rows = run(conn, embedder, items, k=args.k, index_name=args.index_name, cache=cache)
    summary = {
        "name": args.name,
        "split": args.split,
        "config": {"model": args.model, "index_name": args.index_name, "k": args.k},
        "commit": git_commit(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "skipped_without_evidence": len(items) - len(rows),
        **summarize(rows),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}-{args.name}"
    with (args.out_dir / f"{stem}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (args.out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    skipped = summary["skipped_without_evidence"]
    print(f"{stem}: {summary['n']} items with evidence ({skipped} without, skipped)")
    print("  overall  " + "  ".join(f"{m} {summary['overall'][m]:.3f}" for m in HEADLINE))
    for c, s in summary["by_category"].items():
        print(f"  {c:13s} n={s['n']:2d}  " + "  ".join(f"{m} {s[m]:.3f}" for m in HEADLINE))


if __name__ == "__main__":
    main()
