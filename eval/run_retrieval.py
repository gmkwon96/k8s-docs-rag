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
from rag.rerank import FREE_TIER_MODELS, VoyageReranker
from rag.retrieve import hybrid_search, keyword_search, vector_search
from rag.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results" / "retrieval"
HEADLINE = ("recall@5", "recall@10", "hit@5", "mrr", "ndcg@10")


def query_vectors(
    embedder, questions: list[str], cache: Cache, input_type: str = "query"
) -> list[list[float]]:
    """Embeddings for search, from the cache when possible; one batched call for the rest."""
    keys = [hashlib.sha256(f"{input_type}\n{q}".encode()).hexdigest() for q in questions]
    missing = sorted({(k, q) for k, q in zip(keys, questions, strict=True) if cache.get(k) is None})
    if missing:
        vectors, _ = embedder.embed([q for _, q in missing], input_type)
        cache.add([(k, v) for (k, _), v in zip(missing, vectors, strict=True)])
    return [cache.get(k) for k in keys]


def load_rewrites(split: str, query_mode: str) -> dict[str, str] | None:
    if query_mode == "original":
        return None
    mode = "hyde" if query_mode == "hyde" else "rewrite"
    path = ROOT / "eval" / "results" / "rewrites" / f"{split}-{mode}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found; run eval.make_rewrites --mode {mode} first")
    return {r["id"]: r["text"] for r in map(json.loads, path.read_text().splitlines())}


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


METHODS = ("vector", "keyword", "hybrid")


def search(
    conn, method: str, item, vector, *, k: int, embedder, index_name: str, version_filter=True
):
    if method == "vector":
        return vector_search(
            conn,
            vector,
            item.version,
            k=k,
            model=embedder.name,
            index_name=index_name,
            version_filter=version_filter,
        )
    if not version_filter:
        raise SystemExit("--no-version-filter is only implemented for --method vector")
    if method == "keyword":
        return keyword_search(conn, item.question, item.version, k=k, index_name=index_name)
    return hybrid_search(
        conn, vector, item.question, item.version, k=k, model=embedder.name, index_name=index_name
    )


def run(
    conn,
    embedder,
    items,
    *,
    k: int,
    index_name: str,
    cache: Cache,
    method: str = "vector",
    reranker: VoyageReranker | None = None,
    candidates: int = 50,
    query_mode: str = "original",
    rewrites: dict[str, str] | None = None,
    version_filter: bool = True,
) -> list[dict]:
    """With a reranker, fetch `candidates` hits and keep the reranker's top k.

    query_mode (E7) picks what is embedded: the question, its rewrite, a HyDE passage
    (embedded as a document), or "union": candidates for the question and for its rewrite,
    merged. Reranking always uses the original question."""
    items = [i for i in items if i.evidence]
    vectors = query_vectors(embedder, [i.question for i in items], cache)
    if query_mode in ("rewrite", "union"):
        alt = query_vectors(embedder, [rewrites[i.id] for i in items], cache)
    elif query_mode == "hyde":
        alt = query_vectors(embedder, [rewrites[i.id] for i in items], cache, "document")
    rows = []
    for n, (item, vector) in enumerate(zip(items, vectors, strict=True)):
        t0 = time.monotonic()
        depth = candidates if reranker else k
        search_vector = vector if query_mode in ("original", "union") else alt[n]
        hits = search(
            conn,
            method,
            item,
            search_vector,
            k=depth,
            embedder=embedder,
            index_name=index_name,
            version_filter=version_filter,
        )
        if query_mode == "union":
            seen = {h.content_id for h in hits}
            extra = search(
                conn,
                method,
                item,
                alt[n],
                k=depth,
                embedder=embedder,
                index_name=index_name,
                version_filter=version_filter,
            )
            hits += [h for h in extra if h.content_id not in seen]
        if reranker:
            hits = reranker.rerank(item.question, hits, k)
        latency = time.monotonic() - t0
        facts = [[(p.url, p.quote) for p in e.passages()] for e in item.evidence]
        metrics = score([(h.url, h.text) for h in hits], facts)
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
    parser.add_argument(
        "--exact",
        action="store_true",
        help="exact nearest neighbours (no HNSW) to rule out approximate-search misses",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--method", choices=METHODS, default="vector")
    parser.add_argument("--rerank", choices=FREE_TIER_MODELS, help="rerank candidates")
    parser.add_argument("--candidates", type=int, default=50, help="hits fed to the reranker")
    parser.add_argument(
        "--query-mode", choices=["original", "rewrite", "hyde", "union"], default="original"
    )
    parser.add_argument("--no-version-filter", action="store_true", help="E6: search all versions")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.split == "test":
        print("note: the test split is scored once per milestone; make sure this run is one")

    embedder = EMBEDDERS[args.model]
    cache = Cache(CACHE_DIR / f"{embedder.name}-query.jsonl")
    items = load_split(args.split)
    with psycopg.connect(get_settings().database_url) as conn:
        if args.exact:
            conn.execute("SET enable_indexscan = off")
            conn.commit()
        rows = run(
            conn,
            embedder,
            items,
            k=args.k,
            index_name=args.index_name,
            cache=cache,
            method=args.method,
            reranker=VoyageReranker(args.rerank) if args.rerank else None,
            candidates=args.candidates,
            query_mode=args.query_mode,
            rewrites=load_rewrites(args.split, args.query_mode),
            version_filter=not args.no_version_filter,
        )
    summary = {
        "name": args.name,
        "split": args.split,
        "config": {
            "method": args.method,
            "rerank": args.rerank,
            "candidates": args.candidates if args.rerank else None,
            "query_mode": args.query_mode,
            "version_filter": not args.no_version_filter,
            "model": args.model,
            "index_name": args.index_name,
            "exact": args.exact,
            "k": args.k,
        },
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
