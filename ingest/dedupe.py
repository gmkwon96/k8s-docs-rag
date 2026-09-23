"""Merge identical chunks across versions and pages so each text is embedded once.

Reads data/chunks/v<version>/chunks.jsonl for every version and writes
data/index/<embed-text mode>/contents.jsonl plus report.json.

Two chunks are the same content when the exact string we would embed is the same:
- "plain" (default): the chunk text
- "breadcrumb": the heading path ("Pod > PodSpec") plus the chunk text, so the same text
  under different headings stays separate (experiment E1)

Only the text is shared. Metadata stays per occurrence, because identical text can mean
different things in different versions: a section's inherited feature state (alpha in
1.35, beta in 1.36), its URL (v1-35.docs.kubernetes.io vs kubernetes.io), its heading path.

Usage:
    uv run python -m ingest.dedupe
    uv run python -m ingest.dedupe --embed-text breadcrumb
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ingest.chunk import CHUNKS_DIR, count_tokens
from ingest.fetch import LOCK_PATH, load_lock

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"

EMBED_TEXT_MODES = ("plain", "breadcrumb")
OCCURRENCE_FIELDS = (
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


def embed_text(chunk: dict, mode: str) -> str:
    if mode == "plain":
        return chunk["text"]
    if mode == "breadcrumb":
        return " > ".join(chunk["heading_path"]) + "\n\n" + chunk["text"]
    raise ValueError(f"unknown embed text mode: {mode}")


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def dedupe(chunks: list[dict], mode: str = "plain") -> list[dict]:
    """Group chunks by the string to embed, keeping first-seen order."""
    contents: dict[str, dict] = {}
    for chunk in chunks:
        text = embed_text(chunk, mode)
        key = hashlib.sha256(text.encode()).hexdigest()
        content = contents.get(key)
        if content is None:
            content = contents[key] = {
                "key": key,
                "content_hash": chunk["content_hash"],
                "kind": chunk["kind"],
                "embed_text": text,
                "text": chunk["text"],
                "n_tokens": chunk["n_tokens"] if mode == "plain" else count_tokens(text),
                "versions": [],
                "occurrences": [],
            }
        content["occurrences"].append({f: chunk.get(f) for f in OCCURRENCE_FIELDS})
    for content in contents.values():
        content["versions"] = sorted(
            {o["version"] for o in content["occurrences"]}, key=version_key
        )
    return list(contents.values())


def differs(occurrences: list[dict], field: str) -> bool:
    """Whether a field takes more than one value across the versions of one content."""
    per_version: dict[str, set] = {}
    for o in occurrences:
        per_version.setdefault(o["version"], set()).add(json.dumps(o[field], sort_keys=True))
    return len({frozenset(v) for v in per_version.values()}) > 1


def pages_per_version(content: dict) -> Counter:
    return Counter(v for v, _ in {(o["version"], o["record_id"]) for o in content["occurrences"]})


def build_report(chunks: list[dict], contents: list[dict], mode: str) -> dict:
    tokens_in = sum(
        c["n_tokens"] if mode == "plain" else count_tokens(embed_text(c, mode)) for c in chunks
    )
    tokens_out = sum(c["n_tokens"] for c in contents)
    multi = [c for c in contents if len(c["versions"]) > 1]
    return {
        "embed_text": mode,
        "chunks": len(chunks),
        "contents": len(contents),
        "content_ratio": round(len(contents) / len(chunks), 3) if chunks else 0,
        "tokens_before": tokens_in,
        "tokens_after": tokens_out,
        "token_ratio": round(tokens_out / tokens_in, 3) if tokens_in else 0,
        "chunks_per_version": dict(Counter(c["version"] for c in chunks)),
        "version_sets": {
            ",".join(k): n for k, n in Counter(tuple(c["versions"]) for c in contents).most_common()
        },
        # e.g. the same API type definition inlined into several resource pages
        "on_several_pages_in_a_version": sum(
            any(n > 1 for n in pages_per_version(c).values()) for c in contents
        ),
        # e.g. identical parameter lists under several operations of one API page
        "repeated_within_a_page": sum(
            len(c["occurrences"]) > len({(o["version"], o["record_id"]) for o in c["occurrences"]})
            for c in contents
        ),
        "metadata_differs_across_versions": {
            field: sum(differs(c["occurrences"], field) for c in multi)
            for field in ("feature_states", "heading_path", "anchor")
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--versions", nargs="+", help="default: versions in the lock file")
    parser.add_argument("--embed-text", choices=EMBED_TEXT_MODES, default="plain")
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=INDEX_DIR)
    args = parser.parse_args(argv)
    versions = sorted(args.versions or load_lock(LOCK_PATH)["sources"], key=version_key)

    chunks = []
    for version in versions:
        path = args.chunks_dir / f"v{version}" / "chunks.jsonl"
        chunks += [json.loads(line) for line in path.read_text().splitlines()]
    contents = dedupe(chunks, args.embed_text)
    report = {"versions": versions, **build_report(chunks, contents, args.embed_text)}

    out = args.out_dir / args.embed_text
    out.mkdir(parents=True, exist_ok=True)
    with (out / "contents.jsonl").open("w") as f:
        for c in contents:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{args.embed_text}: {report['chunks']:,} chunks -> {report['contents']:,} contents "
        f"({report['content_ratio']:.0%}), tokens {report['tokens_before']:,} -> "
        f"{report['tokens_after']:,} ({report['token_ratio']:.0%}) -> {out}"
    )


if __name__ == "__main__":
    main()
