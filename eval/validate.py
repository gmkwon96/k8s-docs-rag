"""Check the golden set against its schema, itself, and the docs it quotes.

Errors (exit code 1):
- schema violations (see eval.golden.Item)
- duplicate ids, or the same question twice (including across dev and test: leakage)
- a version that is not indexed
- an evidence quote that is not in the cleaned docs of the item's version at its URL: in the
  section its #anchor names, or anywhere on the page when there is no anchor
Warnings:
- a quote that no single chunk of the current chunking contains (it straddles a chunk
  boundary, so retrieval can't be scored as relevant under this chunking)

Also prints the category mix of each split against the plan's targets.

Usage:
    uv run python -m eval.validate
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

from eval.golden import CATEGORY_TARGETS, DATASET_DIR, SPLITS, Item, contains_quote, load, normalize
from ingest.chunk import CHUNKS_DIR, parse_sections
from ingest.clean import CLEAN_DIR
from rag.query import indexed_versions


def split_url(url: str) -> tuple[str, str]:
    """Absolute or site-relative URL -> (path with query, fragment)."""
    parts = urlsplit(url)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    return path, parts.fragment


class Docs:
    """Cleaned records and current chunks of each version, indexed by URL path."""

    def __init__(self, clean_dir: Path = CLEAN_DIR, chunks_dir: Path = CHUNKS_DIR):
        self.clean_dir, self.chunks_dir = clean_dir, chunks_dir

    @cache  # noqa: B019  (one instance per run)
    def records(self, version: str) -> dict[str, list[dict]]:
        by_path = defaultdict(list)
        for line in (self.clean_dir / f"v{version}" / "pages.jsonl").read_text().splitlines():
            record = json.loads(line)
            by_path[split_url(record["url"])[0]].append(record)
        return by_path

    @cache  # noqa: B019
    def chunks(self, version: str) -> dict[str, list[str]]:
        by_record = defaultdict(list)
        path = self.chunks_dir / f"v{version}" / "chunks.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                chunk = json.loads(line)
                by_record[chunk["record_id"]].append(normalize(chunk["text"]))
        return by_record

    @cache  # noqa: B019
    def sections(self, version: str, record_id: str) -> dict[str, str]:
        """Section anchor -> that section's own text (normalized), for one record."""
        record = next(
            r for rs in self.records(version).values() for r in rs if r["id"] == record_id
        )
        out: dict[str, str] = {}
        for s in parse_sections(record["title"], record["text"]):
            out[s.anchor] = normalize("\n\n".join(b.text for b in s.blocks))
        return out


def check_evidence(item: Item, docs: Docs) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for ev in (p for e in item.evidence for p in e.passages()):
        path, anchor = split_url(ev.url)
        candidates = docs.records(item.version).get(path, [])
        if not candidates:
            errors.append(f"{item.id}: no page at {path} in v{item.version}")
            continue
        # Glossary terms and feature gates share a path; their anchor picks the record.
        keyed = [r for r in candidates if split_url(r["url"])[1]]
        if keyed:
            candidates = [r for r in keyed if split_url(r["url"])[1] == anchor]
            if not candidates:
                errors.append(f"{item.id}: no record for anchor #{anchor} at {path}")
                continue
        matches = [r for r in candidates if contains_quote(r["text"], ev.quote)]
        if not matches:
            errors.append(f"{item.id}: quote not found at {ev.url} (v{item.version}): {ev.quote!r}")
            continue
        record = matches[0]
        if anchor and not keyed:
            sections = docs.sections(item.version, record["id"])
            if anchor not in sections:
                errors.append(f"{item.id}: #{anchor} is not a section of {path}")
            elif normalize(ev.quote) not in sections[anchor]:
                errors.append(f"{item.id}: quote is on {path} but not in section #{anchor}")
        chunks = docs.chunks(item.version).get(record["id"], [])
        if chunks and not any(normalize(ev.quote) in c for c in chunks):
            warnings.append(f"{item.id}: quote straddles a chunk boundary: {ev.quote[:60]!r}")
    return errors, warnings


def validate(dataset_dir: Path = DATASET_DIR, docs: Docs | None = None) -> dict:
    docs = docs or Docs()
    errors, warnings = [], []
    items: dict[str, list[Item]] = {}
    for split in SPLITS:
        try:
            items[split] = load(dataset_dir / f"{split}.jsonl")
        except ValueError as e:
            errors.append(str(e))
            items[split] = []
        for item in items[split]:
            if item.split != split:
                errors.append(f"{item.id}: split {item.split!r} in {split}.jsonl")

    everything = [i for split in SPLITS for i in items[split]]
    for item_id, n in Counter(i.id for i in everything).items():
        if n > 1:
            errors.append(f"duplicate id {item_id}")
    by_question = defaultdict(list)
    for i in everything:
        by_question[normalize(i.question).lower()].append(i.id)
    for ids in by_question.values():
        if len(ids) > 1:
            errors.append(f"same question in {', '.join(ids)}")

    versions = indexed_versions()
    for item in everything:
        if item.version not in versions:
            errors.append(f"{item.id}: version {item.version} is not indexed")
            continue
        e, w = check_evidence(item, docs)
        errors += e
        warnings += w

    mix = {}
    for split in SPLITS:
        n = len(items[split])
        counts = Counter(i.category for i in items[split])
        mix[split] = {
            c: {"n": counts[c], "share": round(counts[c] / n, 3) if n else 0, "target": t}
            for c, t in CATEGORY_TARGETS.items()
        }
    return {
        "items": {s: len(items[s]) for s in SPLITS},
        "versions": dict(Counter(i.version for i in everything)),
        "mix": mix,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args(argv)
    report = validate(args.dataset_dir)
    for split in SPLITS:
        print(f"{split}: {report['items'][split]} items")
        for c, m in report["mix"][split].items():
            print(f"  {c:14s} {m['n']:3d}  {m['share']:5.0%}  (target {m['target']:.0%})")
    print(f"versions: {report['versions']}")
    for w in report["warnings"]:
        print(f"warning: {w}")
    for e in report["errors"]:
        print(f"error: {e}")
    print(f"{len(report['errors'])} errors, {len(report['warnings'])} warnings")
    sys.exit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
