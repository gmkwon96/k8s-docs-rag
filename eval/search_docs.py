"""Find passages in the cleaned docs, for writing and reviewing golden-set evidence.

Prints each match with the version-independent section URL (path#anchor) to put in an
item's evidence, and the matching line to copy as the quote.

Usage:
    uv run python -m eval.search_docs "terminationGracePeriodSeconds" --version 1.37
    uv run python -m eval.search_docs "rollout undo" --path /docs/concepts/workloads/
"""

import argparse
import re

from eval.validate import Docs, split_url
from ingest.chunk import parse_sections


def search(docs: Docs, pattern: str, version: str, path_prefix: str = "", limit: int = 20):
    rx = re.compile(pattern, re.I)
    hits = []
    for path, records in sorted(docs.records(version).items()):
        if not path.startswith(path_prefix):
            continue
        for record in records:
            record_anchor = split_url(record["url"])[1]
            for section in parse_sections(record["title"], record["text"]):
                for block in section.blocks:
                    for line in block.text.splitlines():
                        if rx.search(line):
                            anchor = record_anchor or section.anchor
                            url = path + (f"#{anchor}" if anchor else "")
                            hits.append((url, " > ".join(section.path), line.strip()))
                            if len(hits) >= limit:
                                return hits
    return hits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pattern", help="regular expression, case-insensitive")
    parser.add_argument("--version", default="1.37")
    parser.add_argument("--path", default="", help="only pages under this URL path")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    for url, heading, line in search(Docs(), args.pattern, args.version, args.path, args.limit):
        print(f"{url}\n  [{heading}]\n  {line[:300]}\n")


if __name__ == "__main__":
    main()
