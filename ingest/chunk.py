"""Split cleaned doc records into retrieval chunks, one JSONL file per version.

Reads data/clean/v<version>/pages.jsonl (from ingest.clean) and writes
data/chunks/v<version>/chunks.jsonl plus report.json.

Strategy "heading" (the baseline):
- A section is a heading plus everything up to the next heading, outside fenced code.
- A section that fits in MAX_TOKENS is one chunk. A larger one is packed greedily from
  blocks (paragraphs, lists, code blocks); a code block is split only if it alone exceeds
  MAX_TOKENS, and each piece is re-fenced.
- A chunk under MIN_TOKENS is merged into the next chunk of the same record when the
  result still fits (typically a heading with a short intro followed by its subsections).
- Each chunk carries its heading path (page title > h2 > h3 ...), the anchor of its first
  heading for citation links, and the feature states that apply to it: those marked in
  its section or in any ancestor section.

Token counts use tiktoken's o200k_base as a model-agnostic proxy until the embedding
model is chosen (experiment E3).

Usage:
    uv run python -m ingest.chunk                  # all versions in ingest/sources.lock.json
    uv run python -m ingest.chunk --versions 1.37
"""

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import tiktoken

from ingest.clean import CLEAN_DIR
from ingest.fetch import LOCK_PATH, load_lock

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "chunks"

MAX_TOKENS = 512
MIN_TOKENS = 64

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{#([^}]+)\})?\s*$")
FENCE_OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
# The code span clean.py renders for every feature-state shortcode (the label before it
# comes from i18n, "Feature state:" in English).
FEATURE_STATE_RE = re.compile(r"`Kubernetes v[\d.]+ \[[\w-]+\]`")


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------- parsing


@dataclass
class Block:
    text: str
    is_code: bool = False  # a whole fenced code block, fence lines included


@dataclass
class Section:
    level: int  # 0 for the part before the first heading
    title: str
    anchor: str
    path: list[str]  # heading titles from the page title down to this section
    blocks: list[Block] = field(default_factory=list)
    feature_states: list[dict] = field(default_factory=list)  # own + inherited


def anchorize(text: str) -> str:
    """Hugo's autoHeadingIDType = "blackfriday": runs of non-alphanumerics become one '-'."""
    out, dash = [], False
    for ch in plain_heading(text):
        if ch.isalnum() or unicodedata.category(ch).startswith(("L", "N")):
            if dash and out:
                out.append("-")
            dash = False
            out.append(ch.lower())
        else:
            dash = True
    return "".join(out)


def plain_heading(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> their text
    return re.sub(r"[`*_]", "", text).strip()


def closes(line: str, fence: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(fence) and set(stripped) == {fence[0]}


def split_lines(text: str) -> list[tuple[str, int | None]]:
    """Pair each line with the index of the fenced code block it belongs to, if any.

    A fence that never closes is treated as plain text, so one stray fence can't swallow
    the rest of the page."""
    lines = text.splitlines()
    code_id: list[int | None] = [None] * len(lines)
    i = n = 0
    while i < len(lines):
        m = FENCE_OPEN_RE.match(lines[i])
        if m:
            end = next((j for j in range(i + 1, len(lines)) if closes(lines[j], m[2])), None)
            if end is not None:
                code_id[i : end + 1] = [n] * (end + 1 - i)
                n += 1
                i = end + 1
                continue
        i += 1
    return list(zip(lines, code_id, strict=True))


def blocks_from(lines: list[tuple[str, int | None]]) -> list[Block]:
    """Group lines into blocks: each fenced code block, or a run of non-blank lines."""
    blocks: list[Block] = []
    buf: list[str] = []
    code: list[str] = []
    code_id = None

    def flush_text():
        if buf:
            blocks.append(Block("\n".join(buf)))
            buf.clear()

    def flush_code():
        if code:
            blocks.append(Block("\n".join(code), is_code=True))
            code.clear()

    for line, cid in lines:
        if cid is not None:
            if cid != code_id:
                flush_text()
                flush_code()
                code_id = cid
            code.append(line)
            continue
        flush_code()
        code_id = None
        if line.strip():
            buf.append(line)
        else:
            flush_text()
    flush_text()
    flush_code()
    return blocks


def parse_sections(title: str, text: str) -> list[Section]:
    lines = split_lines(text)
    sections: list[Section] = []
    stack: list[Section] = []  # open ancestors by level
    current = Section(0, title, "", [title])
    body: list[tuple[str, int | None]] = []
    seen_anchors: Counter = Counter()

    def close_current():
        current.blocks = blocks_from(body)
        sections.append(current)

    for line, code in lines:
        m = None if code is not None else HEADING_RE.match(line)
        if not m:
            body.append((line, code))
            continue
        level, heading, explicit = len(m[1]), m[2], m[3]
        if level == 1 and not sections and not any(ln.strip() for ln, _ in body):
            # A leading h1 is the record's own title ("# Pod", "# Feature gate: X"): keep
            # the line, but not as a section, so the record URL's anchor (#term-pod, #X)
            # stays.
            body.append((line, None))
            continue
        close_current()
        body = []
        while stack and stack[-1].level >= level:
            stack.pop()
        parent_path = stack[-1].path if stack else [title]
        anchor = explicit or anchorize(heading)
        if not explicit:
            n = seen_anchors[anchor]
            seen_anchors[anchor] += 1
            anchor = f"{anchor}-{n}" if n else anchor
        name = plain_heading(heading)
        current = Section(level, name, anchor, [*parent_path, name])
        body.append((f"{m[1]} {heading}", None))  # keep the heading, minus {#anchor}
        stack.append(current)
    close_current()
    return sections


def assign_feature_states(sections: list[Section], states: list[dict]) -> int:
    """Attach each marker's state to its section and all descendant sections.

    Returns the number of markers found, which should equal len(states)."""
    markers = 0
    by_path: dict[tuple, list[dict]] = {}
    for section in sections:
        own = []
        for block in section.blocks:
            for _ in FEATURE_STATE_RE.finditer(block.text):
                if markers < len(states):
                    own.append(states[markers])
                markers += 1
        inherited = []
        for depth in range(1, len(section.path)):
            inherited += by_path.get(tuple(section.path[:depth]), [])
        by_path[tuple(section.path)] = by_path.get(tuple(section.path), []) + own
        section.feature_states = dedupe_states(inherited + own)
    return markers


def dedupe_states(states: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in states:
        key = json.dumps(s, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# ---------------------------------------------------------------------------- chunking


@dataclass
class Piece:
    text: str
    tokens: int
    section: Section


def split_block(block: Block, limit: int) -> list[str]:
    """Split one oversized block by lines (code is re-fenced), then by words if needed."""
    if block.is_code:
        lines = block.text.splitlines()
        opening, body = lines[0], lines[1:-1]
        closing = lines[-1]
        overhead = count_tokens(f"{opening}\n{closing}") + 2
        return [f"{opening}\n{part}\n{closing}" for part in pack_lines(body, limit - overhead)]
    return pack_lines(items(block.text.splitlines()), limit)


def items(lines: list[str]) -> list[str]:
    """Join indented continuation lines to the line they continue (a list item, or a flag
    and its description), so a split never separates them."""
    out: list[str] = []
    for line in lines:
        if out and line[:1].isspace():
            out[-1] += "\n" + line
        else:
            out.append(line)
    return out


def pack_lines(lines: list[str], limit: int) -> list[str]:
    parts, buf, size = [], [], 0
    for line in lines:
        n = count_tokens(line) + 1
        if n > limit:  # a huge item: its own lines, then word windows
            if buf:
                parts.append("\n".join(buf))
                buf, size = [], 0
            sub = line.splitlines()
            parts += pack_lines(sub, limit) if len(sub) > 1 else pack_words(line, limit)
            continue
        if size + n > limit and buf:
            parts.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += n
    if buf:
        parts.append("\n".join(buf))
    return parts


def pack_words(line: str, limit: int) -> list[str]:
    tokens = _encoding().encode(line, disallowed_special=())
    return [_encoding().decode(tokens[i : i + limit]) for i in range(0, len(tokens), limit)]


def section_pieces(section: Section, max_tokens: int) -> list[Piece]:
    pieces, buf, size = [], [], 0

    def flush():
        nonlocal buf, size
        if buf:
            text = "\n\n".join(buf)
            pieces.append(Piece(text, count_tokens(text), section))
        buf, size = [], 0

    for block in section.blocks:
        n = count_tokens(block.text) + 2
        if n > max_tokens:
            flush()
            for part in split_block(block, max_tokens):
                pieces.append(Piece(part, count_tokens(part), section))
            continue
        if size + n > max_tokens:
            flush()
        buf.append(block.text)
        size += n
    flush()
    return pieces


@dataclass
class Chunk:
    pieces: list[Piece]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pieces)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


def merge_small(pieces: list[Piece], min_tokens: int, max_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: Chunk | None = None
    for piece in pieces:
        if pending is not None:
            candidate = Chunk([*pending.pieces, piece])
            if candidate.tokens <= max_tokens:
                pending = candidate
                if pending.tokens >= min_tokens:
                    chunks.append(pending)
                    pending = None
                continue
            chunks.append(pending)
            pending = None
        chunk = Chunk([piece])
        if piece.tokens < min_tokens:
            pending = chunk
        else:
            chunks.append(chunk)
    if pending is not None:
        # A small tail joins the previous chunk when it fits, else stands alone.
        if chunks and Chunk([*chunks[-1].pieces, *pending.pieces]).tokens <= max_tokens:
            chunks[-1] = Chunk([*chunks[-1].pieces, *pending.pieces])
        else:
            chunks.append(pending)
    return chunks


def chunk_record(record: dict, max_tokens: int = MAX_TOKENS, min_tokens: int = MIN_TOKENS):
    sections = parse_sections(record["title"], record["text"])
    states = record.get("feature_states") or []
    markers = assign_feature_states(sections, states)
    pieces = [p for s in sections for p in section_pieces(s, max_tokens)]
    chunks = merge_small(pieces, min_tokens, max_tokens)

    base_url, _, record_anchor = record["url"].partition("#")
    out = []
    for i, chunk in enumerate(chunks):
        first = chunk.pieces[0].section
        anchor = first.anchor or record_anchor
        states_in_chunk = dedupe_states([s for p in chunk.pieces for s in p.section.feature_states])
        text = chunk.text
        out.append(
            {
                "chunk_id": f"{record['id']}::{i}",
                "record_id": record["id"],
                "version": record["version"],
                "kind": record["kind"],
                "url": f"{base_url}#{anchor}" if anchor else base_url,
                "title": record["title"],
                "heading_path": first.path,
                "anchor": anchor,
                "text": text,
                "n_tokens": count_tokens(text),
                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "feature_states": states_in_chunk,
                "content_type": record.get("content_type", ""),
                "source_path": record["source_path"],
                "commit": record.get("commit", ""),
            }
        )
    return out, {"feature_state_markers": markers, "feature_states": len(states)}


# ---------------------------------------------------------------------------- driver


def percentiles(xs: list[int]) -> dict[str, int]:
    if not xs:
        return {}
    xs = sorted(xs)
    pick = lambda p: xs[min(len(xs) - 1, int(len(xs) * p / 100))]  # noqa: E731
    return {
        "p50": pick(50),
        "p90": pick(90),
        "p99": pick(99),
        "max": xs[-1],
        "mean": round(statistics.mean(xs)),
    }


def chunk_version(records: list[dict], max_tokens: int, min_tokens: int):
    chunks, mismatched = [], []
    for record in records:
        out, check = chunk_record(record, max_tokens, min_tokens)
        chunks += out
        if check["feature_state_markers"] != check["feature_states"]:
            mismatched.append(record["id"])
    tokens = [c["n_tokens"] for c in chunks]
    report = {
        "records": len(records),
        "chunks": len(chunks),
        "chunks_by_kind": dict(Counter(c["kind"] for c in chunks)),
        "tokens": percentiles(tokens),
        "total_tokens": sum(tokens),
        "over_max": sum(t > max_tokens for t in tokens),
        "under_min": sum(t < min_tokens for t in tokens),
        "unique_content_hashes": len({c["content_hash"] for c in chunks}),
        "feature_state_mismatches": mismatched,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
    }
    return chunks, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--versions", nargs="+", help="default: versions in the lock file")
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    args = parser.parse_args(argv)
    versions = args.versions or list(load_lock(LOCK_PATH)["sources"])

    for version in versions:
        src = args.clean_dir / f"v{version}" / "pages.jsonl"
        records = [json.loads(line) for line in src.read_text().splitlines()]
        chunks, report = chunk_version(records, args.max_tokens, args.min_tokens)
        report = {"version": version, **report}
        out = args.out_dir / f"v{version}"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "chunks.jsonl").open("w") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{version}: {report['chunks']} chunks from {report['records']} records, "
            f"tokens {report['tokens']}, over max {report['over_max']}, "
            f"under min {report['under_min']}, unique hashes {report['unique_content_hashes']}"
        )


if __name__ == "__main__":
    main()
