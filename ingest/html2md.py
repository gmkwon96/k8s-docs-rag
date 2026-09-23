"""Convert the raw HTML embedded in Kubernetes docs markdown into plain markdown.

Generated reference pages (API fields, component flags, metrics, config APIs) are HTML
tables and lists inside markdown. Only tags in HTML_TAGS are touched, so command
placeholders such as `<pod-name>` or `<path>` in prose survive. Run it on text outside
fenced code blocks.
"""

import html
import re
from html.parser import HTMLParser

HTML_TAGS = {
    "a", "b", "blockquote", "br", "caption", "code", "col", "colgroup", "dd", "div", "dl",
    "dt", "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "label", "li", "ol", "p",
    "pre", "small", "span", "strong", "sub", "sup", "table", "tbody", "td", "tfoot", "th",
    "thead", "tr", "tt", "u", "ul",
}  # fmt: skip
TAG_NAMES = "|".join(sorted(HTML_TAGS, key=len, reverse=True))
ANY_TAG_RE = re.compile(rf"</?(?:{TAG_NAMES})(?=[\s/>])[^<>]*>", re.I)
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.I | re.S)


def _sub(pattern: str, repl, text: str) -> str:
    return re.sub(pattern, repl, text, flags=re.I | re.S)


def inline(fragment: str, *, in_cell: bool = False) -> str:
    """Convert inline and simple block HTML to markdown. Entities are decoded last."""
    s = fragment
    # API reference marks required fields with a bold asterisk.
    s = _sub(r"(?:&nbsp;)?<strong>\*</strong>", " (required)", s)
    s = _sub(r"<pre\b[^>]*>(.*?)</pre\s*>", lambda m: f"\n```\n{m[1].strip()}\n```\n", s)
    s = _sub(r"<(code|tt)\b[^>]*>(.*?)</\1\s*>", lambda m: f"`{strip_tags(m[2])}`", s)
    s = _sub(r"<(strong|b)\b[^>]*>(.*?)</\1\s*>", lambda m: f"**{m[2].strip()}**", s)
    s = _sub(r"<(em|i)\b[^>]*>(.*?)</\1\s*>", lambda m: f"*{m[2].strip()}*", s)
    s = _sub(r'<a\b[^>]*\bhref="([^"]*)"[^>]*>(.*?)</a\s*>', lambda m: link(m[2], m[1]), s)
    s = _sub(r'<img\b[^>]*\balt="([^"]*)"[^>]*>', lambda m: m[1], s)
    s = _sub(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", lambda m: f"\n\n{'#' * int(m[1])} {m[2]}\n\n", s)
    if in_cell:
        s = _sub(r"<br\s*/?>|</?p\b[^>]*>", " ", s)
    else:
        s = _sub(r"<br\s*/?>", "\n", s)
        s = _sub(r"</?(p|div|blockquote|dl|hr)\b[^>]*>", "\n\n", s)
    s = _sub(r"<(li|dt)\b[^>]*>", "\n- ", s)
    s = _sub(r"<dd\b[^>]*>", ": ", s)
    s = _sub(r"</label\s*>", " ", s)
    s = strip_tags(s)
    s = html.unescape(s).replace("\xa0", " ")
    if in_cell:
        s = re.sub(r"\s+", " ", s)
    return s.strip() if in_cell else s


def link(text: str, url: str) -> str:
    text = text.strip()
    return f"[{text}]({url})" if text else ""


def strip_tags(s: str) -> str:
    return ANY_TAG_RE.sub("", s)


class _TableParser(HTMLParser):
    """Collect rows of raw cell HTML; nested tables are kept as raw cell content."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.rows: list[tuple[list[str], bool]] = []  # (cells, is_header_row)
        self.caption = ""
        self._depth = 0
        self._cell: list[str] | None = None
        self._row: list[str] | None = None
        self._row_header = False
        self._in_caption = False

    def handle_starttag(self, tag, attrs):
        raw = self.get_starttag_text()
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                return
        if self._depth > 1 or (self._cell is not None and tag not in ("td", "th", "tr")):
            self._append(raw)
        elif tag == "tr":
            self._row, self._row_header = [], False
        elif tag in ("td", "th"):
            self._cell = []
            self._row_header |= tag == "th"
        elif tag == "caption":
            self._in_caption = True

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth -= 1
            if self._depth >= 1:
                self._append(f"</{tag}>")
            return
        if self._depth > 1 or (self._cell is not None and tag not in ("td", "th", "tr")):
            self._append(f"</{tag}>")
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is None:
                self._row = []
            self._row.append("".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append((self._row, self._row_header))
            self._row = None
        elif tag == "caption":
            self._in_caption = False

    def handle_startendtag(self, tag, attrs):
        if self._cell is not None or self._depth > 1:
            self._append(self.get_starttag_text())

    def handle_data(self, data):
        if self._in_caption:
            self.caption += data
        else:
            self._append(data)

    def handle_entityref(self, name):
        self.handle_data(f"&{name};")

    def handle_charref(self, name):
        self.handle_data(f"&#{name};")

    def _append(self, s: str):
        if self._cell is not None:
            self._cell.append(s)


def table(fragment: str) -> str:
    parser = _TableParser()
    parser.feed(fragment)
    parser.close()
    headers: list[str] = []
    lines = [f"Table: {inline(parser.caption, in_cell=True)}"] if parser.caption.strip() else []
    for raw_cells, is_header in parser.rows:
        cells = [cell_text(c) for c in raw_cells]
        if is_header and not headers:
            headers = cells
            continue
        filled = [c for c in cells if c]
        if not filled:
            continue
        if cells[0] == "" and len(filled) == 1 and lines:
            lines.append(f"  {filled[0]}")  # continuation row, e.g. a flag's description
        elif len(headers) == 2 and len(cells) == 2:
            lines.append(f"- {cells[0]}: {cells[1]}")
        elif headers and len(headers) == len(cells):
            lines.append(
                "- " + "; ".join(f"{h}: {c}" for h, c in zip(headers, cells, strict=True) if c)
            )
        else:
            lines.append("- " + " | ".join(filled))
    return "\n\n" + "\n".join(lines) + "\n\n"


def cell_text(raw: str) -> str:
    if TABLE_RE.search(raw):  # nested table: flatten onto one line
        raw = TABLE_RE.sub(lambda m: " ".join(table(m[0]).split()), raw)
    return inline(raw, in_cell=True)


def convert(text: str) -> str:
    """HTML in a markdown fragment (no fenced code inside) -> markdown."""
    if "<" not in text:
        return text
    text = TABLE_RE.sub(lambda m: table(m[0]), text)
    return inline(text)


def count_tags(text: str) -> int:
    return len(ANY_TAG_RE.findall(text))
