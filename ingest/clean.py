"""Turn raw Hugo docs into clean markdown records, one JSONL file per version.

Reads data/raw/v<version>/ (from ingest.fetch) and writes data/clean/v<version>/:
  pages.jsonl          one record per rendered page, glossary term and feature gate
  feature_gates.jsonl  structured feature gate stage history (the version metadata asset)
  report.json          counts, unknown shortcodes, missing files, leftover shortcode tags

Shortcodes are rendered the way the site's templates in layouts/shortcodes/ render them,
reduced to plain markdown. Labels (Note:, What's next, ...) come from i18n/en/en.toml.

Usage:
    uv run python -m ingest.clean                  # all versions in ingest/sources.lock.json
    uv run python -m ingest.clean --versions 1.37
"""

import argparse
import json
import os
import re
import subprocess
import textwrap
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ingest.fetch import DATA_DIR as RAW_DIR
from ingest.fetch import DOCS_PATH, EXAMPLES_PATH, INCLUDES_PATH, LOCK_PATH, load_lock
from ingest.html2md import convert as html_to_markdown
from ingest.html2md import count_tags as count_html_tags
from ingest.shortcodes import Node, Shortcode, parse

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"

# Pages about contributing to the docs site, not about Kubernetes.
EXCLUDED_PREFIXES = ("contribute/", "doc-contributor-tools/")
EXCLUDED_PAGES = {"test.md"}
GLOSSARY_DIR = "reference/glossary/"
FEATURE_GATES_DIR = "reference/command-line-tools-reference/feature-gates/"
FEATURE_GATES_URL = "/docs/reference/command-line-tools-reference/feature-gates/"

# Rendered as nothing: interactive widgets, diagrams, and lists we cover with our own
# records (feature-gate-table/list are rebuilt from the per-gate records).
DROPPED = {
    "comment",
    "mermaid",
    "thirdparty-content",
    "cve-feed",
    "doc-versions-list",
    "cncf-landscape",
    "kat-button",
    "page-api-reference",
    "feature-gate-table",
    "feature-gate-list",
    "tutorials/carousel",
    "tutorials/carousel-item",
}

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
LEFTOVER_RE = re.compile(r"\{\{[<%]")


@dataclass
class GlossaryTerm:
    id: str
    title: str
    full_link: str
    aka: list[str]
    body: str  # raw markdown, may contain shortcodes


@dataclass
class FeatureGate:
    name: str
    stages: list[dict]
    removed: bool
    description: str = ""  # rendered markdown, filled after loading

    @property
    def current(self) -> dict:
        return self.stages[-1]


@dataclass
class Site:
    """Everything a shortcode may need that isn't on the page itself."""

    root: Path
    version: str  # "1.36"
    params: dict
    i18n: dict[str, str]
    glossary: dict[str, GlossaryTerm] = field(default_factory=dict)
    gates: dict[str, FeatureGate] = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)

    @property
    def latest(self) -> str:
        return self.params["latest"].removeprefix("v")

    @property
    def base_url(self) -> str:
        if self.version == self.latest:
            return "https://kubernetes.io"
        return f"https://v{self.version.replace('.', '-')}.docs.kubernetes.io"

    @property
    def patch_version(self) -> str:
        for v in self.params.get("versions", []):
            if v["version"] == f"v{self.version}" and v["githubbranch"].startswith("v"):
                return v["githubbranch"].removeprefix("v")
        return f"{self.version}.0"

    def t(self, key: str) -> str:
        return self.i18n.get(key, key)

    def warn(self, kind: str, detail: str) -> None:
        self.stats[f"{kind}: {detail}"] += 1


@dataclass
class Page:
    site: Site
    path: Path  # markdown file
    front: dict
    feature_states: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------- helpers


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    front = yaml.safe_load(text[3:end]) or {}
    return front, text[end + 4 :].lstrip("\n")


def load_i18n(path: Path) -> dict[str, str]:
    data = tomllib.loads(path.read_text())
    return {k: v.get("other", "") for k, v in data.items() if isinstance(v, dict)}


def outside_fences(text: str, fn) -> str:
    """Apply fn to the parts of text that are not inside fenced code blocks."""
    out, buf, in_fence = [], [], False
    for line in text.splitlines(keepends=True):
        if FENCE_RE.match(line):
            if not in_fence:
                out.append(fn("".join(buf)))
                buf = []
            else:
                out.append("".join(buf))
                buf = []
            out.append(line)
            in_fence = not in_fence
            continue
        buf.append(line)
    out.append("".join(buf) if in_fence else fn("".join(buf)))
    return "".join(out)


def tidy(text: str) -> str:
    text = outside_fences(text, lambda s: html_to_markdown(HTML_COMMENT_RE.sub("", s)))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def prose(text: str) -> str:
    """Text with fenced code blocks (and their fence lines) removed."""
    kept, in_fence = [], False
    for line in text.splitlines(keepends=True):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            kept.append(line)
    return "".join(kept)


def url_path(rel: Path, front: dict) -> str:
    """content/en-relative markdown path -> site URL path, e.g. /docs/concepts/pods/."""
    parts = list(rel.parts)
    name = parts.pop()
    if name not in ("_index.md", "index.md"):
        parts.append(Path(name).stem)
    if front.get("slug") and name not in ("_index.md", "index.md"):
        parts[-1] = str(front["slug"])
    return "/" + "/".join(parts) + "/" if parts else "/"


def resolve_ref(page: Page, target: str) -> str:
    target, _, anchor = target.partition("#")
    content = page.site.root / "content/en"
    if target.startswith("/"):
        path = content / target.lstrip("/")
    else:
        path = Path(os.path.normpath(page.path.parent / target))
    md = path.parent / f"{path.name}.md"  # not with_suffix: names like kubelet-config.v1beta1
    candidates = [path, md, path / "_index.md", path / "index.md"]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        page.site.warn("unresolved ref", target)
        rel = path.relative_to(content) if path.is_relative_to(content) else Path(target)
        url = "/" + str(rel).strip("/") + "/"
    else:
        url = url_path(found.relative_to(content), split_front_matter(found.read_text())[0])
    return url + (f"#{anchor}" if anchor else "")


def fenced(code: str, lang: str) -> str:
    code = textwrap.dedent(code).strip("\n").rstrip()
    return f"\n```{lang}\n{code}\n```\n"


# ---------------------------------------------------------------------------- rendering


def render(nodes: list[Node], page: Page) -> str:
    return "".join(n if isinstance(n, str) else render_shortcode(n, page) for n in nodes)


def render_markdown(text: str, page: Page) -> str:
    return render(parse(text), page)


def render_shortcode(sc: Shortcode, page: Page) -> str:
    site = page.site
    site.stats[f"shortcode {sc.name}"] += 1
    if sc.name in DROPPED:
        return ""
    handler = HANDLERS.get(sc.name)
    if handler is None:
        site.warn("unknown shortcode", sc.name)
        return render(sc.inner, page) if sc.inner else ""
    return handler(sc, page, lambda: render(sc.inner or [], page))


def callout(label: str, body: str) -> str:
    # No surrounding newlines: the tags sit on their own (possibly indented) source lines.
    return f"**{label}** {body.strip()}"


def h_note(sc, page, inner):
    return callout(page.site.t(sc.name), inner())


def h_alert(sc, page, inner):
    title = sc.get("title")
    if not title:
        return f"\n{inner().strip()}\n"
    return callout(title if title.endswith(":") else f"{title}:", inner())


def h_heading(sc, page, inner):
    return page.site.t(f"{sc.get(0)}_heading")


def glossary_term(page: Page, term_id: str) -> GlossaryTerm | None:
    # Hugo's Resources.GetMatch is case-insensitive: term_id="StatefulSet" -> statefulset.md
    term = page.site.glossary.get(term_id.lower())
    if term is None:
        page.site.warn("unknown glossary term", term_id)
    return term


def h_glossary_tooltip(sc, page, inner):
    if sc.get("text"):
        return sc.get("text")
    term = glossary_term(page, sc.get("term_id"))
    return term.title if term else sc.get("term_id")


def h_glossary_definition(sc, page, inner):
    term = glossary_term(page, sc.get("term_id"))
    if term is None:
        return ""
    body = tidy(render_markdown(term.body, page)).strip()
    paragraphs = re.split(r"\n\s*\n", body)
    first = paragraphs[0]
    if prepend := sc.get("prepend"):
        paragraphs[0] = f"{prepend} {first[:1].lower()}{first[1:]}"
    text = paragraphs[0] if sc.get("length") == "short" else "\n\n".join(paragraphs)
    return f"\n{text}\n"


def h_feature_state(sc, page, inner):
    site = page.site
    if name := sc.get("feature_gate_name"):
        gate = site.gates.get(name)
        if gate is None:
            site.warn("unknown feature gate", name)
            return ""
        stage = gate.current
        state, since = stage["stage"], str(stage["fromVersion"])
        default = bool(stage.get("defaultValue"))
        page.feature_states.append(
            {"feature_gate": name, "state": state, "since": since, "default_enabled": default}
        )
        enabled = site.t("feature_gate_enabled" if default else "feature_gate_disabled")
        return f"{site.t('feature_state')} `Kubernetes v{since} [{state}]` {enabled}"

    state = sc.get("state")
    since = (sc.get("for_k8s_version") or f"v{site.version}").removeprefix("v")
    page.feature_states.append({"feature_gate": None, "state": state, "since": since})
    return f"{site.t('feature_state')} `Kubernetes v{since} [{state}]`"


def h_skew(sc, page, inner):
    """Mirror layouts/shortcodes/skew.html: most variants count from site `latest`."""
    site = page.site
    major, latest_minor = (int(x) for x in site.latest.split("."))
    current_minor = int(site.version.split(".")[1])
    which = sc.get(0)
    match which:
        case "currentVersion":
            return site.version
        case "currentPatchVersion":
            return site.patch_version
        case "latestVersion":
            return site.latest
        case "nextMinorVersion":
            return f"{major}.{latest_minor + 1}"
        case "prevMinorVersion":
            return f"{major}.{latest_minor - 1}"
        case "oldestMinorVersion":
            return f"{major}.{latest_minor - 2}"
        case "currentVersionAddMinor" | "latestVersionAddMinor":
            base = current_minor if which.startswith("current") else latest_minor
            return f"{major}{sc.get(2) or '.'}{base + int(sc.get(1))}"
    site.warn("unknown skew variant", which)
    return ""


def h_param(sc, page, inner):
    key = sc.get(0)
    value = page.front.get(key, page.site.params.get(key))
    if value is None:
        page.site.warn("unknown param", key)
        return ""
    return str(value)


def h_version_check(sc, page, inner):
    site = page.site
    min_version = page.front.get("min-kubernetes-server-version")
    text = ""
    if min_version == site.params["version"]:
        text = f"{site.t('version_check_mustbe')}{min_version}. "
    elif min_version:
        text = f"{site.t('version_check_mustbeorlater')}{min_version}. "
    return f"{text}{site.t('version_check_tocheck').strip()} `kubectl version`."


def h_include(sc, page, inner):
    name = sc.get(0)
    includes = page.site.root / INCLUDES_PATH
    matches = sorted(includes.glob(f"{name}*"))
    if not matches:
        base = page.path.parent / name
        matches = [p for p in (base, base.with_suffix(".md"), base / "_index.md") if p.is_file()]
    if not matches:
        page.site.warn("missing include", name)
        return ""
    return include_file(matches[0], page)


def include_file(path: Path, page: Page) -> str:
    _, body = split_front_matter(path.read_text())
    return "\n" + render_markdown(body, page) + "\n"


def h_code_sample(sc, page, inner):
    file = sc.get("file").lstrip("/")
    path = page.site.root / EXAMPLES_PATH / file
    if not path.is_file():
        page.site.warn("missing example", file)
        return ""
    lang = sc.get("language") or path.suffix.lstrip(".")
    return f"\nExample file `{file}`:\n{fenced(path.read_text(), lang)}"


def h_highlight(sc, page, inner):
    if "hl_inline=true" in sc.get(1):
        return f"`{inner().strip()}`"
    return fenced(inner(), sc.get(0))


def h_tab(sc, page, inner):
    name = sc.get("name").strip()
    if include := sc.get("include").strip():
        path = page.path.parent / include
        if not path.is_file():
            page.site.warn("missing tab include", include)
            return ""
        body = include_file(path, page)
    elif lang := sc.get("codelang"):
        body = fenced(inner(), lang)
    else:
        body = inner()
    return f"**{name}**\n\n{body.strip()}\n"


def h_details(sc, page, inner):
    summary = sc.get("summary")
    return f"\n**{summary}**\n\n{inner().strip()}\n" if summary else inner()


def h_table(sc, page, inner):
    caption = sc.get("caption")
    return (f"\nTable: {caption}\n" if caption else "") + inner()


def h_ref(sc, page, inner):
    return resolve_ref(page, sc.get(0))


def h_api_reference(sc, page, inner):
    target = f"/docs/reference/kubernetes-api/{sc.get('page')}"
    url = resolve_ref(page, target)
    anchor = sc.get("anchor")
    text = sc.get("text") or anchor
    if not text:
        path = page.site.root / "content/en" / (target.lstrip("/") + ".md")
        meta = split_front_matter(path.read_text())[0] if path.is_file() else {}
        text = meta.get("api_metadata", {}).get("kind") or sc.get("page").rsplit("/", 1)[-1]
    return f"[{text}]({url}{f'#{anchor}' if anchor else ''})"


def h_link(sc, page, inner):
    return f"[{sc.get('text')}]({sc.get('url')})"


def h_figure(sc, page, inner):
    text = " ".join(t for t in (sc.get("title"), sc.get("caption") or sc.get("alt")) if t)
    return f"\n*Figure: {html_to_markdown(text).strip()}*\n" if text else ""


def h_message(key):
    def handler(sc, page, inner):
        return callout(page.site.t("note"), html_to_markdown(page.site.t(key)).strip())

    return handler


def h_feature_gate_description(sc, page, inner):
    gate = page.site.gates.get(sc.get("name"))
    if gate is None:
        page.site.warn("unknown feature gate", sc.get("name"))
        return ""
    return f"`{gate.name}`: {gate.description.strip()}"


HANDLERS = {
    "note": h_note,
    "caution": h_note,
    "warning": h_note,
    "alert": h_alert,
    "pageinfo": lambda sc, page, inner: inner(),
    "heading": h_heading,
    "glossary_tooltip": h_glossary_tooltip,
    "glossary_definition": h_glossary_definition,
    "feature-state": h_feature_state,
    "skew": h_skew,
    "param": h_param,
    "version-check": h_version_check,
    "latest-version": lambda sc, page, inner: page.site.params["latest"],
    "latest-semver": lambda sc, page, inner: page.site.latest,
    "release-branch": lambda sc, page, inner: f"release-{page.site.latest}",
    "latest-release-notes": lambda sc, page, inner: (
        f"https://git.k8s.io/kubernetes/CHANGELOG/CHANGELOG-{page.site.latest}.md"
    ),
    "include": h_include,
    "code_sample": h_code_sample,
    "code": h_code_sample,
    "highlight": h_highlight,
    "tabs": lambda sc, page, inner: inner(),
    "tab": h_tab,
    "details": h_details,
    "table": h_table,
    "example": lambda sc, page, inner: inner() or sc.get("file"),
    "ref": h_ref,
    "relref": h_ref,
    "api-reference": h_api_reference,
    "link": h_link,
    "figure": h_figure,
    "dockershim-removal": h_message("dockershim_message"),
    "legacy-repos-deprecation": h_message("legacy_repos_message"),
    "feature-gate-description": h_feature_gate_description,
    "tutorials/modules": lambda sc, page, inner: inner(),
    "tutorials/module": lambda sc, page, inner: f"- [{sc.get('title')}]({sc.get('path')})",
}


# ---------------------------------------------------------------------------- records


def stage_line(stage: dict) -> str:
    since, until = stage["fromVersion"], stage.get("toVersion")
    span = f"{since}–{until}" if until and until != since else (since if until else f"{since}+")
    default = "enabled" if stage.get("defaultValue") else "disabled"
    locked = ", locked" if stage.get("locked") else ""
    return f"- {stage['stage']} ({default} by default{locked}): {span}"


def feature_gate_text(gate: FeatureGate, version: str) -> str:
    lines = [f"# Feature gate: {gate.name}", ""]
    stage = gate.current
    if gate.removed:
        lines.append(f"The `{gate.name}` feature gate has been removed from Kubernetes.")
    else:
        default = "enabled" if stage.get("defaultValue") else "disabled"
        lines.append(
            f"In Kubernetes v{version}, the `{gate.name}` feature gate is at the "
            f"{stage['stage']} stage and is {default} by default."
        )
    lines += ["", "Stage history:", *(stage_line(s) for s in gate.stages), ""]
    lines.append(gate.description.strip())
    return "\n".join(lines).strip() + "\n"


def record(site: Site, kind: str, rel: Path, url: str, title: str, text: str, **extra) -> dict:
    return {
        "id": f"v{site.version}:{url}",
        "version": site.version,
        "kind": kind,
        "source_path": str(Path(DOCS_PATH) / rel),
        "url": site.base_url + url,
        "title": title,
        "text": text,
        **extra,
    }


def load_site(raw: Path, version: str) -> Site:
    raw = raw.resolve()
    params = tomllib.loads((raw / "hugo.toml").read_text())["params"]
    site = Site(raw, version, params, load_i18n(raw / "i18n/en/en.toml"))
    docs = raw / DOCS_PATH
    for path in sorted((docs / GLOSSARY_DIR).glob("*.md")):
        if path.name == "index.md":
            continue
        front, body = split_front_matter(path.read_text())
        site.glossary[path.stem.lower()] = GlossaryTerm(
            id=path.stem,
            title=front["title"],
            full_link=front.get("full_link") or "",
            aka=[a for a in (front.get("aka") or []) if a],
            body=body.replace("<!--more-->", ""),
        )
    raw_gates = {}
    for path in sorted((docs / FEATURE_GATES_DIR).glob("*.md")):
        if path.name == "index.md":
            continue
        front, body = split_front_matter(path.read_text())
        stages = front.get("stages") or []
        if not stages:
            site.warn("feature gate without stages", path.stem)
            continue
        for s in stages:
            s["stage"] = str(s["stage"]).strip()
            s["fromVersion"] = str(s["fromVersion"])
            if s.get("toVersion") is not None:
                s["toVersion"] = str(s["toVersion"])
        gate = FeatureGate(front["title"], stages, removed=bool(front.get("removed")))
        site.gates[gate.name] = gate
        raw_gates[gate.name] = (path, front, body)
    # Descriptions may use shortcodes, including other gates' feature-state.
    for name, (path, front, body) in raw_gates.items():
        page = Page(site, path, front)
        site.gates[name].description = tidy(render_markdown(body, page))
    return site


def clean_version(raw: Path, version: str) -> tuple[list[dict], list[FeatureGate], dict]:
    site = load_site(raw, version)
    docs = site.root / DOCS_PATH
    content = site.root / "content/en"
    records, skipped = [], Counter()

    for path in sorted(docs.rglob("*.md")):
        rel = path.relative_to(docs)
        rel_str = rel.as_posix()
        if rel_str.startswith(EXCLUDED_PREFIXES) or rel_str in EXCLUDED_PAGES:
            skipped["excluded section"] += 1
            continue
        if rel_str.startswith((GLOSSARY_DIR, FEATURE_GATES_DIR)) and path.name != "index.md":
            continue  # emitted as glossary / feature gate records below
        if rel_str == GLOSSARY_DIR + "index.md":
            continue  # renders the glossary terms, which we emit individually
        front, body = split_front_matter(path.read_text())
        build = front.get("_build") or {}
        if front.get("headless") or build.get("render") in (False, "never") or front.get("draft"):
            skipped["not rendered"] += 1
            continue
        page = Page(site, path, front)
        text = tidy(render_markdown(body, page))
        if not text.strip():
            skipped["empty"] += 1
            continue
        records.append(
            record(
                site,
                "page",
                rel,
                url_path(path.relative_to(content), front),
                str(front.get("title", "")),
                text,
                description=str(front.get("description") or "").strip(),
                content_type=str(front.get("content_type") or "").strip().strip('"'),
                feature_states=page.feature_states,
                min_server_version=front.get("min-kubernetes-server-version"),
            )
        )

    for term in site.glossary.values():
        page = Page(site, docs / GLOSSARY_DIR / f"{term.id}.md", {})
        body = tidy(render_markdown(term.body, page))
        aka = f"\nAlso known as: {', '.join(term.aka)}.\n" if term.aka else ""
        records.append(
            record(
                site,
                "glossary",
                Path(GLOSSARY_DIR) / f"{term.id}.md",
                f"/docs/reference/glossary/?all=true#term-{term.id}",
                term.title,
                f"# {term.title}\n\n{body}{aka}",
                full_link=term.full_link,
            )
        )

    for gate in site.gates.values():
        records.append(
            record(
                site,
                "feature_gate",
                Path(FEATURE_GATES_DIR) / f"{gate.name}.md",
                f"{FEATURE_GATES_URL}#{gate.name}",
                gate.name,
                feature_gate_text(gate, version),
                feature_gate=gate.name,
                removed=gate.removed,
            )
        )

    leftovers = sum(len(LEFTOVER_RE.findall(r["text"])) for r in records)
    html_tags = sum(count_html_tags(prose(r["text"])) for r in records)
    report = {
        "version": version,
        "records": dict(Counter(r["kind"] for r in records)),
        "skipped": dict(skipped),
        "leftover_shortcode_tags": leftovers,
        "leftover_html_tags": html_tags,
        "warnings": {k: v for k, v in sorted(site.stats.items()) if not k.startswith("shortcode ")},
        "shortcodes": {
            k.removeprefix("shortcode "): v
            for k, v in site.stats.most_common()
            if k.startswith("shortcode ")
        },
    }
    return records, list(site.gates.values()), report


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--versions", nargs="+", help="default: versions in the lock file")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=CLEAN_DIR)
    args = parser.parse_args(argv)
    versions = args.versions or list(load_lock(LOCK_PATH)["sources"])

    for version in versions:
        raw = args.raw_dir / f"v{version}"
        records, gates, report = clean_version(raw, version)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=raw, capture_output=True, text=True
        ).stdout.strip()
        for r in records:
            r["commit"] = commit
        out = args.out_dir / f"v{version}"
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "pages.jsonl", records)
        write_jsonl(out / "feature_gates.jsonl", (asdict(g) for g in gates))
        report["commit"] = commit
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        warnings = sum(report["warnings"].values())
        print(
            f"{version}: {report['records']}, skipped {report['skipped']}, "
            f"{warnings} warnings, {report['leftover_shortcode_tags']} leftover shortcode tags, "
            f"{report['leftover_html_tags']} leftover HTML tags -> {out}"
        )


if __name__ == "__main__":
    main()
