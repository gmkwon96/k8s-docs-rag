"""Work out which Kubernetes version a question is about."""

import re
from dataclasses import dataclass
from functools import lru_cache

from ingest.dedupe import version_key
from ingest.fetch import LOCK_PATH, load_lock

# "1.36", "v1.36", "1.36.2", "Kubernetes 1.36"; two-digit minors only, to skip "1.5 seconds"
VERSION_RE = re.compile(r"(?<![\d.])v?1\.(\d{2})(?:\.\d+)?(?![\d.]*\d)")


@lru_cache(maxsize=1)
def indexed_versions() -> tuple[str, ...]:
    return tuple(sorted(load_lock(LOCK_PATH)["sources"], key=version_key))


@dataclass
class VersionChoice:
    version: str  # the version we search and answer for
    requested: str | None  # what the question or caller asked for, if anything
    note: str = ""  # shown to the model and the user when we had to substitute


def mentioned_versions(question: str) -> list[str]:
    return [f"1.{m}" for m in VERSION_RE.findall(question)]


def choose_version(
    question: str, explicit: str | None = None, versions: tuple[str, ...] | None = None
) -> VersionChoice:
    versions = versions or indexed_versions()
    latest = versions[-1]
    mentioned = mentioned_versions(question)
    requested = explicit or (mentioned[0] if mentioned else None)
    if requested is None:
        return VersionChoice(latest, None)
    if requested in versions:
        return VersionChoice(requested, requested)
    note = (
        f"Documentation for Kubernetes {requested} is not indexed "
        f"(indexed: {', '.join(versions)}); answering from {latest}."
    )
    return VersionChoice(latest, requested, note)
