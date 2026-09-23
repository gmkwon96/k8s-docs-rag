"""Golden set: questions with reference answers and chunking-independent evidence.

An item's evidence is a verbatim quote from the cleaned docs of its version plus the URL
(path and anchor) of the section it comes from. A retrieved chunk counts as relevant when
it contains one of the item's quotes, so the labels survive any change of chunking
(experiment E1) and can be re-validated whenever the docs are re-fetched.

Files: eval/dataset/dev.jsonl and eval/dataset/test.jsonl, one item per line. Tune on dev
only; score test once per milestone.
"""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "eval" / "dataset"
SPLITS = ("dev", "test")

Category = Literal["fact", "howto", "version", "multihop", "unanswerable", "false_premise"]
# Target share of each category (plan.md, "정답 세트")
CATEGORY_TARGETS: dict[str, float] = {
    "fact": 0.25,
    "howto": 0.20,
    "version": 0.20,
    "multihop": 0.15,
    "unanswerable": 0.15,
    "false_premise": 0.05,
}
Origin = Literal["community-paraphrased", "synthesized", "manual"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(pattern=r"^/docs/")  # version-independent path, with #anchor if any
    quote: str = Field(min_length=12)  # verbatim from data/clean/v<version>/pages.jsonl


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^(dev|test)-\d{3}$")
    split: Literal["dev", "test"]
    question: str = Field(min_length=10)
    version: str = Field(pattern=r"^1\.\d{2}$")
    category: Category
    answerable: bool
    reference_answer: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    origin: Origin
    notes: str = ""

    @model_validator(mode="after")
    def consistent(self):
        if not self.id.startswith(self.split):
            raise ValueError(f"id {self.id} does not match split {self.split}")
        if self.category == "unanswerable" and self.answerable:
            raise ValueError("unanswerable items must have answerable=false")
        if self.category not in ("unanswerable", "false_premise") and not self.answerable:
            raise ValueError(f"{self.category} items must be answerable")
        if self.answerable and not self.evidence:
            raise ValueError("answerable items need at least one evidence quote")
        if self.category == "unanswerable" and self.evidence:
            raise ValueError("unanswerable items have no evidence")
        return self


def normalize(text: str) -> str:
    """Whitespace-insensitive form used to match quotes against chunks and records."""
    return re.sub(r"\s+", " ", text).strip()


def contains_quote(text: str, quote: str) -> bool:
    return normalize(quote) in normalize(text)


def load(path: Path) -> list[Item]:
    if not path.exists():
        return []
    items = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(Item.model_validate(json.loads(line)))
        except Exception as e:
            raise ValueError(f"{path.name}:{n}: {e}") from e
    return items


def load_split(split: str, dataset_dir: Path = DATASET_DIR) -> list[Item]:
    return load(dataset_dir / f"{split}.jsonl")
