"""Embedding models, shared by indexing (documents) and retrieval (queries).

Each entry in EMBEDDERS is one row key in the embeddings table: a model plus the settings
that change its vectors (dimension, dtype). Adding a variant means adding an entry.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import voyageai

from rag.settings import get_settings
from rag.usage import USAGE_DIR, Ledger

# Our token counts (tiktoken o200k_base) run ~1% below Voyage's; estimate generously.
ESTIMATE_MARGIN = 1.10

InputType = Literal["document", "query"]


@dataclass(frozen=True)
class VoyageEmbedder:
    name: str  # key in the embeddings table
    model: str
    dim: int
    max_batch_texts: int = 1000
    max_batch_tokens: int = 250_000  # API limit is 320K for voyage-4; our counts are ~1% low

    @cached_property
    def ledger(self) -> Ledger:
        return Ledger(USAGE_DIR / "voyage.jsonl", get_settings().voyage_token_budget)

    @cached_property
    def client(self) -> voyageai.Client:
        key = get_settings().voyage_api_key
        if not key:
            raise RuntimeError("VOYAGE_API_KEY is not set (see .env.example)")
        return voyageai.Client(api_key=key, max_retries=5)

    def estimate_tokens(self, texts: list[str]) -> int:
        from ingest.chunk import count_tokens

        return int(sum(count_tokens(t) for t in texts) * ESTIMATE_MARGIN) + len(texts)

    def embed(self, texts: list[str], input_type: InputType) -> tuple[list[list[float]], int]:
        """Returns the vectors and the tokens billed. Refuses to exceed the token budget."""
        self.ledger.check(self.estimate_tokens(texts))
        result = self.client.embed(
            texts,
            model=self.model,
            input_type=input_type,
            output_dimension=self.dim,
            truncation=False,  # chunks are <= 512 tokens; fail loudly rather than truncate
        )
        self.ledger.record(self.model, result.total_tokens, input_type)
        return result.embeddings, result.total_tokens


EMBEDDERS = {e.name: e for e in [VoyageEmbedder("voyage-4", "voyage-4", 1024)]}
