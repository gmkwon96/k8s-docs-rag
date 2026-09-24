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


@dataclass(frozen=True)
class LocalEmbedder:
    """An open-weights model run locally with sentence-transformers (experiment E3).

    Free to run, so there is no ledger. Needs the optional `local` dependency group:
    `uv sync --group local`. Queries get the model's retrieval instruction ("query" prompt);
    documents are embedded as-is, as the model card recommends. On Apple silicon the model
    runs in float16 on MPS (fp32 exhausted memory on a 16GB machine and was ~3x slower).
    """

    name: str
    model: str
    dim: int
    max_batch_texts: int = 256  # one embed() call; encode() splits it into batch_size
    max_batch_tokens: int = 100_000
    batch_size: int = 16

    @cached_property
    def st(self):
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.backends.mps.is_available():
            return SentenceTransformer(
                self.model, device="mps", model_kwargs={"torch_dtype": torch.float16}
            )
        return SentenceTransformer(self.model, device="cpu")

    def embed(self, texts: list[str], input_type: InputType) -> tuple[list[list[float]], int]:
        prompt = "query" if input_type == "query" else None
        vectors = self.st.encode(
            texts, prompt_name=prompt, normalize_embeddings=True, batch_size=self.batch_size
        )
        if self.st.device.type == "mps":
            import torch

            torch.mps.empty_cache()
        return vectors.astype("float32").tolist(), 0


EMBEDDERS = {
    e.name: e
    for e in [
        VoyageEmbedder("voyage-4", "voyage-4", 1024),
        LocalEmbedder("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B", 1024),
    ]
}
