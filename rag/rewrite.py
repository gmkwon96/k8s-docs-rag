"""Query transformations for retrieval (experiment E7).

- rewrite: turn a user question into a search query in the documentation's own terms,
  without trusting the question's premise (a made-up field should be searched as the
  real concept it resembles)
- hyde: write a short passage in the style of the Kubernetes docs that would answer the
  question (Hypothetical Document Embeddings); the passage is embedded instead of the
  question

Both only change what is embedded for vector search; reranking still uses the original
question so the user's intent is preserved.
"""

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 400

REWRITE_SYSTEM = """\
You turn a user's question about Kubernetes into a search query for the official Kubernetes
documentation. Use the terms the documentation would use: resource kinds, field names,
kubectl subcommands, component names, feature names. Do not assume the question's premise
is true: if it names a field, flag or behavior that may not exist, search for the real
concept it resembles instead. Keep the Kubernetes version if the question gives one.
Reply with the query only, one line, no quotes."""

HYDE_SYSTEM = """\
You write a short passage (3-5 sentences) in the style of the official Kubernetes
documentation that would answer the user's question, using the terms the documentation
would use (resource kinds, field names, kubectl commands). It is only used to find the
real documentation, so it may be imperfect, but do not repeat a premise in the question
that the documentation would contradict. Reply with the passage only."""

SYSTEMS = {"rewrite": REWRITE_SYSTEM, "hyde": HYDE_SYSTEM}


def request(mode: str, question: str, version: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEMS[mode],
        "messages": [
            {"role": "user", "content": f"Kubernetes version: {version}\n\nQuestion: {question}"}
        ],
    }


def text_of(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text").strip()
