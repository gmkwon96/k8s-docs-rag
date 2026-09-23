"""Answer a question from retrieved chunks with Claude, citing the chunks it used.

Each chunk is a plain-text document with citations enabled; its URL, version and feature
states go in the document's `context` (visible to the model, not citable). The response's
citations point back to documents by index, which we map to numbered sources.
"""

from dataclasses import dataclass, field

from rag.retrieve import Hit

MODEL = "claude-sonnet-5"
EFFORT = "medium"
MAX_TOKENS = 2000  # thinking + answer; also the output side of the worst-case cost check

NOT_FOUND = "I couldn't find this in the Kubernetes documentation."

SYSTEM = f"""You answer questions about Kubernetes using only the documentation excerpts \
provided with each question. They come from the official Kubernetes docs for one \
Kubernetes version, stated with the question.

- Base every statement on the excerpts and cite them. Do not add facts from memory.
- If the excerpts don't contain the answer, reply with exactly "{NOT_FOUND}" and then, in \
one sentence, point to the most relevant excerpt if any.
- If the question assumes something the excerpts contradict (a field, flag or behavior \
that doesn't exist), say so plainly instead of answering as if it were true.
- Answer for the stated version. When an excerpt carries a feature state (alpha, beta, \
stable, deprecated), mention it where it matters.
- Be concise: lead with the answer, then the details or commands that support it.
- Answer in English."""


@dataclass
class Source:
    number: int
    hit: Hit
    cited_texts: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str  # with [n] markers after cited passages
    sources: list[Source]  # cited chunks, one number per URL, in order of first use
    stop_reason: str
    found: bool  # False when the model reports the docs don't cover the question
    usd: float
    usage: dict


def document(hit: Hit) -> dict:
    states = "; ".join(
        f"{s['feature_gate'] or 'feature'}: {s['state']} since v{s['since']}"
        for s in hit.feature_states
    )
    context = f"URL: {hit.url}\nKubernetes version: {hit.version}"
    if states:
        context += f"\nFeature state: {states}"
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": hit.text},
        "title": " > ".join(hit.heading_path),
        "context": context,
        "citations": {"enabled": True},
    }


def request(question: str, version: str, hits: list[Hit], note: str = "") -> dict:
    prompt = f"Kubernetes version: {version}\n"
    if note:
        prompt += f"Note: {note}\n"
    prompt += f"\nQuestion: {question}"
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT},
        "messages": [
            {
                "role": "user",
                "content": [*(document(h) for h in hits), {"type": "text", "text": prompt}],
            }
        ],
    }


def parse(response, hits: list[Hit], usd: float) -> Answer:
    if response.stop_reason == "refusal":
        return Answer(
            "The model declined to answer this question.",
            [],
            "refusal",
            False,
            usd,
            usage_dict(response.usage),
        )
    sources: dict[str, Source] = {}  # URL -> source: chunks of one section share a number
    parts = []
    for block in response.content:
        if block.type != "text":
            continue  # thinking blocks
        parts.append(block.text)
        markers = []
        for citation in block.citations or []:
            hit = hits[citation.document_index]
            if hit.url not in sources:
                sources[hit.url] = Source(len(sources) + 1, hit)
            sources[hit.url].cited_texts.append(citation.cited_text)
            marker = f"[{sources[hit.url].number}]"
            if marker not in markers:
                markers.append(marker)
        if markers:
            parts.append("".join(markers))
    text = "".join(parts).strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n(Answer truncated: output limit reached.)"
    return Answer(
        text=text,
        sources=sorted(sources.values(), key=lambda s: s.number),
        stop_reason=response.stop_reason,
        found=not text.startswith(NOT_FOUND),
        usd=usd,
        usage=usage_dict(response.usage),
    )


def usage_dict(usage) -> dict:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
    }


def generate(claude, question: str, version: str, hits: list[Hit], note: str = "") -> Answer:
    response, usd = claude.create(purpose="answer", **request(question, version, hits, note))
    return parse(response, hits, usd)
