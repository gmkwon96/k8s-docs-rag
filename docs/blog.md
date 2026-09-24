# Better retrieval, same answers: what a $5 eval budget taught me about a docs RAG

*Draft.* I built a question-answering service over the official Kubernetes documentation (versions 1.35-1.37) and spent most of the effort on evaluation instead of features: a 200-question golden set, retrieval metrics, a Claude judge calibrated against human scores, and a bootstrap confidence interval on every number. The whole thing, experiments included, ran on under $5 of Claude API usage. Here is what the numbers said, including where they contradicted what I expected.

## The setup

- **Corpus.** kubernetes/website at pinned commits for three versions. Hugo shortcodes are rendered to markdown, feature states (alpha/beta/stable per version) are kept as metadata, and pages are split at headings into chunks of at most 512 tokens. 33,147 chunks collapse to 14,025 unique texts, because most of the docs don't change between minor versions.
- **Pipeline.** Pick the Kubernetes version (explicit, mentioned in the question, or latest), retrieve with voyage-4 embeddings filtered to that version, answer with claude-sonnet-5 using the Citations API.
- **Golden set.** 120 dev and 80 test questions across six categories: facts, how-tos, version-dependent questions, multi-hop questions, questions the docs don't answer, and false premises ("How do I set `replicas` on a DaemonSet?"). Evidence is a verbatim quote plus the section URL, not a chunk id. A retrieved chunk counts if it comes from that page and contains the quote, so the labels survive any re-chunking.
- **Statistics.** About 100 items per split means a two-point difference is often noise. Every comparison is a paired bootstrap over the same items (10,000 resamples), reported as a difference with a 95% interval.

## Finding 1: reranking was the one big retrieval win

Retrieving 50 candidates by vector search and reranking them with Voyage's rerank-3 down to 8 raised dev MRR by +0.131 [+0.070, +0.194] and nDCG@10 by +0.112. On the held-out test split, scored once with the configuration chosen on dev, the gain reproduced: MRR +0.130 [+0.068, +0.194].

Everything else I tried on the retrieval side was flat or worse:

| change | result |
|---|---|
| keyword search (Postgres full text) | Recall@10 −0.126, clearly worse |
| hybrid (RRF of vector + keyword) | Recall@5 +0.034, not significant; nothing on top of rerank |
| heading path prefixed to each chunk | +0.026 vector-only (P≈0.06), ±0 after rerank |
| open-weights Qwen3-Embedding-0.6B instead of voyage-4 | −0.025 Recall@10 with rerank, not significant |
| dropping the version filter | MRR −0.024 [−0.050, −0.004], worse on version questions |

The reranker also closed most of the gap between embedding models. If you can't call an embedding API, a small local model plus a reranker gets you close.

## Finding 2: the better retrieval barely changed the answers

I expected answer correctness to follow retrieval. It didn't:

- correctness: 0.938 → 0.946, difference +0.008 [−0.033, +0.054]
- faithfulness (claims supported by the cited sources): 0.954 → 0.975, difference +0.021 [−0.004, +0.050]

Per item, 7 answers got better and 10 got slightly worse. The explanation is a ceiling: with the baseline's top 8 chunks the model already answered 94% of questions correctly, because the fact it needs is usually somewhere in those 8 chunks even when it isn't ranked first. Better ranking mostly changed how cleanly the answer was supported, which is where the faithfulness gain shows up.

The practical lesson: measure retrieval and generation separately, and don't assume a retrieval gain will show up in end-to-end quality. What the generator sees is the whole top k, so a ranking metric like MRR can move a lot while the answer barely changes.

## Finding 3: rewriting the query with an LLM made things worse

Query rewriting is a standard trick, so I tried three variants with claude-haiku-4-5, always reranking against the original question:

- **Rewrite** into documentation terms: nDCG@10 −0.024 [−0.053, −0.001]. Rewrites turned questions into keyword lists and lost the intent ("maximum pods per node" became "kubelet configuration"). The prompt said not to trust the question's premise, but they carried false premises along anyway.
- **HyDE** (embed a hypothetical answer passage): Recall@10 +0.020, not significant.
- **Union** of both candidate sets: exactly zero change. The extra candidates changed the top 8 for 48 questions but never displaced an evidence chunk.

None of these justified an extra LLM call on every query.

## Finding 4: the cheap model was statistically indistinguishable

On a 50-question stratified sample, claude-haiku-4-5 answers scored +0.010 [−0.040, +0.060] correctness against claude-sonnet-5, at about a third of the cost per answer ($0.0031 vs $0.0100) and much shorter output. Two caveats keep this from being a clean win. At n=50 the interval can't rule out a five-point gap. And the judge is also Haiku, so self-preference can't be excluded.

## The judge needed calibrating, and I had to be honest about it

I scored 50 answers by hand, blind to the judge. Raw Cohen's kappa was 0.31 at 92% agreement. That kappa is low because I had scored 49 of 50 answers as fully correct, and with scores that skewed, one disagreement swings kappa a lot. I adjudicated the five disagreements against the sources (the judge was right on three, I was on two) and revised the rubric; kappa against the adjudicated scores is 0.88. That second number is optimistic: I revised the rubric on the same items and adjudicated with the judge's reasons in view. The README says so, and a fresh sample would give an honest figure.

## Two bugs the eval caught

- **Stale data.** The first heading-path experiment scored −0.16 across the board, far too large to be real. The heading-path index had been built from files that predated an anchor fix, so its citation URLs pointed at anchors that no longer exist. Regenerated, the effect was +0.026.
- **Shared ANN index.** Loading that second index under the same embedding model also quietly degraded the *baseline* (Recall@10 0.891 → 0.871). The HNSW index is partial per model, and the index-name filter after the approximate scan lost neighbours. Comparing both indexes with exact search, dropping the extra index and rebuilding HNSW brought the baseline back to its original numbers exactly.

Neither bug would have shown up in a handful of manual test questions.

## Staying under $5

Every Claude call is checked against a local dollar ledger before it is sent. The worst case is counted input tokens (from the free count_tokens endpoint) plus `max_tokens` of output. Generation and judging run through the Message Batches API at half price. Each batch reserves its worst case up front and releases the unused part when it settles. Embedding and reranking stayed inside Voyage's free tokens under a similar token ledger. The public demo serves answers generated by the eval run instead of calling Claude live, so running it costs the operator nothing.

Total: about $4.30 of Claude usage and roughly 26M free Voyage tokens, for the experiments, two full generation runs on dev, one on test, and all the judging.
