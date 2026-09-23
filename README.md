# k8s-docs-rag

Q&A over the official Kubernetes documentation, with sentence-level citations, Kubernetes-version awareness, and an evaluation pipeline that reports every change as a number with a confidence interval.

> **Status:** work in progress. See [`plan.md`](plan.md) for the design and milestones.

## Local setup (macOS)

```sh
brew install uv postgresql@18 pgvector
brew services start postgresql@18
/opt/homebrew/opt/postgresql@18/bin/createdb k8s_docs_rag

uv sync                              # installs Python 3.14 + dependencies
cp .env.example .env
uv run python -m scripts.init_db     # enables pgvector
uv run pytest                        # DB tests use a throwaway k8s_docs_rag_test database

uv run python -m ingest.fetch        # docs at the commits pinned in ingest/sources.lock.json
uv run python -m ingest.clean        # Hugo shortcodes -> markdown, data/clean/v<version>/
uv run python -m ingest.chunk        # heading-based chunks, data/chunks/v<version>/
uv run python -m ingest.dedupe       # one entry per unique text, data/index/plain/
uv run python -m ingest.load         # into Postgres as index "heading-plain"
```

`ingest.fetch --update --versions 1.35 1.36 1.37` re-pins each version to its branch tip: `release-1.xx` if that branch exists, else `main` (which documents the current release).

## License

Documentation content comes from [kubernetes/website](https://github.com/kubernetes/website) (CC BY 4.0).
