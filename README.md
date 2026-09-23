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
uv run pytest

uv run python -m ingest.fetch        # docs at the commits pinned in ingest/sources.lock.json
```

`ingest.fetch --update --versions 1.35 1.36 1.37` re-pins each version to its branch tip: `release-1.xx` if that branch exists, else `main` (which documents the current release).

## License

Documentation content comes from [kubernetes/website](https://github.com/kubernetes/website) (CC BY 4.0).
