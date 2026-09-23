-- Applied by scripts/init_db.py; every statement is safe to re-run.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per unique text to embed, per index. An index is one chunking + embed-text
-- combination (e.g. "heading-plain"), so experiment variants live side by side.
CREATE TABLE IF NOT EXISTS contents (
    id          bigserial PRIMARY KEY,
    index_name  text NOT NULL,
    key         text NOT NULL,  -- sha256 of embed_text
    kind        text NOT NULL,  -- page | glossary | feature_gate
    text        text NOT NULL,
    embed_text  text NOT NULL,
    n_tokens    integer NOT NULL,
    versions    text[] NOT NULL,  -- Kubernetes versions this text appears in
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (index_name, key)
);
CREATE INDEX IF NOT EXISTS contents_versions_idx ON contents USING gin (versions);
CREATE INDEX IF NOT EXISTS contents_tsv_idx ON contents USING gin (tsv);

-- Where each text appears. Metadata is per occurrence because the same text can carry a
-- different URL, heading path or inherited feature state in different versions.
CREATE TABLE IF NOT EXISTS occurrences (
    content_id      bigint NOT NULL REFERENCES contents (id) ON DELETE CASCADE,
    version         text NOT NULL,
    chunk_id        text NOT NULL,
    record_id       text NOT NULL,
    url             text NOT NULL,
    title           text NOT NULL,
    heading_path    text[] NOT NULL,
    anchor          text NOT NULL,
    feature_states  jsonb NOT NULL,
    content_type    text NOT NULL,
    source_path     text NOT NULL,
    commit          text NOT NULL,
    PRIMARY KEY (content_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS occurrences_version_idx ON occurrences (content_id, version);

-- One row per text per embedding model, so models can be compared on the same contents.
-- Dimensions vary by model; each model gets a partial HNSW index on a typed cast
-- (created by ingest.embed), e.g.
--   CREATE INDEX ... ON embeddings USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
--   WHERE model = 'voyage-4';
CREATE TABLE IF NOT EXISTS embeddings (
    content_id  bigint NOT NULL REFERENCES contents (id) ON DELETE CASCADE,
    model       text NOT NULL,
    embedding   vector NOT NULL,
    PRIMARY KEY (content_id, model)
);
