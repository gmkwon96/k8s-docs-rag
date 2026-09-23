import json
import os
from pathlib import Path

import psycopg
import pytest

from ingest.load import load, main

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost/k8s_docs_rag_test")


@pytest.fixture(scope="session")
def test_db_url():
    """A throwaway database with the schema applied; skipped when Postgres is unavailable."""
    admin_url = TEST_DB.rsplit("/", 1)[0] + "/postgres"
    name = TEST_DB.rsplit("/", 1)[1]
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as admin:
            exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", [name])
            if exists.fetchone() is None:
                admin.execute(f'CREATE DATABASE "{name}"')
    except psycopg.OperationalError as e:
        pytest.skip(f"database unavailable: {e}")
    with psycopg.connect(TEST_DB) as conn:
        conn.execute(SCHEMA.read_text())
    return TEST_DB


@pytest.fixture
def conn(test_db_url):
    with psycopg.connect(test_db_url) as c:
        c.execute("TRUNCATE contents, occurrences, embeddings RESTART IDENTITY")
        c.commit()
        yield c


def content(key, text, versions, occurrences=None):
    return {
        "key": key,
        "content_hash": key,
        "kind": "page",
        "embed_text": text,
        "text": text,
        "n_tokens": len(text.split()),
        "versions": versions,
        "occurrences": occurrences or [occurrence(v, f"v{v}:/docs/{key}/::0") for v in versions],
    }


def occurrence(version, chunk_id, feature_states=()):
    return {
        "version": version,
        "chunk_id": chunk_id,
        "record_id": chunk_id.split("::")[0],
        "url": f"https://kubernetes.io{chunk_id.split(':', 1)[1].split('::')[0]}",
        "title": "Pods",
        "heading_path": ["Pods", "Lifecycle"],
        "anchor": "lifecycle",
        "feature_states": list(feature_states),
        "content_type": "concept",
        "source_path": "content/en/docs/pods.md",
        "commit": "abc",
    }


def rows(conn, sql, *params):
    return conn.execute(sql, params).fetchall()


def test_load_inserts_contents_and_occurrences(conn):
    state = {"feature_gate": "G", "state": "beta", "since": "1.36", "default_enabled": True}
    contents = [
        content("a", "pods restart when containers fail", ["1.36", "1.37"]),
        content("b", "only in 1.37", ["1.37"], [occurrence("1.37", "v1.37:/docs/b/::0", [state])]),
    ]
    stats = load(conn, "heading-plain", contents)
    assert stats == {
        "index_name": "heading-plain",
        "contents": 2,
        "inserted": 2,
        "kept": 0,
        "deleted": 0,
        "occurrences": 3,
    }
    assert rows(conn, "SELECT key, versions FROM contents ORDER BY key") == [
        ("a", ["1.36", "1.37"]),
        ("b", ["1.37"]),
    ]
    [(fs, path)] = rows(
        conn,
        "SELECT feature_states, heading_path FROM occurrences WHERE chunk_id = %s",
        "v1.37:/docs/b/::0",
    )
    assert fs == [state] and path == ["Pods", "Lifecycle"]


def test_reload_keeps_ids_of_unchanged_texts_and_their_embeddings(conn):
    load(conn, "heading-plain", [content("a", "alpha", ["1.36"]), content("b", "beta", ["1.36"])])
    [(a_id,)] = rows(conn, "SELECT id FROM contents WHERE key = 'a'")
    conn.execute("INSERT INTO embeddings VALUES (%s, 'm', '[1,2,3]')", [a_id])
    conn.commit()

    stats = load(
        conn,
        "heading-plain",
        [content("a", "alpha", ["1.36", "1.37"]), content("c", "gamma", ["1.37"])],
    )

    assert (stats["inserted"], stats["kept"], stats["deleted"]) == (1, 1, 1)
    assert rows(conn, "SELECT id, versions FROM contents WHERE key = 'a'") == [
        (a_id, ["1.36", "1.37"])
    ]
    assert rows(conn, "SELECT key FROM contents ORDER BY key") == [("a",), ("c",)]
    assert rows(conn, "SELECT content_id, model FROM embeddings") == [(a_id, "m")]
    assert rows(conn, "SELECT count(*) FROM occurrences") == [(3,)]


def test_indexes_are_isolated(conn):
    load(conn, "heading-plain", [content("a", "alpha", ["1.36"])])
    load(conn, "heading-breadcrumb", [content("a", "Pods > alpha", ["1.36"])])
    load(conn, "heading-plain", [])
    assert rows(conn, "SELECT index_name FROM contents") == [("heading-breadcrumb",)]


def test_version_filter_and_keyword_search(conn):
    load(
        conn,
        "heading-plain",
        [
            content("a", "Set terminationGracePeriodSeconds on the Pod spec", ["1.35", "1.36"]),
            content("b", "Set terminationGracePeriodSeconds per probe", ["1.37"]),
        ],
    )
    hits = rows(
        conn,
        """
        SELECT key FROM contents, websearch_to_tsquery('english', %s) q
        WHERE versions @> ARRAY[%s] AND tsv @@ q
        """,
        "terminationGracePeriodSeconds",
        "1.36",
    )
    assert hits == [("a",)]


def test_main_reads_contents_jsonl(conn, test_db_url, tmp_path, capsys):
    index_dir = tmp_path / "plain"
    index_dir.mkdir()
    (index_dir / "contents.jsonl").write_text(json.dumps(content("a", "alpha", ["1.37"])) + "\n")
    main(["--index-dir", str(tmp_path), "--database-url", test_db_url])
    assert "heading-plain: 1 contents (1 new, 0 kept, 0 deleted), 1 occurrences" in (
        capsys.readouterr().out
    )
