import psycopg
import pytest

from rag.settings import Settings, get_settings


def test_settings_default_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert Settings(_env_file=None).database_url == "postgresql://localhost/k8s_docs_rag"


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    assert Settings(_env_file=None).database_url == "postgresql://example/db"


@pytest.fixture
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"database unavailable: {e}")
    with c:
        yield c


def test_pgvector_extension_installed(conn):
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None, "run scripts/init_db.py first"
