"""Apply db/schema.sql to DATABASE_URL. Safe to run repeatedly.

Usage: uv run python -m scripts.init_db
"""

from pathlib import Path

import psycopg

from rag.settings import get_settings

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def main() -> None:
    url = get_settings().database_url
    with psycopg.connect(url) as conn:
        conn.execute(SCHEMA.read_text())
    print(f"Applied {SCHEMA.name} to {url}")


if __name__ == "__main__":
    main()
