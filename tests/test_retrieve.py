from ingest.embed import store
from ingest.load import load
from rag.retrieve import vector_search
from tests.test_load import conn, content, occurrence, test_db_url  # noqa: F401  (fixtures)


def test_vector_search_filters_by_version_and_returns_that_versions_url(conn):  # noqa: F811
    load(
        conn,
        "heading-plain",
        [
            content("a", "alpha", ["1.36", "1.37"]),
            content("b", "beta", ["1.37"]),
            content("c", "gamma", ["1.36"]),
        ],
    )
    ids = dict(conn.execute("SELECT key, id FROM contents").fetchall())
    store(conn, "fake-2d", [(ids["a"], [1.0, 0.0]), (ids["b"], [0.9, 0.1]), (ids["c"], [0.0, 1.0])])

    hits = vector_search(conn, [1.0, 0.0], "1.36", k=5, model="fake-2d", dim=2)
    assert [h.text for h in hits] == ["alpha", "gamma"]  # "beta" is 1.37 only
    assert hits[0].url == "https://kubernetes.io/docs/a/" and hits[0].version == "1.36"
    assert hits[0].score > hits[1].score
