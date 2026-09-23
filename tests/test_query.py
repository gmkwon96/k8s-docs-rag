from rag.query import choose_version, mentioned_versions

VERSIONS = ("1.35", "1.36", "1.37")


def test_defaults_to_latest():
    assert choose_version("How do I roll back?", versions=VERSIONS).version == "1.37"


def test_version_in_question():
    for q in ["In 1.35 is it beta?", "kubectl v1.36.2 flags", "Kubernetes 1.36 sidecars"]:
        assert choose_version(q, versions=VERSIONS).version in ("1.35", "1.36")


def test_explicit_version_wins():
    assert choose_version("In 1.35?", "1.36", versions=VERSIONS).version == "1.36"


def test_unindexed_version_falls_back_with_note():
    c = choose_version("Kubernetes 1.30 sidecars", versions=VERSIONS)
    assert (c.version, c.requested) == ("1.37", "1.30")
    assert "not indexed" in c.note


def test_other_decimals_are_not_versions():
    assert mentioned_versions("wait 1.5 seconds, then 10.25 more") == []
