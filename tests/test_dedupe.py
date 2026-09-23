import hashlib
import json

from ingest.dedupe import build_report, dedupe, main


def chunk(version, text, page="pods", heading=("Pods",), feature_states=(), n=0):
    base = (
        "https://kubernetes.io"
        if version == "1.37"
        else f"https://v{version.replace('.', '-')}.docs.kubernetes.io"
    )
    return {
        "chunk_id": f"v{version}:/docs/{page}/::{n}",
        "record_id": f"v{version}:/docs/{page}/",
        "version": version,
        "kind": "page",
        "url": f"{base}/docs/{page}/",
        "title": heading[0],
        "heading_path": list(heading),
        "anchor": "",
        "text": text,
        "n_tokens": len(text.split()),
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "feature_states": list(feature_states),
        "content_type": "concept",
        "source_path": f"content/en/docs/{page}.md",
        "commit": f"commit-{version}",
    }


def test_identical_text_across_versions_is_stored_once():
    chunks = [chunk("1.35", "same text"), chunk("1.36", "same text"), chunk("1.37", "new")]
    contents = dedupe(chunks)
    assert [c["text"] for c in contents] == ["same text", "new"]
    assert contents[0]["versions"] == ["1.35", "1.36"]
    assert [o["url"] for o in contents[0]["occurrences"]] == [
        "https://v1-35.docs.kubernetes.io/docs/pods/",
        "https://v1-36.docs.kubernetes.io/docs/pods/",
    ]


def test_metadata_is_kept_per_occurrence():
    alpha = {"feature_gate": "G", "state": "alpha", "since": "1.28"}
    beta = {"feature_gate": "G", "state": "beta", "since": "1.36"}
    chunks = [chunk("1.35", "t", feature_states=[alpha]), chunk("1.36", "t", feature_states=[beta])]
    [content] = dedupe(chunks)
    assert [o["feature_states"] for o in content["occurrences"]] == [[alpha], [beta]]
    report = build_report(chunks, [content], "plain")
    assert report["metadata_differs_across_versions"]["feature_states"] == 1


def test_same_text_on_several_pages_keeps_every_occurrence():
    chunks = [
        chunk("1.37", "Container fields", page="pod-v1", heading=("Pod", "Container")),
        chunk(
            "1.37", "Container fields", page="deployment-v1", heading=("Deployment", "Container")
        ),
        chunk("1.37", "Params", page="pod-v1", n=1),
        chunk("1.37", "Params", page="pod-v1", n=2),
    ]
    contents = dedupe(chunks)
    assert len(contents) == 2
    assert [o["record_id"] for o in contents[0]["occurrences"]] == [
        "v1.37:/docs/pod-v1/",
        "v1.37:/docs/deployment-v1/",
    ]
    report = build_report(chunks, contents, "plain")
    assert report["on_several_pages_in_a_version"] == 1
    assert report["repeated_within_a_page"] == 1


def test_breadcrumb_mode_separates_different_heading_paths():
    chunks = [
        chunk("1.37", "Container fields", page="pod-v1", heading=("Pod", "Container")),
        chunk(
            "1.37", "Container fields", page="deployment-v1", heading=("Deployment", "Container")
        ),
        chunk("1.36", "Container fields", page="pod-v1", heading=("Pod", "Container")),
    ]
    contents = dedupe(chunks, "breadcrumb")
    assert [c["embed_text"] for c in contents] == [
        "Pod > Container\n\nContainer fields",
        "Deployment > Container\n\nContainer fields",
    ]
    assert contents[0]["versions"] == ["1.36", "1.37"]
    assert contents[0]["text"] == "Container fields"


def test_report_counts_and_ratios():
    chunks = [chunk(v, "a b c d") for v in ("1.35", "1.36", "1.37")] + [chunk("1.37", "e f")]
    report = build_report(chunks, dedupe(chunks), "plain")
    assert (report["chunks"], report["contents"]) == (4, 2)
    assert (report["tokens_before"], report["tokens_after"]) == (14, 6)
    assert report["version_sets"] == {"1.35,1.36,1.37": 1, "1.37": 1}


def test_main_is_deterministic(tmp_path):
    chunks_dir = tmp_path / "chunks"
    for v, texts in {"1.36": ["x", "y"], "1.37": ["y", "z"], "1.35": ["x"]}.items():
        (chunks_dir / f"v{v}").mkdir(parents=True)
        (chunks_dir / f"v{v}" / "chunks.jsonl").write_text(
            "".join(json.dumps(chunk(v, t, n=i)) + "\n" for i, t in enumerate(texts))
        )
    args = ["--versions", "1.37", "1.35", "1.36", "--chunks-dir", str(chunks_dir)]
    main([*args, "--out-dir", str(tmp_path / "a")])
    main([*args, "--out-dir", str(tmp_path / "b")])
    a = (tmp_path / "a/plain/contents.jsonl").read_text()
    assert a == (tmp_path / "b/plain/contents.jsonl").read_text()
    contents = [json.loads(line) for line in a.splitlines()]
    # versions are read oldest first regardless of argument order
    assert [(c["text"], c["versions"]) for c in contents] == [
        ("x", ["1.35", "1.36"]),
        ("y", ["1.36", "1.37"]),
        ("z", ["1.37"]),
    ]
    report = json.loads((tmp_path / "a/plain/report.json").read_text())
    assert report["versions"] == ["1.35", "1.36", "1.37"]
