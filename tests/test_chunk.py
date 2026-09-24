import json
import re

from ingest.chunk import (
    anchorize,
    chunk_record,
    chunk_record_fixed,
    count_tokens,
    main,
    parse_sections,
)


def record(text, title="Page", url="https://kubernetes.io/docs/page/", **extra):
    return {
        "id": "v1.37:/docs/page/",
        "version": "1.37",
        "kind": "page",
        "url": url,
        "title": title,
        "text": text,
        "source_path": "content/en/docs/page.md",
        "feature_states": [],
        **extra,
    }


def paragraph(words, word="pods"):
    return " ".join([word] * words)


def fences(text):
    return len(re.findall(r"^\s*```", text, re.M))


def test_anchorize_matches_hugo_blackfriday():
    assert anchorize("What is a Pod?") == "what-is-a-pod"
    assert anchorize("Use case: dotfiles in a secret volume") == (
        "use-case-dotfiles-in-a-secret-volume"
    )
    assert anchorize("`KubeletConfiguration` fields") == "kubeletconfiguration-fields"
    assert anchorize("[Link text](/docs/x/) and more") == "link-text-and-more"


def test_sections_heading_paths_and_anchors():
    text = (
        "Intro.\n\n## Setup\n\nA.\n\n### Linux {#on-linux}\n\nB.\n\n## Setup\n\nC.\n\n## Use\n\nD."
    )
    sections = parse_sections("Guide", text)
    assert [(s.path, s.anchor) for s in sections] == [
        (["Guide"], ""),
        (["Guide", "Setup"], "setup"),
        (["Guide", "Setup", "Linux"], "on-linux"),
        (["Guide", "Setup"], "setup-1"),  # duplicate headings get a numeric suffix
        (["Guide", "Use"], "use"),
    ]
    # the explicit {#id} is dropped from the text
    assert sections[2].blocks[0].text == "### Linux"


def test_h1_repeating_the_title_is_not_a_section():
    sections = parse_sections("Pod", "# Pod\n\nThe smallest object.")
    assert len(sections) == 1 and sections[0].path == ["Pod"]
    assert [b.text for b in sections[0].blocks] == ["# Pod", "The smallest object."]


def test_headings_inside_code_are_ignored_and_unclosed_fences_are_text():
    text = "```bash\n# not a heading\n```\n\n## Real\n\n```\nnever closed\n\n## Also real"
    sections = parse_sections("T", text)
    assert [s.path[-1] for s in sections] == ["T", "Real", "Also real"]
    assert sections[0].blocks[0].is_code


def test_small_sections_merge_forward():
    text = "## A\n\nShort.\n\n## B\n\n" + paragraph(100)
    [chunk] = chunk_record(record(text))[0]
    assert chunk["text"].startswith("## A\n\nShort.\n\n## B")
    assert chunk["anchor"] == "a"
    assert chunk["url"] == "https://kubernetes.io/docs/page/#a"
    assert chunk["heading_path"] == ["Page", "A"]


def test_large_section_splits_at_block_boundaries():
    text = "## Big\n\n" + "\n\n".join(
        paragraph(150, w) for w in ("alpha", "beta", "gamma", "delta")
    )
    chunks, _ = chunk_record(record(text))
    assert len(chunks) == 2
    assert all(c["n_tokens"] <= 512 for c in chunks)
    assert all(c["anchor"] == "big" for c in chunks)
    # paragraphs stay whole: each block is a single repeated word
    for c in chunks:
        for block in c["text"].split("\n\n"):
            if not block.startswith("##"):
                assert len(set(block.split())) == 1


def test_code_block_is_kept_whole_when_it_fits():
    code = "```yaml\n" + "\n".join(f"key{i}: value" for i in range(60)) + "\n```"
    text = "## Example\n\n" + paragraph(300) + "\n\n" + code
    chunks, _ = chunk_record(record(text))
    assert any(code in c["text"] for c in chunks)
    assert all(fences(c["text"]) % 2 == 0 for c in chunks)


def test_oversized_code_block_is_split_and_refenced():
    code = "```yaml\n" + "\n".join(f"key{i}: some longer value here" for i in range(400)) + "\n```"
    chunks, _ = chunk_record(record("## Huge\n\n" + code))
    assert len(chunks) > 1
    for c in chunks:
        assert c["n_tokens"] <= 512
        assert fences(c["text"]) == 2
        assert "```yaml" in c["text"]


def test_split_keeps_continuation_lines_with_their_item():
    rows = "\n".join(f"- --flag-{i} string\n  {paragraph(12, 'described')}" for i in range(80))
    chunks, _ = chunk_record(record("## Options\n\n" + rows))
    assert len(chunks) > 1
    for c in chunks:
        body = c["text"].removeprefix("## Options\n\n")
        assert body.startswith("- --flag-")
        assert not body.rstrip().splitlines()[-1].startswith("- ")  # ends with a description


def test_feature_states_apply_to_section_and_descendants_only():
    text = (
        "## A\n\nFeature state: `Kubernetes v1.30 [beta]` enabled by default\n\n"
        + paragraph(120)
        + "\n\n### A.1\n\n"
        + paragraph(120)
        + "\n\n## B\n\n"
        + paragraph(120)
    )
    state = {"feature_gate": "G", "state": "beta", "since": "1.30", "default_enabled": True}
    chunks, check = chunk_record(record(text, feature_states=[state]))
    assert check == {"feature_state_markers": 1, "feature_states": 1}
    by_anchor = {c["anchor"]: c["feature_states"] for c in chunks}
    assert by_anchor == {"a": [state], "a-1": [state], "b": []}


def test_page_level_feature_state_applies_to_every_chunk():
    text = "Feature state: `Kubernetes v1.24 [stable]`\n\n" + paragraph(120) + "\n\n## X\n\n"
    text += paragraph(120)
    state = {"feature_gate": None, "state": "stable", "since": "1.24"}
    chunks, _ = chunk_record(record(text, feature_states=[state]))
    assert len(chunks) == 2 and all(c["feature_states"] == [state] for c in chunks)


def test_record_anchor_is_kept_for_single_section_records():
    url = "https://kubernetes.io/docs/reference/glossary/?all=true#term-pod"
    [chunk] = chunk_record(record("# Pod\n\nThe smallest object.", title="Pod", url=url))[0]
    assert chunk["url"] == url
    assert chunk["heading_path"] == ["Pod"]


def test_leading_h1_that_differs_from_title_keeps_record_anchor():
    url = "https://kubernetes.io/docs/reference/command-line-tools-reference/feature-gates/#MyGate"
    text = "# Feature gate: MyGate\n\nStage history:\n- alpha: 1.30"
    [chunk] = chunk_record(record(text, title="MyGate", url=url))[0]
    assert chunk["url"] == url
    assert chunk["anchor"] == "MyGate"
    assert chunk["heading_path"] == ["MyGate"]
    assert chunk["text"].startswith("# Feature gate: MyGate\n\n")


def test_chunk_fields():
    [c] = chunk_record(record(paragraph(80)))[0]
    assert c["chunk_id"] == "v1.37:/docs/page/::0"
    assert c["n_tokens"] == count_tokens(c["text"])
    assert len(c["content_hash"]) == 64


def test_main_writes_chunks_and_report(tmp_path):
    clean = tmp_path / "clean" / "v1.37"
    clean.mkdir(parents=True)
    (clean / "pages.jsonl").write_text(json.dumps(record("## A\n\n" + paragraph(100))) + "\n")
    out = tmp_path / "chunks"
    main(["--versions", "1.37", "--clean-dir", str(tmp_path / "clean"), "--out-dir", str(out)])
    chunks = (out / "v1.37/chunks.jsonl").read_text().splitlines()
    report = json.loads((out / "v1.37/report.json").read_text())
    assert len(chunks) == report["chunks"] == 1
    assert report["over_max"] == 0 and report["feature_state_mismatches"] == []


def test_fixed_windows_overlap_and_link_to_their_starting_section():
    text = "\n\n".join(
        [paragraph(300, "intro"), "## Alpha", paragraph(300, "alpha"), "## Beta", paragraph(300)]
    )
    chunks, _ = chunk_record_fixed(record(text), max_tokens=256, overlap=32)
    assert all(c["n_tokens"] <= 258 for c in chunks)
    assert chunks[0]["url"] == "https://kubernetes.io/docs/page/"
    assert any(c["anchor"] == "alpha" for c in chunks)
    assert chunks[-1]["anchor"] == "beta"
    # consecutive windows share their boundary text
    tail = chunks[0]["text"].split()[-5:]
    assert " ".join(tail) in chunks[1]["text"]
