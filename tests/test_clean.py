import json
import textwrap

import pytest

from ingest.clean import clean_version, main

HUGO = """
[params]
latest = "v1.37"
version = "v1.36"

[[params.versions]]
version = "v1.36"
githubbranch = "v1.36.4"
"""

I18N = """
[note]
other = "Note:"
[caution]
other = "Caution:"
[whatsnext_heading]
other = "What's next"
[prerequisites_heading]
other = "Before you begin"
[feature_state]
other = "Feature state:"
[feature_gate_enabled]
other = "enabled by default"
[feature_gate_disabled]
other = "disabled by default"
[version_check_mustbeorlater]
other = "Your Kubernetes server must be at or later than version "
[version_check_tocheck]
other = "To check the version, enter "
"""

PODS = """\
---
title: Pods
content_type: concept
min-kubernetes-server-version: v1.30
---
<!-- overview -->
A {{< glossary_tooltip text="Pod" term_id="pod" >}} runs on a
{{< glossary_tooltip term_id="Node" >}}.

{{< feature-state feature_gate_name="MyGate" >}}

{{< feature-state for_k8s_version="v1.24" state="stable" >}}

## {{% heading "prerequisites" %}}

{{< include "task-tutorial-prereqs.md" >}} {{< version-check >}}

Client v{{< skew currentVersion >}}, patch {{< skew currentPatchVersion >}},
previous {{< skew prevMinorVersion >}}, minus one {{< skew currentVersionAddMinor -1 >}},
docs {{< param "version" >}}.

{{< note >}}
Pods are *ephemeral*.
{{< /note >}}

{{% code_sample file="pods/simple-pod.yaml" %}}

{{< tabs name="os" >}}
{{% tab name="Linux" %}}
Use bash.
{{% /tab %}}
{{< tab name="Zsh" include="included/zsh.md" />}}
{{< /tabs >}}

See [the API]({{< ref "/docs/reference/kubernetes-api/pod-v1" >}}) and
{{< api-reference page="pod-v1" anchor="PodSpec" >}}.

{{< mermaid >}}
graph LR; A-->B
{{< /mermaid >}}

```yaml
# <!-- comment kept inside code -->
kind: Pod
```

## {{% heading "whatsnext" %}}
"""


@pytest.fixture
def site(tmp_path):
    raw = tmp_path / "v1.36"
    files = {
        "hugo.toml": HUGO,
        "i18n/en/en.toml": I18N,
        "content/en/docs/concepts/pods.md": PODS,
        "content/en/docs/concepts/_index.md": "---\ntitle: Concepts\n---\n",
        "content/en/docs/reference/kubernetes-api/pod-v1.md": "---\ntitle: Pod\n---\nPod API.\n",
        "content/en/docs/reference/glossary/index.md": "---\ntitle: Glossary\n---\n",
        "content/en/docs/reference/glossary/pod.md": (
            "---\ntitle: Pod\nid: pod\nfull_link: /docs/concepts/pods/\naka:\n- po\n---\n"
            'The smallest {{< glossary_tooltip text="object" term_id="node" >}}.\n\n'
            "<!--more-->\n\nMore detail.\n"
        ),
        "content/en/docs/reference/glossary/node.md": (
            "---\ntitle: Node\nid: node\n---\nA machine.\n"
        ),
        "content/en/docs/reference/command-line-tools-reference/feature-gates/index.md": (
            '---\ntitle: Feature Gates\n---\n{{< feature-gate-table include="alpha" >}}\n'
            "Gates are flags.\n"
        ),
        "content/en/docs/reference/command-line-tools-reference/feature-gates/MyGate.md": (
            "---\ntitle: MyGate\ncontent_type: feature_gate\n_build:\n  render: false\n"
            'stages:\n  - stage: alpha\n    defaultValue: false\n    fromVersion: "1.30"\n'
            '    toVersion: "1.31"\n  - stage: beta\n    defaultValue: true\n'
            '    fromVersion: "1.32"\n---\nEnables the thing.\n'
        ),
        "content/en/docs/concepts/included/zsh.md": "---\nheadless: true\n---\nUse zsh.\n",
        "content/en/docs/contribute/style.md": "---\ntitle: Style\n---\nHow to write docs.\n",
        "content/en/includes/task-tutorial-prereqs.md": "You need a cluster.\n",
        "content/en/examples/pods/simple-pod.yaml": "apiVersion: v1\nkind: Pod\n",
    }
    for rel, text in files.items():
        path = raw / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text) if rel.endswith(".toml") else text)
    return raw


@pytest.fixture
def cleaned(site):
    records, gates, report = clean_version(site, "1.36")
    return {r["id"]: r for r in records}, gates, report


def pods(cleaned):
    return cleaned[0]["v1.36:/docs/concepts/pods/"]


def test_page_metadata(cleaned):
    r = pods(cleaned)
    assert r["kind"] == "page"
    assert r["title"] == "Pods"
    assert r["url"] == "https://v1-36.docs.kubernetes.io/docs/concepts/pods/"
    assert r["source_path"] == "content/en/docs/concepts/pods.md"
    assert r["content_type"] == "concept"
    assert r["min_server_version"] == "v1.30"


def test_glossary_tooltip_uses_text_or_term_title_case_insensitively(cleaned):
    assert "A Pod runs on a\nNode." in pods(cleaned)["text"]


def test_feature_state_from_gate_and_explicit(cleaned):
    r = pods(cleaned)
    assert "Feature state: `Kubernetes v1.32 [beta]` enabled by default" in r["text"]
    assert "Feature state: `Kubernetes v1.24 [stable]`" in r["text"]
    assert r["feature_states"] == [
        {"feature_gate": "MyGate", "state": "beta", "since": "1.32", "default_enabled": True},
        {"feature_gate": None, "state": "stable", "since": "1.24"},
    ]


def test_headings_include_and_version_check(cleaned):
    text = pods(cleaned)["text"]
    assert "## Before you begin" in text
    assert "## What's next" in text
    assert "You need a cluster." in text
    assert (
        "must be at or later than version v1.30. To check the version, enter `kubectl version`."
        in text
    )


def test_version_substitutions_follow_hugo_templates(cleaned):
    text = pods(cleaned)["text"]
    # prevMinorVersion counts from site `latest` (1.37), not from this version.
    assert "Client v1.36, patch 1.36.4,\nprevious 1.36, minus one 1.35,\ndocs v1.36." in text


def test_callout_code_sample_and_tabs(cleaned):
    text = pods(cleaned)["text"]
    assert "**Note:** Pods are *ephemeral*." in text
    assert "Example file `pods/simple-pod.yaml`:\n\n```yaml\napiVersion: v1\nkind: Pod\n```" in text
    assert "**Linux**\n\nUse bash." in text
    assert "**Zsh**\n\nUse zsh." in text


def test_refs_become_site_urls(cleaned):
    text = pods(cleaned)["text"]
    assert "[the API](/docs/reference/kubernetes-api/pod-v1/)" in text
    assert "[PodSpec](/docs/reference/kubernetes-api/pod-v1/#PodSpec)" in text


def test_dropped_shortcodes_and_comments(cleaned):
    text = pods(cleaned)["text"]
    assert "graph LR" not in text
    assert "overview" not in text
    assert "# <!-- comment kept inside code -->" in text  # comments inside code survive
    assert "{{" not in text


def test_skipped_pages(cleaned):
    records, _, report = cleaned
    urls = {r["url"].removeprefix("https://v1-36.docs.kubernetes.io") for r in records.values()}
    assert "/docs/contribute/style/" not in urls  # excluded section
    assert "/docs/concepts/included/zsh/" not in urls  # headless, only via include
    assert "/docs/concepts/" not in urls  # empty section index
    assert report["skipped"] == {"excluded section": 1, "not rendered": 1, "empty": 1}


def test_glossary_records(cleaned):
    records = cleaned[0]
    r = records["v1.36:/docs/reference/glossary/?all=true#term-pod"]
    assert r["kind"] == "glossary"
    assert r["text"] == "# Pod\n\nThe smallest object.\n\nMore detail.\n\nAlso known as: po.\n"
    assert r["full_link"] == "/docs/concepts/pods/"


def test_feature_gate_records(cleaned):
    records, gates, _ = cleaned
    [gate] = gates
    assert gate.name == "MyGate" and not gate.removed
    assert [s["stage"] for s in gate.stages] == ["alpha", "beta"]

    url = "/docs/reference/command-line-tools-reference/feature-gates/"
    r = records[f"v1.36:{url}#MyGate"]
    assert r["kind"] == "feature_gate"
    assert "is at the beta stage and is enabled by default" in r["text"]
    assert "- alpha (disabled by default): 1.30–1.31" in r["text"]
    assert "- beta (enabled by default): 1.32+" in r["text"]
    assert "Enables the thing." in r["text"]
    # The gates index page is a normal page; its generated table is dropped.
    assert records[f"v1.36:{url}"]["text"] == "Gates are flags.\n"


def test_report_is_clean(cleaned):
    _, _, report = cleaned
    assert report["warnings"] == {}
    assert report["leftover_shortcode_tags"] == 0
    assert report["leftover_html_tags"] == 0
    assert report["records"] == {"page": 3, "glossary": 2, "feature_gate": 1}


def test_unknown_shortcode_is_reported(site):
    (site / "content/en/docs/concepts/new.md").write_text(
        "---\ntitle: New\n---\n{{< shiny >}}kept{{< /shiny >}}\n"
    )
    records, _, report = clean_version(site, "1.36")
    assert report["warnings"] == {"unknown shortcode: shiny": 1}
    new = next(r for r in records if r["title"] == "New")
    assert new["text"] == "kept\n"


def test_latest_version_uses_kubernetes_io(site):
    hugo = site / "hugo.toml"
    hugo.write_text(hugo.read_text().replace('version = "v1.36"', 'version = "v1.37"', 1))
    records, _, _ = clean_version(site, "1.37")
    assert records[0]["url"].startswith("https://kubernetes.io/docs/")


def test_main_writes_outputs(site, tmp_path):
    out = tmp_path / "clean"
    main(["--versions", "1.36", "--raw-dir", str(site.parent), "--out-dir", str(out)])
    pages = (out / "v1.36/pages.jsonl").read_text().splitlines()
    assert len(pages) == 6
    gates = [
        json.loads(line) for line in (out / "v1.36/feature_gates.jsonl").read_text().splitlines()
    ]
    assert gates[0]["name"] == "MyGate"
    assert json.loads((out / "v1.36/report.json").read_text())["version"] == "1.36"
