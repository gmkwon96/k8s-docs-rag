from ingest.shortcodes import Shortcode, parse, parse_args


def only_shortcodes(nodes):
    return [n for n in nodes if isinstance(n, Shortcode)]


def test_plain_text_passes_through():
    assert parse("no shortcodes here") == ["no shortcodes here"]


def test_standalone_with_named_args():
    [sc] = only_shortcodes(parse('a {{< glossary_tooltip text="Pods" term_id="pod" >}} b'))
    assert sc.name == "glossary_tooltip"
    assert sc.kwargs == {"text": "Pods", "term_id": "pod"}
    assert sc.inner is None


def test_positional_args_and_markdown_delimiter():
    [sc] = only_shortcodes(parse('{{% heading "whatsnext" %}}'))
    assert (sc.name, sc.args) == ("heading", ["whatsnext"])
    [sc] = only_shortcodes(parse('{{< skew currentVersionAddMinor -1 "." >}}'))
    assert sc.args == ["currentVersionAddMinor", "-1", "."]


def test_paired_shortcode_collects_inner():
    nodes = parse("x {{< note >}}Be careful.{{< /note >}} y")
    assert nodes[0] == "x "
    assert nodes[1].name == "note"
    assert nodes[1].inner == ["Be careful."]
    assert nodes[2] == " y"


def test_nested_pairs_with_standalone_inside():
    text = (
        '{{< tabs name="t" >}}{{% tab name="Linux" %}}'
        "run {{< skew currentVersion >}}{{% /tab %}}{{< /tabs >}}"
    )
    [tabs] = parse(text)
    [tab] = tabs.inner
    assert tab.kwargs["name"] == "Linux"
    assert tab.inner[0] == "run "
    assert tab.inner[1].name == "skew" and tab.inner[1].inner is None


def test_unclosed_shortcode_is_standalone_and_keeps_following_text():
    nodes = parse('{{< feature-state state="beta" >}} Text after.')
    assert nodes[0].name == "feature-state" and nodes[0].inner is None
    assert nodes[1] == " Text after."


def test_self_closing_tag():
    [sc] = only_shortcodes(parse('{{< tab name="Bash" include="included/bash.md" />}}'))
    assert sc.kwargs["include"] == "included/bash.md"
    assert sc.inner is None


def test_multiline_args():
    text = '{{< tutorials/module\n    path="/docs/x/"\n    title="1. Create" >}}'
    [sc] = parse(text)
    assert sc.kwargs == {"path": "/docs/x/", "title": "1. Create"}


def test_escaped_shortcode_becomes_literal_text():
    assert "".join(parse("use {{</* note */>}} like this")) == "use {{< note >}} like this"


def test_stray_closing_tag_is_dropped():
    assert parse("a {{< /note >}} b") == ["a ", " b"]


def test_parse_args_unquotes():
    assert parse_args(r'"a \"b\"" key=`raw` bare') == (['a "b"', "bare"], {"key": "raw"})
