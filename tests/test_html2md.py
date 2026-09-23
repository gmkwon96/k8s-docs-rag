from ingest.html2md import convert, count_tags


def test_text_without_html_is_unchanged():
    assert convert("Run `kubectl get pods`.\n") == "Run `kubectl get pods`.\n"


def test_placeholders_that_are_not_html_tags_survive():
    text = "Use `<key>=<value>` or kubectl describe pod <pod-name> at <path>."
    assert convert(text) == text


def test_inline_tags():
    html = 'Set <code>replicas</code> to <em>3</em>, see <a href="/docs/x/">docs</a>.<br/>Next'
    assert convert(html) == "Set `replicas` to *3*, see [docs](/docs/x/).\nNext"


def test_entities_are_decoded():
    assert convert("<code>&lt;quantity&gt;</code>&nbsp;&amp; more") == "`<quantity>` & more"


def test_api_field_table_becomes_key_value_list():
    html = """<table>
  <thead><tr><th>Field</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>containers</code>&nbsp;<strong>*</strong><br/><em>Container array</em></td>
        <td>List of containers.</td></tr>
    <tr><td><code>hostname</code><br/><em>string</em></td><td>Pod <p>hostname</p>.</td></tr>
  </tbody>
</table>"""
    assert convert(html).strip() == (
        "- `containers` (required) *Container array*: List of containers.\n"
        "- `hostname` *string*: Pod hostname ."
    )


def test_flag_table_joins_description_rows():
    html = """<table><tbody>
<tr><td colspan="2">--address string&nbsp;&nbsp;Default: 0.0.0.0</td></tr>
<tr><td></td><td><p>The IP address to serve on.</p></td></tr>
</tbody></table>"""
    assert (
        convert(html).strip()
        == "- --address string Default: 0.0.0.0\n  The IP address to serve on."
    )


def test_multi_column_table_uses_header_names():
    html = (
        "<table><caption>Ports</caption><tr><th>Protocol</th><th>Port</th><th>Used by</th></tr>"
        "<tr><td>TCP</td><td>6443</td><td>All</td></tr></table>"
    )
    assert convert(html).strip() == "Table: Ports\n- Protocol: TCP; Port: 6443; Used by: All"


def test_headings_lists_and_labels():
    html = '<h2>Overview</h2><ul><li><label class="x">Type:</label><span>Counter</span></li></ul>'
    out = convert(html)
    assert "## Overview" in out
    assert "- Type: Counter" in out
    assert count_tags(out) == 0
