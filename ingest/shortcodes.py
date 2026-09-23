"""Parse Hugo shortcodes into a tree of text and Shortcode nodes.

Handles `{{< name args >}}` and `{{% name args %}}`, closing tags `{{< /name >}}`,
self-closing `{{< name />}}`, and escaped `{{</* name */>}}` (emitted as literal text).
Whether a shortcode is paired is decided by whether a matching closing tag follows, so
no list of paired shortcode names is needed.
"""

import re
from dataclasses import dataclass, field

TAG_RE = re.compile(
    r"\{\{(?P<delim>[<%])(?P<esc>/\*)?\s*(?P<close>/)?\s*(?P<name>[\w./-]+)"
    r"(?P<args>.*?)\s*(?P<esc_end>\*/)?(?P<self>/)?[%>]\}\}",
    re.S,
)
ARG_RE = re.compile(r"""(?:([\w-]+)=)?("(?:[^"\\]|\\.)*"|`[^`]*`|[^\s"`]+)""")


@dataclass
class Shortcode:
    name: str
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)
    inner: list[Node] | None = None  # None: standalone; list: paired (possibly empty)

    def get(self, key: str | int, default: str = "") -> str:
        if isinstance(key, int):
            return self.args[key] if key < len(self.args) else default
        return self.kwargs.get(key, default)


Node = str | Shortcode


def parse_args(raw: str) -> tuple[list[str], dict[str, str]]:
    args, kwargs = [], {}
    for key, value in ARG_RE.findall(raw):
        if value.startswith('"'):
            value = re.sub(r"\\(.)", r"\1", value[1:-1])
        elif value.startswith("`"):
            value = value[1:-1]
        if key:
            kwargs[key] = value
        else:
            args.append(value)
    return args, kwargs


def parse(text: str) -> list[Node]:
    root: list[Node] = []
    # Open shortcodes that may still be closed later, each with the nodes collected since.
    stack: list[tuple[Shortcode, list[Node]]] = []

    def current() -> list[Node]:
        return stack[-1][1] if stack else root

    def unwind_top() -> None:
        # The top shortcode never got a closing tag: it is standalone, and what we
        # collected after it belongs to its parent.
        sc, children = stack.pop()
        current().append(sc)
        current().extend(children)

    pos = 0
    for m in TAG_RE.finditer(text):
        if m.start() > pos:
            current().append(text[pos : m.start()])
        pos = m.end()

        if m["esc"]:
            literal = m[0].replace("/*", "", 1)
            literal = literal[: literal.rfind("*/")] + literal[literal.rfind("*/") + 2 :]
            current().append(literal)
            continue

        name = m["name"]
        if m["close"]:
            depth = next(
                (i for i in range(len(stack) - 1, -1, -1) if stack[i][0].name == name), None
            )
            if depth is None:
                continue  # stray closing tag
            while len(stack) - 1 > depth:
                unwind_top()
            sc, children = stack.pop()
            sc.inner = children
            current().append(sc)
            continue

        args, kwargs = parse_args(m["args"])
        sc = Shortcode(name, args, kwargs)
        if m["self"]:
            current().append(sc)
        else:
            stack.append((sc, []))

    if pos < len(text):
        current().append(text[pos:])
    while stack:
        unwind_top()
    return root
