"""The visible-text layer: a sealed text as a reader sees it once its Markdown is rendered.

The most common path of a model's answer is a copy out of a chat window, where the Markdown is already rendered:
no ``**``, ``#`` or list markers, links as their text, paragraphs joined differently. Against the content layer
(canonical form v4) such a copy is MODIFIED although every visible word is the sealed one. The record therefore also
carries the digest of the visible text and its sentence blocks, and the verifier compares a candidate with them.

Two sides, deliberately unequal:
  sealer    ``project_source``: renders the canonical text as Markdown (CommonMark with GFM tables and
            strikethrough; raw HTML is shown as text, as chat windows do), keeps what a reader sees, then
            ``normalize``. Needs markdown-it-py; without it the layer is simply not written.
  verifier  ``normalize`` only: the candidate is taken literally, never rendered. Markup in a candidate stays text,
            so a Markdown source whose link target was changed is not its rendered copy (MODIFIED), and a rendered
            copy carries no link targets to change.
``normalize``: canonical form v4, then per line one list marker at its start (bullets - * + and the round bullets
a copy may show, an ordinal ``12.`` or ``12)``, a task box ``[ ]`` / ``[x]`` or the ballot box a copy shows for it),
then every run of whitespace one space.
Not covered, by construction: formatting (emphasis, headings, code fences, tables as such), line breaks, list
numbering, image alt texts. A match is reported as its own verdict, VERIFIED_TEXT.
"""
from __future__ import annotations

import re

from .canon import CANON_V4, canonicalize, segment

PLAIN_V1 = "ep1-plain/v1:canon-v4+commonmark-gfm-visible+list-marker+ws"
PLAIN_ID = PLAIN_V1                      # new records
SUPPORTED_PLAIN = (PLAIN_V1,)
# written as code points: the source stays ASCII; the JS implementation uses the same classes
_BULLETS = "-*+" + "".join(map(chr, (0x2022, 0x25E6, 0x25AA, 0x2023, 0x25CF, 0x00B7)))
_BOXES = "".join(map(chr, (0x2610, 0x2611, 0x2612)))       # the ballot boxes a copy of a task list may show
_MARKER = re.compile("^[ \t]*(?:(?:[" + re.escape(_BULLETS) + "]|[0-9]{1,9}[.)])[ \t]+)?"
                     "(?:(?:\\[[ xX]\\]|[" + _BOXES + "])[ \t]+)?")
_WS = re.compile("[ \t\n\x0b\x0c\u1680\u2000-\u200a\u2028\u2029\u205f\u3000]+")


def _lines(text: str) -> list[str]:
    """Visible lines: canonical form v4; per line one list marker at its start removed, whitespace runs one space;
    empty lines dropped."""
    out: list[str] = []
    for line in canonicalize(text, CANON_V4).split("\n"):
        line = _WS.sub(" ", _MARKER.sub("", line, count=1)).strip(" ")
        if line:
            out.append(line)
    return out


def _check(form: str) -> None:
    if form not in SUPPORTED_PLAIN:
        raise ValueError(f"unsupported visible-text form {form!r}")


def normalize(text: str, form: str = PLAIN_ID) -> str:
    """The visible text of a candidate taken literally: its visible lines joined by one space (the digest does not
    depend on how a copy breaks its lines)."""
    _check(form)
    return " ".join(_lines(text))


def segment_visible(text: str, form: str = PLAIN_ID) -> tuple[str, list[tuple[int, int]]]:
    """(normalize(text), blocks): sentence-like blocks (canon.segment) inside each visible line, as spans of the
    normalized text. A line is a paragraph, a heading, a list item, a table row or a line of code, so a change is
    located to its item even where list items and cells end without a full stop."""
    _check(form)
    lines, spans, pos = _lines(text), [], 0
    for line in lines:
        spans.extend((pos + a, pos + b) for a, b in segment(line))
        pos += len(line) + 1
    return " ".join(lines), spans


def _inline(children) -> str:
    out: list[str] = []
    for t in children or ():                    # an image's alt text sits inside its own token: not visible text
        if t.type in ("text", "code_inline"):
            out.append(t.content)
        elif t.type == "softbreak":             # rendered as a space: the paragraph stays one line
            out.append(" ")
        elif t.type == "hardbreak":
            out.append("\n")
    return "".join(out)


def render_visible(md: str) -> str:
    """What a reader sees of ``md`` rendered as Markdown, one visible line per block: a paragraph, heading or list
    item, a table row (its cells), a line of code."""
    from markdown_it import MarkdownIt
    parser = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])
    lines: list[str] = []
    row: list[str] | None = None
    for t in parser.parse(md):
        if t.type == "tr_open":
            row = []
        elif t.type == "tr_close":
            lines.append(" ".join(row or []))
            row = None
        elif t.type == "inline":
            (row if row is not None else lines).append(_inline(t.children))
        elif t.type in ("fence", "code_block"):
            lines.append(t.content)
    return "\n".join(lines)


def project_source(ctext: str, form: str = PLAIN_ID) -> str:
    """The sealer's side: the visible text of a canonical text rendered as Markdown."""
    return normalize(render_visible(ctext), form)


def segment_source(ctext: str, form: str = PLAIN_ID) -> tuple[str, list[tuple[int, int]]]:
    """The sealer's side of segment_visible: (visible text, blocks) of a canonical text rendered as Markdown."""
    return segment_visible(render_visible(ctext), form)


def available() -> bool:
    try:
        import markdown_it  # noqa: F401
        return True
    except ImportError:
        return False
