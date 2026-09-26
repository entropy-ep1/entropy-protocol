"""Presentation typography: quotes, apostrophes, dashes and the ellipsis set for a language, by substitution only.

Sealed text is generated in canonical form (straight quotes, two hyphens for a long dash, three full stops; see
canongen.py). A reader may want the typography of their language. ``smarten`` replaces characters with forms that
canonical form v4 folds back (canon.py): a double quote with a curly quote or a guillemet, a single quote with a curly
one, "--" with an em dash, a spaced "-" with an en dash, "..." with the ellipsis. The smartened text therefore has the
canonical form of the sealed text and verifies as it does. Code (fenced blocks, inline spans) and Markdown structure
(list markers, table rules) are left alone, and the result is checked: if its canonical form differed in any way, the
text is returned unchanged.
"""
from __future__ import annotations

import re

from .canon import CANON_ID, canonicalize

_Q = {  # language -> (outer open, outer close, inner open, inner close, apostrophe); every form folds back
    "en": (0x201C, 0x201D, 0x2018, 0x2019, 0x2019),
    "ru": (0x00AB, 0x00BB, 0x201E, 0x201C, 0x2019),
    "uk": (0x00AB, 0x00BB, 0x201E, 0x201C, 0x2019),
    "be": (0x00AB, 0x00BB, 0x201E, 0x201C, 0x2019),
    "de": (0x201E, 0x201C, 0x201A, 0x2018, 0x2019),
    "pl": (0x201E, 0x201D, 0x00AB, 0x00BB, 0x2019),
    "fr": (0x00AB, 0x00BB, 0x201C, 0x201D, 0x2019),
    "es": (0x00AB, 0x00BB, 0x201C, 0x201D, 0x2019),
    "it": (0x00AB, 0x00BB, 0x201C, 0x201D, 0x2019),
}
EM, EN, ELLIPSIS = chr(0x2014), chr(0x2013), chr(0x2026)
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)            # fenced blocks and inline code spans stay as they are
_OPENERS = set(" \t\n([{<-/") | {chr(0x2014), chr(0x2013)}


def _prose(text: str, lang: str) -> str:
    o1, c1, o2, c2, apo = (chr(c) for c in _Q.get(lang, _Q["en"]))
    out: list[str] = []
    depth = 0                                               # open double quotes: outer, inner, outer, ...
    for i, ch in enumerate(text):
        prev = text[i - 1] if i else "\n"
        nxt = text[i + 1] if i + 1 < len(text) else "\n"
        if ch == '"':
            if prev in _OPENERS or prev in (o1, o2, chr(0x2018)):
                out.append(o1 if depth % 2 == 0 else o2)
                depth += 1
            else:
                depth = max(0, depth - 1)
                out.append(c1 if depth % 2 == 0 else c2)
        elif ch == "'":
            if prev.isalnum() and (nxt.isalnum() or nxt in " \t\n.,;:!?)"):
                out.append(apo)                             # an apostrophe: don't, the 90's, dogs'
            elif prev in _OPENERS or prev in (o1, o2):
                out.append(chr(0x2018))                     # an opening single quote
            else:
                out.append(chr(0x2019))
        else:
            out.append(ch)
    s = "".join(out)
    s = re.sub(r"(?<=[^\s-])--(?=[^\s-])|(?<=\S) -- (?=\S)", lambda m: m.group(0).replace("--", EM), s)
    s = re.sub(r"(?<=[^\s|-]) - (?=[^\s-])", " " + EN + " ", s)     # a spaced hyphen between words, not a bullet
    s = re.sub(r"(?<!\.)\.\.\.(?!\.)", ELLIPSIS, s)
    return s


def smarten(text: str, lang: str = "en", canon: str = CANON_ID) -> str:
    """Typographic quotes, apostrophes, dashes and ellipses for ``lang``, with the canonical form unchanged."""
    lang = (lang or "en").split("-")[0].lower()
    parts, last = [], 0
    for m in _CODE.finditer(text):
        parts.append(_prose(text[last:m.start()], lang))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_prose(text[last:], lang))
    out = "".join(parts)
    return out if canonicalize(out, canon) == canonicalize(text, canon) else text
