"""Canonical text form shared by sealer and verifier.

A record is measured on the canonical form of the final text, so anything the canonical form
erases — carriage returns, byte-order marks, invisible format characters, an embedded C2PA text
manifest, no-break spaces, trailing spaces at line ends, blank lines around a paste, and (v4) the
typographic substitutions word processors make on paste — cannot break verification. The transform is
deterministic and idempotent:
  1. Unicode NFC
  2. delete every U+000D (CR) and U+FEFF (BOM): LF is the only line break, so CRLF, a naive
     LF -> CRLF conversion of text that already holds CRLF (CR CR LF) and stray CRs all agree
  3. delete invisible format characters that carry no content: U+200B (zero-width space),
     U+2060 (word joiner), U+00AD (soft hyphen). U+200C/U+200D (ZWNJ/ZWJ) are kept: scripts and
     emoji sequences depend on them
  4. delete embedded payloads written as variation selectors (U+FE00-FE0F, U+E0100-E01EF): a U+FEFF
     followed by 8 or more selectors (C2PA 2.4 A.8 text-manifest wrapper) is removed as a unit, then
     any remaining run of 8 or more selectors; a single selector after a base character (e.g. an emoji
     presentation selector) is content and is kept
  5. U+00A0, U+2007, U+202F (no-break spaces) -> U+0020
  6. (v4) typographic folding, the inverse of what word processors do on paste (and of Ukrainian
     orthography): double quotes U+201C U+201D U+201E U+201F U+00AB U+00BB -> U+0022; apostrophes and single
     quotes U+2018 U+2019 U+201A U+201B U+02BC -> U+0027; hyphens and en dash U+2010-U+2013, minus U+2212 ->
     U+002D; em dash U+2014 and horizontal bar U+2015 -> "--" (Word turns " - " into an en dash and "--" into
     an em dash); ellipsis U+2026 -> "..." (Word turns "..." into an ellipsis)
  7. delete spaces and tabs that end a line or the text; strip the text
Blocks (content layer): sentence-like spans, split at line breaks and after . ! ? followed by
whitespace; v4 also after the full stops of scripts written without spaces (Chinese/Japanese, Arabic,
Devanagari, Ethiopic, Armenian) and keeps closing quotes and brackets with their sentence.
Records name their version; a verifier reads every supported version.
"""
from __future__ import annotations

import unicodedata

CANON_V3 = "ep1-canon/v3:nfc+del-cr-bom-zw-shy+del-vs-runs+nbsp+rstrip-lines+strip"
CANON_V4 = "ep1-canon/v4:nfc+del-cr-bom-zw-shy+del-vs-runs+nbsp+typo+rstrip-lines+strip;seg2"
# v5: the deletions before NFC. In v4 a deleted character could leave combining marks adjacent in a non-canonical
# order, or a base and a combining mark that NFC would compose: canonicalize(canonicalize(x)) != canonicalize(x), so a
# zero-width character inserted between such marks broke verification after transport (found by fuzzing, 27.10.2026:
# U+0E48 U+0301 ZWJ U+0E48 U+0301). With NFC after every deletion, and only starters replaced or removed afterwards,
# the form is idempotent by construction (tests/test_fuzz.py checks it).
CANON_V5 = "ep1-canon/v5:del-cr-bom-zw-shy+del-vs-runs+nfc+nbsp+typo+rstrip-lines+strip;seg2"
CANON_ID = CANON_V5                     # new records
SUPPORTED = (CANON_V3, CANON_V4, CANON_V5)
_TRAILING = " \t"
_DELETE = {"\r", "﻿", "​", "⁠", "­"}
_SPACE = {" ", " ", " "}
_VS_RUN = 8
# v4 typographic folding (code points written as numbers: the source stays ASCII)
_FOLD = {**{chr(c): '"' for c in (0x201C, 0x201D, 0x201E, 0x201F, 0x00AB, 0x00BB)},
         **{chr(c): "'" for c in (0x2018, 0x2019, 0x201A, 0x201B, 0x02BC)},
         **{chr(c): "-" for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2212)},
         **{chr(c): "--" for c in (0x2014, 0x2015)},
         chr(0x2026): "..."}
# v4 segmentation: full stops of scripts written without spaces, closing quotes and brackets
_SCRIPT_STOPS = frozenset(map(chr, (0x3002, 0xFF01, 0xFF1F, 0xFF0E, 0x061F, 0x06D4, 0x0964, 0x0965, 0x1362, 0x0589)))
_CLOSERS = frozenset(")]}\"'" + "".join(map(chr, (0x300D, 0x300F, 0x3011, 0x300B, 0x3009, 0xFF09, 0xFF3D))))


def _is_vs(ch: str) -> bool:
    o = ord(ch)
    return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF


def _check(version: str) -> None:
    if version not in SUPPORTED:
        raise ValueError(f"unsupported canonical form {version!r}")


def canonicalize(text: str, version: str = CANON_ID) -> str:
    return canonicalize_with_map(text, version)[0]


def segment(ctext: str, version: str = CANON_ID) -> list[tuple[int, int]]:
    """Content-defined blocks of canonical text: sentence-like spans, split at line breaks and after . ! ?
    followed by whitespace (or at the end); v4 also after the full stops of scripts written without spaces,
    with closing quotes and brackets kept in the sentence. Surrounding whitespace excluded. Deterministic."""
    _check(version)
    spans: list[tuple[int, int]] = []
    start, n = 0, len(ctext)
    v4 = version in (CANON_V4, CANON_V5)                                # seg2

    def emit(a: int, b: int) -> None:
        piece = ctext[a:b]
        if piece.strip():
            s = a + (len(piece) - len(piece.lstrip()))
            e = b - (len(piece) - len(piece.rstrip()))
            spans.append((s, e))

    i = 0
    while i < n:
        ch = ctext[i]
        if ch == "\n":
            emit(start, i)
            start = i + 1
        elif not v4:
            if ch in ".!?" and (i + 1 == n or ctext[i + 1].isspace()):
                emit(start, i + 1)
                start = i + 1
        elif ch in ".!?" or ch in _SCRIPT_STOPS:
            j = i + 1
            while j < n and ctext[j] in _CLOSERS:
                j += 1
            if ch in _SCRIPT_STOPS or j == n or ctext[j].isspace():
                emit(start, j)
                start = j
                i = j - 1
        i += 1
    emit(start, n)
    return spans


def block_digest(piece: str) -> str:
    import hashlib
    return hashlib.sha256(piece.encode()).hexdigest()[:16]


def canonicalize_with_map(text: str, version: str = CANON_ID) -> tuple[str, dict[int, int]]:
    """Return (canonical_text, index_map).

    ``index_map`` maps indices of the source to indices of the canonical text - for v3 and v4 indices of
    ``unicodedata.normalize("NFC", text)``, for v5 indices of ``text`` itself; characters removed by
    canonicalization are absent from the map (v4, v5: a character folded into several, an ellipsis or an em dash,
    maps to the first of them; v5: characters NFC composes map to the composed character).
    """
    _check(version)
    if version == CANON_V5:
        return _canonicalize_v5(text)
    v4 = version == CANON_V4
    src = unicodedata.normalize("NFC", text)
    kept: list[tuple[str, int]] = []
    k, n = 0, len(src)
    while k < n:  # steps 2-3, 5 and 6: deletions, no-break spaces -> space, typographic folding
        ch = src[k]
        if ch == "﻿":     # a U+FEFF that opens a selector run is a text-manifest wrapper: drop both
            e = k + 1
            while e < n and _is_vs(src[e]):
                e += 1
            k = e if e - (k + 1) >= _VS_RUN else k + 1
            continue
        if ch not in _DELETE:
            if v4 and ch in _FOLD:          # a folded character maps to the first character it becomes
                rep = _FOLD[ch]
                kept.extend([(rep[0], k)] + [(c, -1) for c in rep[1:]])
            else:
                kept.append((" " if ch in _SPACE else ch, k))
        k += 1
    tmp: list[str] = []
    tmap: list[int] = []
    k, n = 0, len(kept)
    while k < n:  # step 4, after the deletions so that runs split by a deleted character are seen whole
        if _is_vs(kept[k][0]):
            e = k
            while e < n and _is_vs(kept[e][0]):
                e += 1
            if e - k >= _VS_RUN:
                k = e
                continue
        tmp.append(kept[k][0])
        tmap.append(kept[k][1])
        k += 1
    res: list[str] = []
    rmap: list[int] = []
    k, length = 0, len(tmp)
    while k < length:  # drop spaces/tabs that end a line or the text
        if tmp[k] in _TRAILING:
            e = k
            while e < length and tmp[e] in _TRAILING:
                e += 1
            if e == length or tmp[e] == "\n":
                k = e
                continue
        res.append(tmp[k])
        rmap.append(tmap[k])
        k += 1
    joined = "".join(res)
    lead = len(joined) - len(joined.lstrip())
    canon = joined.strip()
    index_map = {src_i: ci - lead for ci, src_i in enumerate(rmap)
                 if src_i >= 0 and lead <= ci < lead + len(canon)}
    return canon, index_map


def _deleted(text: str) -> list[tuple[str, int]]:
    """The deletions (CR, BOM, zero-widths, soft hyphen; a U+FEFF opening a run of selectors, with the run; runs of
    eight or more variation selectors), as (character, source index) of what is kept."""
    kept: list[tuple[str, int]] = []
    k, n = 0, len(text)
    while k < n:
        ch = text[k]
        if ch == "\ufeff":
            e = k + 1
            while e < n and _is_vs(text[e]):
                e += 1
            k = e if e - (k + 1) >= _VS_RUN else k + 1
            continue
        if ch not in _DELETE:
            kept.append((ch, k))
        k += 1
    out: list[tuple[str, int]] = []
    k, n = 0, len(kept)
    while k < n:                      # runs split by a deleted character are seen whole
        if _is_vs(kept[k][0]):
            e = k
            while e < n and _is_vs(kept[e][0]):
                e += 1
            if e - k >= _VS_RUN:
                k = e
                continue
        out.append(kept[k])
        k += 1
    return out


_SECONDS: frozenset | None = None


def _composing_seconds() -> frozenset:
    """Every character that canonical composition can join to what precedes it, starters included (Hangul vowels
    and trailing consonants, Indic and other vowel signs with combining class 0): built once, when first needed."""
    global _SECONDS
    if _SECONDS is None:
        seconds = set(range(0x1161, 0x1176)) | set(range(0x11A8, 0x11C3))
        for cp in range(0x110000):
            d = unicodedata.decomposition(chr(cp))
            if d and not d.startswith("<"):
                parts = d.split()
                if len(parts) == 2:
                    seconds.add(int(parts[1], 16))
        _SECONDS = frozenset(seconds)
    return _SECONDS


def _nfc_with_map(chars: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """NFC of the characters, each result character carrying the source index of the first character of its
    normalization segment (a segment starts at a starter that nothing before it can compose or reorder with)."""
    s = "".join(c for c, _ in chars)
    if unicodedata.is_normalized("NFC", s):
        return chars
    seconds = _composing_seconds()

    def boundary(ch: str) -> bool:
        if unicodedata.combining(ch) or ord(ch) in seconds:
            return False
        first = unicodedata.normalize("NFD", ch)[0]
        return unicodedata.combining(first) == 0

    out: list[tuple[str, int]] = []
    a, n = 0, len(chars)
    while a < n:
        b = a + 1
        while b < n and not boundary(chars[b][0]):
            b += 1
        seg = "".join(c for c, _ in chars[a:b])
        norm = unicodedata.normalize("NFC", seg)
        if norm == seg:
            out.extend(chars[a:b])
        else:
            out.extend((c, chars[a][1]) for c in norm)
        a = b
    if "".join(c for c, _ in out) != unicodedata.normalize("NFC", s):
        raise AssertionError("normalization segments do not compose to the NFC of the whole")
    return out


def _canonicalize_v5(text: str) -> tuple[str, dict[int, int]]:
    chars = _nfc_with_map(_deleted(text))
    tmp: list[str] = []
    tmap: list[int] = []
    for ch, i in chars:                                     # no-break spaces -> space, typographic folding
        if ch in _FOLD:
            rep = _FOLD[ch]
            tmp.extend(rep)
            tmap.extend([i] + [-1] * (len(rep) - 1))
        else:
            tmp.append(" " if ch in _SPACE else ch)
            tmap.append(i)
    res: list[str] = []
    rmap: list[int] = []
    k, length = 0, len(tmp)
    while k < length:                                       # drop spaces/tabs that end a line or the text
        if tmp[k] in _TRAILING:
            e = k
            while e < length and tmp[e] in _TRAILING:
                e += 1
            if e == length or tmp[e] == "\n":
                k = e
                continue
        res.append(tmp[k])
        rmap.append(tmap[k])
        k += 1
    joined = "".join(res)
    lead = len(joined) - len(joined.lstrip())
    canon = joined.strip()
    index_map: dict[int, int] = {}
    for ci, src_i in enumerate(rmap):
        if src_i >= 0 and lead <= ci < lead + len(canon):
            index_map.setdefault(src_i, ci - lead)          # a composed segment maps to its first character
    return canon, index_map
