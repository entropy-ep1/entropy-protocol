"""The visible-text layer (plain.py): a rendered copy of a sealed Markdown text verifies as VERIFIED_TEXT."""
import types

import pytest

pytest.importorskip("markdown_it")

from entropy_protocol import crypto  # noqa: E402
from entropy_protocol.canon import canonicalize  # noqa: E402
from entropy_protocol.plain import normalize, project_source  # noqa: E402
from entropy_protocol.record import loads  # noqa: E402

MASTER = bytes(range(32))
BULLET, WHITE, BOX, TICK = chr(0x2022), chr(0x25E6), chr(0x2610), chr(0x2611)
SOURCE = ("# Quarterly review\n\nRevenue grew **12%** over the quarter, led by the [new product](https://example.com/p).\n\n"
          "Three things mattered:\n\n1. pricing\n2. *retention*, which rose to 91%\n3. `on-call` load\n\n"
          "| Region | Growth |\n|---|---|\n| EU | 9% |\n| US | 14% |\n\n> Keep the course.\n\nNext review in March.")
COPY = ("Quarterly review\nRevenue grew 12% over the quarter, led by the new product.\nThree things mattered:\n"
        "1.\tpricing\n2.\tretention, which rose to 91%\n3.\ton-call load\nRegion\tGrowth\nEU\t9%\nUS\t14%\n"
        "Keep the course.\nNext review in March.")


def test_the_visible_text_of_markdown_is_what_a_copy_gives_back():
    cases = [
        (SOURCE, COPY),
        ("- one\n- two\n  - nested\n\n```python\nx = 2*3\n```\n\n---\n\nEnd: 2*3=6, a_b_c.",
         f"{BULLET} one\n{BULLET} two\n    {WHITE} nested\nx = 2*3\nEnd: 2*3=6, a_b_c."),
        ("Soft\nbreak, ~~struck~~ text, ![alt](i.png) picture, <b>raw</b> &amp; \\*escaped\\*.",
         "Soft break, struck text,  picture, <b>raw</b> & *escaped*."),
        ("- [ ] todo\n- [x] done", f"{BOX} todo\n{TICK} done"),
    ]
    for md, copy in cases:
        assert project_source(canonicalize(md)) == normalize(copy), md


def test_a_candidate_is_taken_literally():
    """The verifier never renders: markup in a candidate stays text, so a Markdown source whose link target was
    changed is not the visible text of the sealed one, and literal asterisks and underscores are kept."""
    sealed = project_source(canonicalize("See [the report](https://good.example/r) now."))
    assert sealed == "See the report now."
    assert normalize("See [the report](https://evil.example/r) now.") != sealed
    assert normalize("See the report now.") == sealed
    assert project_source("2*3 is 6 and snake_case stays") == "2*3 is 6 and snake_case stays"


def _backend():
    return types.SimpleNamespace(model_ref="fake/model@local", weights_digest="a" * 64, tokenizer_digest="b" * 64,
                                 components={}, environment={}, config_class="fp32", device="cpu", dtype="float32",
                                 position_stats=lambda ctext, offsets, k: [None] * len(offsets))


def _seal(text: str, **params):
    from entropy_protocol.seal import SealParams, _Measured, finish
    m = _Measured(raw=text, ctext=canonicalize(text), anchors=[], props=[], dropped_anchor=0, tokens=0, enforced=0)
    return finish(_backend(), SealParams(**params), m, [], MASTER, "doc-plain-1", salt=bytes(16), nonce=bytes(12))


def _verify(candidate: str, token: str):
    from entropy_protocol.verify import verify
    return verify(_backend(), candidate, token, crypto.log_key(MASTER, crypto.token_doc_id(token)))


def test_a_rendered_copy_verifies_as_its_text():
    s = _seal(SOURCE)
    rec = loads(crypto.decrypt_token(crypto.log_key(MASTER, s.doc_id), s.token))
    assert rec.header.plain.startswith("ep1-plain/v1") and rec.plain_blocks
    assert _verify(SOURCE, s.token).verdict == "VERIFIED"
    r = _verify(COPY, s.token)
    assert r.verdict == "VERIFIED_TEXT" and r.text_match is True and not r.changed_ranges
    edited = COPY.replace("rose to 91%", "rose to 97%")
    r = _verify(edited, s.token)
    assert r.verdict == "MODIFIED" and r.text_match is False
    view = r.text_view
    assert len(view["changed_ranges"]) == 1                      # the rendered copy: exactly the changed sentence
    a, b = view["changed_ranges"][0]
    assert "97%" in view["text"][a:b] and "pricing" not in view["text"][a:b]
    assert len(r.changed_ranges) > 1                             # the canonical view flags every line with markup
    swapped = SOURCE.replace("https://example.com/p", "https://evil.example/p")
    assert _verify(swapped, s.token).verdict == "MODIFIED"      # a changed link target is caught, rendered or not


def test_without_the_layer_a_copy_stays_modified_and_the_record_keeps_its_bytes():
    s = _seal(SOURCE, plain_layer=False)
    rec = loads(crypto.decrypt_token(crypto.log_key(MASTER, s.doc_id), s.token))
    assert rec.header.plain == "" and rec.plain_blocks == ()
    r = _verify(COPY, s.token)
    assert r.verdict == "MODIFIED" and r.text_match is None and r.text_view is None
