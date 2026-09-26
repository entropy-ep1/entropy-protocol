"""Unit tests that need no language model."""
import pytest

from entropy_protocol import crypto
from entropy_protocol.canon import canonicalize, canonicalize_with_map
from entropy_protocol.predicates import boundary, holds, margin
from entropy_protocol.record import HIGH, LOW, Header, Record, dumps, loads

SAMPLES = [
    " The council met on Tuesday.\nIt voted twice.\n\nNothing else happened.",
    "Line one  \nLine two\t\n\n\nLine three",
    "A non-breaking space and caf\u00e9 au lait.",
    "Code already in DOS form:\r\nif (x) {\r\n  y();\r\n}\r\n\r\nDone.",      # CRLF inside the sealed text
    "\ufeffBOM first, then French typography\u202f: oui\u00a0!",
    "Soft\u00adhyphen, zero\u200bwidth, word\u2060joiner; emoji \u2764\ufe0f and ZWJ family \U0001F468\u200d\U0001F469.",
]


def transport(t: str) -> str:
    t = t.replace("\n", "\r\n").replace("\r\n", "  \r\n").replace(" ", "\u00a0", 1)
    return "\n" + t + "\n\n"


@pytest.mark.parametrize("s", SAMPLES)
def test_canon_idempotent_and_transport_invariant(s):
    c = canonicalize(s)
    assert canonicalize(c) == c
    assert canonicalize(transport(s)) == c


@pytest.mark.parametrize("s", SAMPLES)
def test_canon_map_points_at_same_characters(s):
    import unicodedata
    src = unicodedata.normalize("NFC", s)
    c, m = canonicalize_with_map(s)
    for i, ci in m.items():
        assert src[i] not in "\r\ufeff\u200b\u2060\u00ad"         # deleted characters are not mapped
        assert c[ci] == (" " if src[i] in "\u00a0\u2007\u202f" else src[i])


def test_canon_naive_crlf_conversions_agree():
    t = "Intro.\r\n\r\nfor (;;) {\r\n  run();\r\n}\nEnd."
    c = canonicalize(t)
    assert "\r" not in c and "\ufeff" not in c
    for moved in (t.replace("\n", "\r\n"),                      # naive LF -> CRLF: CR CR LF
                  "\n" + t.replace("\n", "  \n") + "\n\n",          # e-mail style trailing spaces
                  "\ufeff" + t.replace("\r\n", "\n"),             # unix line ends plus a BOM
                  t.replace(" ", "\u202f", 2)):                     # narrow no-break spaces
        assert canonicalize(moved) == c


def test_canon_v3_drops_embedded_manifests_and_invisible_format_characters():
    def c2pa_wrapper(payload: bytes) -> str:          # C2PA 2.4 A.8: U+FEFF, then one variation selector per byte
        vs = [chr(0xFE00 + b) if b < 16 else chr(0xE0100 + b - 16)
              for b in b"C2PATXT\x00\x01" + len(payload).to_bytes(4, "big") + payload]
        return "\ufeff" + "".join(vs)
    t = "The council met on Tuesday.\nIt voted twice \u2764\ufe0f"
    assert canonicalize(t + c2pa_wrapper(bytes(range(40)))) == canonicalize(t)
    assert canonicalize(t).endswith("\u2764\ufe0f")                  # a single selector is content
    assert canonicalize("in\u00advisible zero\u200bwidth") == "invisible zerowidth"
    assert canonicalize("\U0001F468\u200d\U0001F469") == "\U0001F468\u200d\U0001F469"   # ZWJ is kept
    split = "a" + "\ufe01" * 5 + "\u200b" + "\ufe02" * 5 + "b"     # runs joined by a deleted character
    assert canonicalize(split) == "ab" and canonicalize(canonicalize(split)) == "ab"


def test_token_roundtrip_tamper_and_keys():
    master = crypto.new_master_key()
    doc = crypto.new_doc_id()
    tok = crypto.encrypt_token(crypto.log_key(master, doc), doc, b"payload")
    assert crypto.decrypt_token(crypto.log_key(master, doc), tok) == b"payload"
    bad = tok[:-3] + ("A" if tok[-3] != "A" else "B") + tok[-2:]
    with pytest.raises(crypto.TokenError):
        crypto.decrypt_token(crypto.log_key(master, doc), bad)
    with pytest.raises(crypto.TokenError):
        crypto.decrypt_token(crypto.log_key(crypto.new_master_key(), doc), tok)
    tok_v, k_v = crypto.reissue_for_verifier(master, tok, "court-1")
    assert crypto.decrypt_token(k_v, tok_v) == b"payload"
    with pytest.raises(crypto.TokenError):          # verifier key opens only its own re-issued token
        crypto.decrypt_token(k_v, tok)
    other = crypto.new_doc_id()
    tok_o = crypto.encrypt_token(crypto.log_key(master, other), other, b"x")
    with pytest.raises(crypto.TokenError):
        crypto.decrypt_token(k_v, tok_o)
    assert crypto.verifier_key(master, doc, "court-1") != crypto.verifier_key(master, doc, "court-2")


def test_bitstream_is_keyed_and_per_document():
    kb = crypto.bits_key(crypto.new_master_key())
    a = [crypto.bit(kb, "doc-a", j) for j in range(256)]
    b = [crypto.bit(kb, "doc-b", j) for j in range(256)]
    assert a != b and 80 < sum(a) < 176
    biased = [crypto.bit(kb, "doc-a", j, low_share=0.75) for j in range(2000)]
    assert 0.20 < sum(biased) / 2000 < 0.30                # about a quarter of checkpoints get HIGH


def test_watermark_prf_density_balance_and_exact_binomial_tail():
    import os
    from fractions import Fraction
    from entropy_protocol.detect import binomial_tail
    kw = crypto.wm_key(crypto.new_master_key())
    decisions = [crypto.wm_decide(kw, tuple(os.urandom(16)), 5) for _ in range(20000)]
    chosen = [d for d in decisions if d is not None]
    assert 0.18 < len(chosen) / len(decisions) < 0.22                 # one checkpoint in five contexts
    assert 0.47 < sum(chosen) / len(chosen) < 0.53                    # LOW and HIGH equally likely
    assert crypto.wm_decide(kw, (1, 2, 3, 4), 5) == crypto.wm_decide(kw, [1, 2, 3, 4], 5)
    assert binomial_tail(10, 10) == 1 / 1024 and binomial_tail(10, 0) == 1.0 and binomial_tail(3, 4) == 0.0
    assert Fraction(binomial_tail(20, 15)).limit_denominator(1 << 20) == Fraction(21700, 1 << 20)


def test_record_roundtrip_and_privacy():
    h = Header(model="m@r", weights_digest="w", tokenizer_digest="t", canon="c", context="bos+text", k=10,
               interval=5, tol=0.0, min_entries=8, indet_frac=0.2, text_digest="d", engine="e")
    rec = Record(h, ((3, LOW), (17, HIGH)))
    assert loads(dumps(rec)) == rec
    raw = dumps(rec)
    for word in (b"council", b"Tuesday"):                # entries are offsets and properties only
        assert word not in raw
    with pytest.raises(ValueError):
        loads(dumps(Record(h, ((3, 7),))))
    from dataclasses import replace
    hc = replace(h, tol_space="logit", cal={"fp32": 0.05, "fp16": 0.4})
    assert loads(dumps(Record(hc, ((3, LOW),)))).header.cal == {"fp32": 0.05, "fp16": 0.4}
    for bad in (replace(h, tol_space="rank"), replace(h, cal={"fp32": -1.0})):
        with pytest.raises(ValueError):
            loads(dumps(Record(bad, ((3, LOW),))))


def test_content_diff_attributes_repeated_sentences_to_the_right_copy():
    from entropy_protocol.canon import block_digest, segment
    from entropy_protocol.verify import _content_diff
    sealed = "Intro here. The fire started at noon. Crews arrived. The fire started at noon. It ended."
    blocks = [(s, e, block_digest(sealed[s:e])) for s, e in segment(sealed)]
    first = sealed.index("The fire")
    cand = sealed[:first] + sealed[first + len("The fire started at noon. "):]        # delete the FIRST copy
    changed, points = _content_diff(cand, blocks)
    assert changed == [] and points == [len("Intro here.")]
    second = sealed.rindex("The fire")
    cand2 = sealed[:second] + sealed[second + len("The fire started at noon. "):]     # delete the SECOND copy
    assert _content_diff(cand2, blocks) == ([], [len(sealed[:second].rstrip())])
    cand3 = sealed.replace("Crews arrived.", "Crews left.")
    changed3, points3 = _content_diff(cand3, blocks)
    s3 = cand3.index("Crews left.")
    assert changed3 == [[s3, s3 + len("Crews left.")]] and points3 == []


def test_segment_blocks_are_deterministic_and_cover_sentences():
    from entropy_protocol.canon import segment
    t = "First sentence. Second one!\nThird line without stop\n\nFourth? Yes."
    spans = segment(t)
    assert [t[a:b] for a, b in spans] == ["First sentence.", "Second one!", "Third line without stop", "Fourth?", "Yes."]
    assert segment(canonicalize(transport(t))) == spans


def test_predicates_and_boundaries():
    top = [0.5, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.008, 0.006]   # k+1 = 11 values
    assert holds(LOW, 0, 10) and not holds(LOW, 1, 10)
    assert holds(HIGH, 10, 10) and not holds(HIGH, 9, 10)
    assert boundary(LOW, 0, top, 10) == 0.2 and boundary(LOW, 3, top, 10) == 0.5
    assert boundary(HIGH, 12, top, 10) == top[9] and boundary(HIGH, 2, top, 10) == top[10]
    assert margin(LOW, 0, 0.5, top, 10) == pytest.approx(0.3)
    import math
    logits = [math.log(x) for x in top]                  # logit space: the margin is a log-probability ratio
    assert margin(LOW, 0, logits[0], logits, 10) == pytest.approx(math.log(0.5 / 0.2))
