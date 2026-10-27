#!/usr/bin/env python3
"""Model-free test vectors for ep1: keys, keyed selection, canonicalization, record, token, commitment,
statements, Merkle log and receipts, exact binomial tails. Deterministic: fixed keys, nonces, salts, times.

    python3 spec/make_vectors.py            # writes spec/test-vectors/ep1-vectors.json
A second implementation reproduces every hex string byte for byte.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from entropy_protocol import cose, crypto, delivery, hpke, statements, tlog  # noqa: E402
from entropy_protocol.canon import CANON_V3, CANON_V4, CANON_V5, block_digest, canonicalize, segment  # noqa: E402
from entropy_protocol.stats import binomial_tail  # noqa: E402
from entropy_protocol.plain import PLAIN_V1, PLAIN_V2, normalize as visible_text, segment_source, segment_visible  # noqa: E402
from entropy_protocol.record import HIGH, LOW, Header, Record, dumps  # noqa: E402

MASTER = bytes(range(32))
DOC = "doc-tv-0001"
VERIFIER = "verifier-1"
NONCE_LOG = bytes(range(100, 112))
NONCE_VER = bytes(range(200, 212))
NONCE_LOG4 = bytes(range(150, 162))
NONCE_LOG5 = bytes(range(170, 182))
SALT = bytes(range(50, 66))
IAT = 1_789_000_000
SIGNER_SEED = bytes([7] * 32)
VERIFIER_SEED = bytes([9] * 32)
LOG_SEED = bytes([11] * 32)
HPKE_RECIPIENT_IKM = b"ep1-tv/hpke-recipient"
HPKE_EPHEMERAL_IKM = b"ep1-tv/hpke-ephemeral"
# RFC 9180 A.2.1 (base mode, DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, ChaCha20-Poly1305), sequence number 0
RFC9180_A21 = {"skRm": "8057991eef8f1f1af18f4a9491d16a1ce333f695d4db8e38da75975c4478e0fb",
               "enc": "1afa08d3dec047a643885163f1180476fa7ddb54c6a8029ea33f95796bf2ac4a",
               "info": "4f6465206f6e2061204772656369616e2055726e", "aad": "436f756e742d30",
               "ct": "1c5250d8034ec2b784ba2cfd69dbdb8af406cfe3ff938e131f0def8c8b60b4db21993c62ce81883d2dd1b51a28",
               "pt": "4265617574792069732074727574682c20747275746820626561757479"}

C2PA = "\ufeff" + "".join(chr(0xFE00 + b) if b < 16 else chr(0xE0100 + b - 16)
                          for b in b"C2PATXT\x00\x01" + (40).to_bytes(4, "big") + bytes(range(40)))
CANON_CASES = [
    "plain text",
    "  leading and trailing  \n\n",
    "line one  \r\nline two\t\r\n\r\nline three",
    "naive conversion\r\r\nof CRLF",
    "\ufeffBOM, NBSP\u00a0here, narrow\u202fno-break, figure\u2007space",
    "soft\u00adhyphen, zero\u200bwidth, word\u2060joiner",
    "emoji ❤\ufe0f kept, ZWJ family \U0001F468\u200d\U0001F469 kept",
    "manifest follows" + C2PA,
    "Café decomposed becomes NFC",
]
SEGMENT_TEXT = "First sentence. Second one!\nThird line without stop\n\nFourth? Yes."
# the visible-text layer (plain.py), the verifier's side: candidates taken literally
VISIBLE_CASES = [
    chr(0x2022) + " one\n  " + chr(0x25E6) + " two\n3) three\n[x] done\n" + chr(0x2610) + " open",
    "Line with\ttabs  and   spaces\n\n\nNext para. Two sentences! Three?",
    "- a\n* b\n+ c\n12. d\n1234567890. ten digits are not a marker\n3.14 is not one either",
    chr(0x00B7) + "no space: kept\n" + chr(0x25CF) + " black circle item\n| a | b | stays as text",
]
VISIBLE_MD = ("# Quarterly review\n\nRevenue grew **12%**, led by the [new product](https://example.com/p).\n\n"
              "1. pricing\n2. *retention*, which rose to 91%\n\n| Region | Growth |\n|---|---|\n| EU | 9% |")
VISIBLE_COPY = ("Quarterly review\nRevenue grew 12%, led by the new product.\n1.\tpricing\n2.\tretention, which rose "
                "to 91%\nRegion\tGrowth\nEU\t9%")
CANON_V4_CASES = [
    "Word paste: \u201cdon\u2019t\u201d \u2014 ok\u2026",
    "typed: \"don't\" -- ok...",
    "Ukrainian: \u00abm\u02bciata\u00bb \u2013 m\u2019iata",
    "minus \u22125, figure dash 1\u20122, hyphens a---b",
    "v3 steps still apply\u00a0and\u200b BOM\ufeff",
]
SEGMENT_TEXT_V4 = "zh \u4eca\u5929\u3002\u597d\uff01\u5417\uff1f ar \u0643\u064a\u0641\u061f He said \"Yes.\" Then (left.) End"
# v5 (deletions before NFC): v4's cases, and the ones where v4 is not idempotent - a deleted character between a base
# and its combining mark, or between combining marks of different classes (found by fuzzing)
CANON_V5_CASES = CANON_V4_CASES + [
    "cafe\u200b\u0301 latte",
    "soft e\u00ad\u0301 hyphen",
    "thai \u0e01\u0e48\u0301\u2060\u0e48\u0301 marks",
    "bom between a\ufeff\u0308 and o\u0308",
    "crlf\r\n\u0301 at a line start",
    "manifest x\ufeff" + "\ufe0f" * 9 + "y",
    "pin: \u1ae8\u0e48 a mark assigned after Unicode 14.0 stays where it is",
    "pin: e\u1ae8\u0301 and o\u0301\u1ae8",
]


def build() -> dict:
    v: dict = {"canon_id": CANON_V3}
    k_bits, k_wm = crypto.bits_key(MASTER), crypto.wm_key(MASTER)
    k_log, k_ver = crypto.log_key(MASTER, DOC), crypto.verifier_key(MASTER, DOC, VERIFIER)
    v["keys"] = {"master": MASTER.hex(), "doc": DOC, "verifier": VERIFIER, "k_bits": k_bits.hex(),
                 "k_doc": crypto.doc_key(MASTER, DOC).hex(), "k_log": k_log.hex(), "k_ver": k_ver.hex(),
                 "k_wm": k_wm.hex()}
    v["bitstream"] = {"j": list(range(32)), "bits": [crypto.bit(k_bits, DOC, j) for j in range(32)],
                      "bits_low_share_0.75": [crypto.bit(k_bits, DOC, j, 0.75) for j in range(32)]}
    contexts = [[i, i + 1, i + 2, i + 3] for i in range(0, 64, 4)]
    v["content_selection"] = {"interval": 5, "contexts": contexts,
                              "decisions": [crypto.wm_decide(k_wm, c, 5) for c in contexts]}
    v["canonicalization"] = [{"input": s, "output": canonicalize(s, CANON_V3)} for s in CANON_CASES]
    spans = segment(SEGMENT_TEXT, CANON_V3)
    v["segmentation"] = {"text": SEGMENT_TEXT, "blocks": [[a, b, block_digest(SEGMENT_TEXT[a:b])] for a, b in spans]}
    # canonical form v5 (the reference): v4's steps with the deletions before NFC - idempotent
    v5_text = canonicalize(SEGMENT_TEXT_V4, CANON_V5)
    v["canonicalization_v5"] = {"canon_id": CANON_V5,
                                # v4 is not version-pinned: its output for a code point assigned after the
                                # library's Unicode version depends on the library, so those cases carry none
                                "cases": [{"input": c, "output": canonicalize(c, CANON_V5),
                                           "v4": None if c.startswith("pin:") else canonicalize(c, CANON_V4)}
                                          for c in CANON_V5_CASES],
                                "segmentation": {"text": SEGMENT_TEXT_V4, "canonical": v5_text,
                                                 "blocks": [[a, b, block_digest(v5_text[a:b])]
                                                            for a, b in segment(v5_text, CANON_V5)]}}
    # canonical form v4 (records sealed from 13.09.2026): typographic folding, blocks for scripts without spaces
    v4_text = canonicalize(SEGMENT_TEXT_V4, CANON_V4)
    v["canonicalization_v4"] = {"canon_id": CANON_V4,
                                "cases": [{"input": c, "output": canonicalize(c, CANON_V4)} for c in CANON_V4_CASES],
                                "segmentation": {"text": SEGMENT_TEXT_V4, "canonical": v4_text,
                                                 "blocks": [[a, b, block_digest(v4_text[a:b])]
                                                            for a, b in segment(v4_text, CANON_V4)]}}

    text = canonicalize("The council met on Tuesday. It voted twice.\nNothing else happened.", CANON_V3)
    header = Header(model="distilgpt2@2290a62682d06624634c1f46a6ad5be0f47f38aa", weights_digest="11" * 32,
                    tokenizer_digest="22" * 32, canon=CANON_V3, context="bos+text", k=1, interval=5, tol=0.0,
                    min_entries=8, indet_frac=0.2, text_digest=crypto.sha256_hex(text), engine="entropy-protocol/tv",
                    tol_space="logit", cal={"fp32": 0.00024, "fp16": 0.338}, components={"weights": "11" * 32},
                    env={"class": "fp32"}, selection="content", wm_context=4, salt=SALT.hex())
    record = Record(header, ((4, LOW), (12, HIGH), (31, LOW)),
                    tuple((a, b, block_digest(text[a:b])) for a, b in segment(text, CANON_V3)))
    plain = dumps(record)
    token = crypto.encrypt_token(k_log, DOC, plain, nonce=NONCE_LOG)
    token_v = crypto.encrypt_token(k_ver, DOC, plain, nonce=NONCE_VER)
    v["record"] = {"text": text, "cbor": plain.hex(), "token_log": token, "token_verifier": token_v,
                   "nonce_log": NONCE_LOG.hex(), "nonce_verifier": NONCE_VER.hex()}
    # a record in canonical form v4 with the confidence gate (wm_gate_pm on the wire; absent when there is no gate)
    raw4 = ("She said " + chr(0x201C) + "yes" + chr(0x201D) + " " + chr(0x2014) + " and left" + chr(0x2026) +
            " Then it rained.")
    text4 = canonicalize(raw4, CANON_V4)
    import dataclasses
    header4 = dataclasses.replace(header, canon=CANON_V4, text_digest=crypto.sha256_hex(text4), wm_gate=0.9)
    record4 = Record(header4, ((4, LOW), (19, HIGH)),
                     tuple((a, b, block_digest(text4[a:b])) for a, b in segment(text4, CANON_V4)))
    plain4 = dumps(record4)
    v["record_v4_gated"] = {"raw": raw4, "text": text4, "cbor": plain4.hex(), "wm_gate_pm": 900,
                            "token_log": crypto.encrypt_token(k_log, DOC, plain4, nonce=NONCE_LOG4),
                            "nonce_log": NONCE_LOG4.hex()}
    # visible-text layer: the verifier's normalization and blocks; a v4 record of a Markdown answer carrying the layer
    v["visible_text"] = {"form": PLAIN_V1, "cases": []}
    for c in VISIBLE_CASES:
        vis, spans = segment_visible(c, PLAIN_V1)
        v["visible_text"]["cases"].append({"input": c, "visible": vis,
                                           "blocks": [[a, b, block_digest(vis[a:b])] for a, b in spans]})
    v["visible_text_v2"] = {"form": PLAIN_V2, "cases": []}
    for c in VISIBLE_CASES + ["- cafe\u200b\u0301 item\n- e\u00ad\u0301 two"]:
        vis, spans = segment_visible(c, PLAIN_V2)
        v["visible_text_v2"]["cases"].append({"input": c, "visible": vis,
                                              "blocks": [[a, b, block_digest(vis[a:b])] for a, b in spans]})
    text5 = canonicalize(VISIBLE_MD, CANON_V4)
    vis5, spans5 = segment_source(text5, PLAIN_V1)
    header5 = dataclasses.replace(header, canon=CANON_V4, text_digest=crypto.sha256_hex(text5), plain=PLAIN_V1,
                                  plain_digest=crypto.sha256_hex(vis5))
    record5 = Record(header5, ((4, LOW),), tuple((a, b, block_digest(text5[a:b])) for a, b in segment(text5, CANON_V4)),
                     tuple((a, b, block_digest(vis5[a:b])) for a, b in spans5))
    plain5 = dumps(record5)
    edited5 = VISIBLE_COPY.replace("91%", "97%")
    v["record_v4_visible"] = {"markdown": VISIBLE_MD, "text": text5, "visible": vis5, "cbor": plain5.hex(),
                              "token_log": crypto.encrypt_token(k_log, DOC, plain5, nonce=NONCE_LOG5),
                              "nonce_log": NONCE_LOG5.hex(), "copy": VISIBLE_COPY, "edited": edited5,
                              "edited_visible": visible_text(edited5, PLAIN_V1)}

    signer = Ed25519PrivateKey.from_private_bytes(SIGNER_SEED)
    gen, commit = statements.generation_statement(k_log, token, signer, "generator.example", iat=IAT)
    v["generation_statement"] = {"signer_seed": SIGNER_SEED.hex(), "issuer": "generator.example", "iat": IAT,
                                 "commitment_cbor": cose.dumps(commit).hex(),
                                 "commitment_sha256": statements.sha256(cose.dumps(commit)).hex(),
                                 "cose_sign1": gen.hex(), "ref": statements.statement_ref(gen).hex()}

    verifier = Ed25519PrivateKey.from_private_bytes(VERIFIER_SEED)
    body = {"v": 1, "verdict": "MODIFIED", "doc": DOC, "cand": statements.sha256(b"candidate"),
            "gen": statements.statement_ref(gen), "counts": [2, 1, 0, 3],
            "ranges": {"changed": [[28, 44]], "deleted": [], "divergent": [[31, 38]]},
            "wts": bytes.fromhex("11" * 32), "tok": bytes.fromhex("22" * 32),
            "dec": {"class": "fp32", "dtype": "float32", "space": "logit", "tol_u": 240, "canon": CANON_V3,
                    "calibrated": True}, "nonce": "00" * 8, "engine": "entropy-protocol/tv"}
    protected = {cose.ALG: cose.ED25519, cose.KID: cose.kid_of(verifier.public_key()),
                 cose.CWT_CLAIMS: {cose.ISS: VERIFIER, cose.SUB: DOC, cose.IAT: IAT},
                 cose.TYP: statements.TYP_VERIFICATION}
    ver = cose.sign1(protected, cose.dumps(body), verifier)
    v["verification_statement"] = {"signer_seed": VERIFIER_SEED.hex(), "payload_cbor": cose.dumps(body).hex(),
                                   "cose_sign1": ver.hex()}

    log = tlog.TransparencyLog(Ed25519PrivateKey.from_private_bytes(LOG_SEED))
    entries = [cose.sign1({cose.ALG: cose.ED25519}, f"entry-{i}".encode(), signer) for i in range(6)] + [gen]
    for e in entries:
        log.register(e)
    v["merkle"] = {"log_seed": LOG_SEED.hex(), "entries": [e.hex() for e in entries],
                   "leaf_hashes": [tlog.leaf_hash(tlog.entry_of(e)).hex() for e in entries],
                   "roots": [log.root(n).hex() for n in range(1, len(entries) + 1)],
                   "inclusion_size7_index3": [p.hex() for p in tlog.inclusion_path(3, log._leaves)],
                   "receipt_index6": log.receipt(6).hex()}
    # record delivered to a verifier by public key: COSE-HPKE integrated encryption (draft-ietf-cose-hpke-27)
    sk_r, pk_r = hpke.derive_keypair(HPKE_RECIPIENT_IKM)
    sk_e, _ = hpke.derive_keypair(HPKE_EPHEMERAL_IKM)
    v["hpke_rfc9180_a21"] = RFC9180_A21
    v["delivered_record"] = {"recipient_ikm": HPKE_RECIPIENT_IKM.decode(), "recipient_sk": sk_r.hex(),
                             "recipient_pk": pk_r.hex(), "recipient_cose_key": cose.x25519_cose_key(pk_r).hex(),
                             "ephemeral_ikm": HPKE_EPHEMERAL_IKM.decode(), "alg": cose.HPKE_4,
                             "content_type": delivery.CT_RECORD,
                             "cose_encrypt0": delivery.deliver(k_log, token, pk_r, sk_e=sk_e).hex()}

    # small n; then past 1023 checkpoints (a double cannot hold 2^n there) and into the subnormal range
    cases = [(10, 10), (20, 15), (30, 26), (40, 34), (18, 17), (1023, 600), (1024, 600), (1100, 700), (2000, 1100),
             (5000, 2600), (1050, 1049), (1060, 1060), (1074, 1074), (1075, 1075), (1076, 1075), (3000, 2900)]
    v["binomial_tail"] = [{"n": n, "m": m, "p": binomial_tail(n, m),
                           "exact": str(Fraction(sum(__import__("math").comb(n, i) for i in range(m, n + 1)), 1 << n))}
                          for n, m in cases]
    return v


def main():
    out = ROOT / "spec" / "test-vectors" / "ep1-vectors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
