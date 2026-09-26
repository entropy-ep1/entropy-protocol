"""End-to-end on a real model (distilgpt2 from the local Hugging Face cache)."""
import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.model

from entropy_protocol import attest, crypto  # noqa: E402
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from entropy_protocol.verify import MODIFIED, NOT_VERIFIABLE, VERIFIED, verify  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    from entropy_protocol.backend_hf import HFBackend
    try:
        return HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")


MASTER = bytes(range(32))       # fixed key, document id and seed -> the same text on every run of one machine


@pytest.fixture(scope="module")
def sealed(backend):
    s = seal(backend, "The history of the printing press shows", MASTER, SealParams(max_new_tokens=160),
             doc_id="doc-fixture-1", seed=7)
    assert s.entries >= 8, s
    return MASTER, s


def test_unchanged_and_transported_text_verifies(backend, sealed):
    master, s = sealed
    key = crypto.log_key(master, s.doc_id)
    assert verify(backend, s.text, s.token, key).verdict == VERIFIED
    moved = "\n" + s.text.replace("\n", "  \r\n").replace(" ", "\u00a0", 1) + "\n\n"
    r = verify(backend, moved, s.token, key)
    assert r.verdict == VERIFIED and r.digest_match


def test_word_edit_is_modified_and_localized(backend, sealed):
    import re
    from entropy_protocol.canon import canonicalize_with_map
    master, s = sealed
    words = list(re.finditer(r"[A-Za-z]{3,}", s.text))
    w = words[len(words) // 2]
    edited = s.text[:w.start()] + "banana" + s.text[w.end():]
    cpos = canonicalize_with_map(s.text)[1][w.start()]
    r = verify(backend, edited, s.token, crypto.log_key(master, s.doc_id))
    assert r.verdict == MODIFIED
    assert any(a <= cpos < b for a, b in r.changed_ranges + r.divergence_ranges), (cpos, r.changed_ranges)
    assert all(a >= cpos - 16 for a, _ in r.divergence_ranges)   # the model layer never diverges before the edit


def test_single_character_edit_is_modified_and_block_localized(backend, sealed):
    master, s = sealed
    mid = len(s.text) // 2
    while not s.text[mid].isalpha():
        mid += 1
    edited = s.text[:mid] + ("x" if s.text[mid] != "x" else "z") + s.text[mid + 1:]
    from entropy_protocol.canon import canonicalize_with_map
    cpos = canonicalize_with_map(s.text)[1][mid]
    r = verify(backend, edited, s.token, crypto.log_key(master, s.doc_id))
    assert r.verdict == MODIFIED
    assert any(a <= cpos < b for a, b in r.changed_ranges), (cpos, r.changed_ranges)


def test_deleted_sentence_reports_deletion_point(backend, sealed):
    import re
    master, s = sealed
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s)", s.text)]
    if len(ends) < 3:
        pytest.skip("generated text has fewer than three sentences")
    a, b = ends[0] + 1, ends[1] + 1
    r = verify(backend, s.text[:a] + s.text[b:], s.token, crypto.log_key(master, s.doc_id))
    assert r.verdict == MODIFIED and (r.deletion_points or r.changed_ranges)


def test_wrong_key_and_verifier_scoped_key(backend, sealed):
    master, s = sealed
    assert verify(backend, s.text, s.token, crypto.log_key(crypto.new_master_key(), s.doc_id)).verdict == NOT_VERIFIABLE
    tok_v, k_v = crypto.reissue_for_verifier(master, s.token, "publisher-1")
    assert verify(backend, s.text, tok_v, k_v).verdict == VERIFIED


def test_calibration_table_in_logit_space(backend):
    from entropy_protocol.predicates import stat_margin
    from entropy_protocol.record import loads
    params = SealParams(max_new_tokens=160, tol_space="logit", cal={"fp32": 0.05, "fp16": 0.3})
    s = seal(backend, "The history of the printing press shows", MASTER, params, seed=11)
    key = crypto.log_key(MASTER, s.doc_id)
    rec = loads(crypto.decrypt_token(key, s.token))
    assert rec.header.tol_space == "logit" and rec.header.cal == {"fp32": 0.05, "fp16": 0.3}
    from entropy_protocol.canon import canonicalize
    stats = backend.position_stats(canonicalize(s.text), [c for c, _ in rec.entries], rec.header.k)
    assert all(stat_margin(p, st, rec.header.k, "logit") >= 0.6 for (_, p), st in zip(rec.entries, stats))
    r = verify(backend, s.text, s.token, key)
    assert r.verdict == VERIFIED and r.calibrated is True
    assert r.decoding["class"] == "fp32" and r.decoding["tol"] == 0.05 and r.decoding["tol_space"] == "logit"
    # a table that does not list this class: evaluated with the largest listed tolerance, flagged
    s2 = seal(backend, "The history of the printing press shows", MASTER,
              SealParams(max_new_tokens=160, tol_space="logit", cal={"bf16": 0.3}), seed=11)
    r2 = verify(backend, s2.text, s2.token, crypto.log_key(MASTER, s2.doc_id))
    assert r2.calibrated is False and r2.decoding["tol"] == 0.3 and r2.verdict == VERIFIED


def test_calibration_measures_precision_noise(backend):
    import torch
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.calibrate import calibrate
    corpus = [backend.canonical_ids(t) for t in ("The council met on Tuesday to discuss the budget.",
                                                  "Water boils at one hundred degrees at sea level.")]
    cal = calibrate(backend, {"cpu:float32": backend,
                              "cpu:bfloat16": lambda: HFBackend("distilgpt2", dtype=torch.bfloat16),
                              "nowhere:float16": lambda: HFBackend("distilgpt2", device="nowhere")},
                    corpus, k=1, space="logit", c=3.0)
    assert cal.delta["cpu:float32"] == 0.0 and cal.delta["cpu:bfloat16"] > 0.0
    assert "nowhere:float16" in cal.errors
    t = cal.table(["fp32", "bf16"])
    assert t["fp32"] == 0.0 and t["bf16"] == round(3.0 * cal.delta["cpu:bfloat16"], 6)
    with pytest.raises(ValueError):
        cal.table(["fp16"])


def test_margin_aware_high_selection(backend):
    import torch
    from entropy_protocol.record import HIGH
    prompt = "The history of the printing press shows"
    ids, enforced = backend.generate(prompt, lambda j, gen: HIGH if j and j % 5 == 0 else None, 80, 0.8, 0.9, 3, 5,
                                     seed=3, high_margin=1.0)
    p_ids = backend.tok(prompt).input_ids
    with torch.no_grad():
        logits = backend.model(torch.tensor([p_ids + ids])).logits[0].float()
    for j in enforced:                      # under the generation context, each HIGH token sits >= 1.0 below the k-th
        row = logits[len(p_ids) + j - 1]
        assert float(torch.sort(row, descending=True).values[2] - row[ids[j]]) >= 1.0 - 1e-4


def test_high_positions_never_emit_special_tokens():
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.record import HIGH
    try:
        qwen = HFBackend("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen2.5-0.5B-Instruct not available offline: {exc}")
    assert {151643, 151645} <= qwen.stop_ids                   # <|endoftext|> and <|im_end|> both end a text
    for seed in range(3):
        ids, enforced = qwen.generate("def parse(line):", lambda j, gen: HIGH, 40, 0.8, 0.9, 10, 40, seed=seed)
        assert ids and not (set(ids) & qwen.stop_ids)
        assert "<|" not in qwen.tok.decode(ids)


def test_watermark_detected_without_the_record(backend):
    from entropy_protocol.detect import detect
    s = seal(backend, "The history of the printing press shows", MASTER,
             SealParams(max_new_tokens=200, min_new_tokens=160), doc_id="doc-wm-1", seed=5)
    kw = crypto.wm_key(MASTER)
    hit = detect(backend, s.text, kw)
    assert hit.detected and hit.p_value < 1e-6 and hit.checkpoints >= 20, hit
    moved = "\ufeff" + s.text.replace("\n", "\r\n") + "\n"          # transport does not matter
    assert detect(backend, moved, kw).p_value == hit.p_value
    other = detect(backend, s.text, crypto.wm_key(crypto.new_master_key()))
    assert not other.detected and other.p_value > 1e-3, other            # another key: no signal
    human = ("I walked to the market this morning because the fridge was empty again. The man who sells apples "
             "asked about my sister, and we talked for a while about the rain. Some weeks are like that: you "
             "start many things and finish few, and the list on the kitchen door keeps growing longer.")
    assert not detect(backend, human, kw).detected
    # a sealed passage inside a longer human text: the localized search finds it and keeps the p-value honest
    mixed = human + "\n\n" + s.text + "\n\n" + human
    loc = detect(backend, mixed, kw, localize=True, alpha=1e-3)
    from entropy_protocol.canon import canonicalize
    start = canonicalize(mixed).index(canonicalize(s.text)[:40])
    assert loc.segment is not None and loc.segment_detected and loc.segment_p_corrected < 1e-3, loc
    assert start - 40 <= loc.segment[0] and loc.segment[1] <= start + len(canonicalize(s.text)) + 40, (loc.segment, start)
    assert not detect(backend, human + " " + human, kw, localize=True, alpha=1e-3).segment_detected



def test_pasted_passage_is_read_from_its_paragraph(backend):
    """A sealed passage pasted between human paragraphs: read after them, its low-margin checkpoints see another
    context; read from its own paragraph start (start token + passage, as it was sealed), every enforced token is
    back. The best window over all readings is corrected by all windows examined, so human text stays clean."""
    from entropy_protocol.detect import detect
    s = seal(backend, "The history of the printing press shows", MASTER,
             SealParams(max_new_tokens=200, min_new_tokens=160), doc_id="doc-wm-1", seed=5)
    kw = crypto.wm_key(MASTER)
    human = ("I walked to the market this morning because the fridge was empty again. The man who sells apples "
             "asked about my sister, and we talked for a while about the rain. Some weeks are like that: you "
             "start many things and finish few, and the list on the kitchen door keeps growing longer.")
    mixed = human + "\n\n" + s.text + "\n\n" + human
    plain = detect(backend, mixed, kw, localize=True, alpha=1e-3, restarts=0)
    read = detect(backend, mixed, kw, localize=True, alpha=1e-3)
    assert read.restarts >= 2 and read.segment_windows > plain.segment_windows   # the passage has paragraphs too
    assert read.segment_p <= plain.segment_p                       # the readings add windows, never lose one
    assert read.segment_detected and read.segment_p_corrected < 1e-6, (read, plain)
    from entropy_protocol.canon import canonicalize
    start = canonicalize(mixed).index(canonicalize(s.text)[:40])        # the segment is in canonical offsets
    assert start - 40 <= read.segment[0] and read.segment[1] <= start + len(canonicalize(s.text)) + 40
    clean = detect(backend, "\n\n".join([human] * 3), kw, localize=True, alpha=1e-3)
    assert clean.restarts == 2 and not clean.segment_detected
    from entropy_protocol.batch import detect_batch                 # the batched detector reads the same way
    both = detect_batch(backend, [mixed, "\n\n".join([human] * 3)], kw, localize=True, alpha=1e-3)
    for b_, s_ in zip(both, (read, clean)):
        assert (b_.restarts, b_.segment_windows, b_.segment, b_.segment_detected) == \
               (s_.restarts, s_.segment_windows, s_.segment, s_.segment_detected)
        assert abs(b_.segment_p - s_.segment_p) <= 1e-12 + 1e-9 * s_.segment_p

def test_signed_evidence_pack(backend, sealed):
    master, s = sealed
    r = verify(backend, s.text, s.token, crypto.log_key(master, s.doc_id))
    sk = attest.new_signing_key()
    signed = attest.sign(attest.evidence_pack(r, "verifier-1", s.token), sk)
    assert attest.verify_signed(signed, sk.public_key())
    signed["pack"]["verdict"] = "MODIFIED"
    assert not attest.verify_signed(signed, sk.public_key())
    assert s.text[:30] not in str(signed)
