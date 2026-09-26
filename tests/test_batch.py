"""Batched sealing / verification / detection agree with the single-text path (distilgpt2, CPU, fp32)."""
import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.model

from entropy_protocol import crypto  # noqa: E402
from entropy_protocol.batch import (detect_batch, generate_batch, score_batch, seal_batch, stats_at,  # noqa: E402
                                    verify_batch)
from entropy_protocol.detect import detect  # noqa: E402
from entropy_protocol.record import LOW  # noqa: E402
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from entropy_protocol.verify import MODIFIED, VERIFIED, verify  # noqa: E402

MASTER = bytes(range(32))
PROMPTS = ["The history of the printing press shows", "Our quarterly report highlights three trends:",
           "It was the coldest winter anyone could remember,", "Doctors recommend that"]


def always_low(j, gen):
    return LOW


@pytest.fixture(scope="module")
def backend():
    from entropy_protocol.backend_hf import HFBackend
    try:
        return HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")


def test_greedy_batch_generation_equals_single(backend):
    """Every position LOW (the first choice), greedy elsewhere: batched (left-padded, two contexts) and single
    generation produce the same tokens, in both enforcement contexts."""
    for align in ("prompt", "free"):
        batch = generate_batch(backend, PROMPTS, [always_low] * len(PROMPTS), 40, 0.0, 0.9, 1, 3, seed=1,
                               align=align)
        for p, (ids, enf) in zip(PROMPTS, batch):
            single, enf1 = backend.generate(p, always_low, 40, 0.0, 0.9, 1, 3, seed=1, align=align)
            assert ids == single, (align, p)
            assert enf == enf1


def test_rows_that_end_leave_the_batch(backend):
    """Texts end at different steps (a period made an end-of-text token for the test): the rows still in the
    batch after compaction continue exactly as they would alone."""
    def every_third(j, gen):          # enforced LOW at every third position, greedy (prompt context) elsewhere
        return LOW if j % 3 == 2 else None

    period = backend.tok(".", add_special_tokens=False)["input_ids"][0]
    saved = backend.stop_ids
    backend.stop_ids = frozenset(saved | {period})
    try:
        for align in ("prompt", "free"):
            mixed = generate_batch(backend, PROMPTS, [every_third] * 4, 60, 0.0, 0.9, 1, 3, seed=1, align=align)
            alone = [backend.generate(p, every_third, 60, 0.0, 0.9, 1, 3, seed=1, align=align) for p in PROMPTS]
            assert mixed == alone, align
            assert len({len(ids) for ids, _ in alone}) > 1, align    # the rows really ended at different steps
    finally:
        backend.stop_ids = saved


def test_scores_match_single_pass(backend):
    texts = ["A short text.", "A somewhat longer text, with more words in it, and a clause at the end."]
    encs = [backend.tok(t, return_offsets_mapping=True, add_special_tokens=False) for t in texts]
    scs = score_batch(backend, [e["input_ids"] for e in encs], k=1)
    for t, e, sc in zip(texts, encs, scs):
        starts = [a for a, _ in e["offset_mapping"]]
        for s1, s2 in zip(backend.position_stats(t, starts, 1), stats_at(e["offset_mapping"], starts, sc)):
            assert (s1.token_index, s1.rank0, s1.char_range) == (s2.token_index, s2.rank0, s2.char_range)
            assert abs(s1.l_tok - s2.l_tok) < 1e-3
            assert all(abs(a - b) < 1e-3 for a, b in zip(s1.top_logits, s2.top_logits))
            assert abs(s1.p_tok - s2.p_tok) < 1e-5


def test_seal_batch_verifies_and_detects(backend):
    params = SealParams(max_new_tokens=120, min_new_tokens=100)
    sealed = seal_batch(backend, PROMPTS, MASTER, params, doc_ids=[f"doc-batch-{i}" for i in range(4)], seed=3)
    keys = [crypto.log_key(MASTER, s.doc_id) for s in sealed]
    assert all(s.entries >= 8 for s in sealed), [s.entries for s in sealed]
    # the single verifier accepts batched seals; the batched verifier gives the same verdicts
    single = [verify(backend, s.text, s.token, k) for s, k in zip(sealed, keys)]
    assert all(r.verdict == VERIFIED for r in single), [(r.verdict, r.reason) for r in single]
    batched = verify_batch(backend, [s.text for s in sealed], [s.token for s in sealed], keys)
    assert [r.verdict for r in batched] == [VERIFIED] * 4
    assert [r.matched for r in batched] == [r.matched for r in single]
    assert batched[0].decoding["env"]["batch"] == 4
    edited = [s.text[: len(s.text) // 2] + " banana" + s.text[len(s.text) // 2:] for s in sealed]
    assert [r.verdict for r in verify_batch(backend, edited, [s.token for s in sealed], keys)] == [MODIFIED] * 4
    # record-free detection: batched = single
    k_wm = crypto.wm_key(MASTER)
    dets = detect_batch(backend, [s.text for s in sealed], k_wm, alpha=1e-3)
    for s, d in zip(sealed, dets):
        d1 = detect(backend, s.text, k_wm, alpha=1e-3)
        assert (d.matched, d.checkpoints, d.p_value) == (d1.matched, d1.checkpoints, d1.p_value)
        assert d.p_value < 1e-2                   # 100-120 tokens of distilgpt2: 13-24 checkpoints


def test_one_batch_many_issuers(backend):
    """Documents of different keys in one generation batch: each record opens only with its own key, and each
    text is detected under its own detection key only."""
    other = bytes(range(32, 64))
    params = SealParams(max_new_tokens=120, min_new_tokens=120)
    sealed = seal_batch(backend, PROMPTS[:2], [MASTER, other], params, doc_ids=["doc-a", "doc-b"], seed=9)
    assert verify(backend, sealed[0].text, sealed[0].token, crypto.log_key(MASTER, "doc-a")).verdict == VERIFIED
    assert verify(backend, sealed[1].text, sealed[1].token, crypto.log_key(other, "doc-b")).verdict == VERIFIED
    assert verify(backend, sealed[1].text, sealed[1].token, crypto.log_key(MASTER, "doc-b")).verdict == "NOT_VERIFIABLE"
    assert detect(backend, sealed[0].text, crypto.wm_key(MASTER), alpha=1e-2).p_value < 1e-2
    # (with this key distilgpt2 falls into a blank-line loop on the second prompt, leaving few checkpoints — the
    # key-dependent degeneration of a small base model; the per-checkpoint match rate still separates the keys)
    for s, own, foreign in ((sealed[0], MASTER, other), (sealed[1], other, MASTER)):
        d_own = detect(backend, s.text, crypto.wm_key(own))
        d_foreign = detect(backend, s.text, crypto.wm_key(foreign))
        assert d_own.checkpoints and d_own.matched / d_own.checkpoints >= 0.75
        assert d_own.matched / d_own.checkpoints > d_foreign.matched / max(1, d_foreign.checkpoints)


def test_batched_and_single_seals_are_equally_strong(backend):
    """Same prompts, batched and single: not the same texts (sampling differs), the same kind of record. Counted
    over the prompts, not per text: distilgpt2 can fall into a loop whose repeated contexts carry few checkpoints
    (seed 5 does this to one single-path text with canonical generation: 3 checkpoints in 120 tokens)."""
    params = SealParams(max_new_tokens=120, min_new_tokens=120)
    b = seal_batch(backend, PROMPTS, MASTER, params, seed=5)
    s = [seal(backend, p, MASTER, params, seed=5) for p in PROMPTS]
    assert all(x.tokens == y.tokens == 120 for x, y in zip(b, s))
    eb, es = sum(x.enforced for x in b), sum(y.enforced for y in s)
    assert eb > 40 and es > 40 and 0.5 < eb / es < 2, (eb, es)
    assert sum(x.entries for x in b) >= 0.8 * eb and sum(y.entries for y in s) >= 0.8 * es


def test_uneven_prompts_in_half_precision_on_mps():
    """Left padding once turned into NaN on MPS in half precision (the padding positions attend to nothing, and
    0 x NaN reached the real positions): the prompt pass now runs row by row, so a batch of chat prompts of different
    lengths is finite and equals the single path."""
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    from entropy_protocol.backend_hf import HFBackend
    try:
        qwen = HFBackend("Qwen/Qwen2.5-0.5B-Instruct", device="mps", dtype=torch.float16)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen2.5-0.5B-Instruct not available offline: {exc}")
    qs = ["What is 2+2?", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in "
          "May. How many clips did Natalia sell altogether in April and May?"]
    enc = [list(qwen.tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                             tokenize=True)) for q in qs]

    def free(j, gen):
        return None

    out = generate_batch(qwen, enc, [free] * 2, 24, 0.0, 0.9, 1, 3, seed=1, align="free", canon_gen=True)
    for (ids, _), e in zip(out, enc):
        assert ids and ids == qwen.generate(qwen.tok.decode(e), free, 24, 0.0, 0.9, 1, 3, seed=1, align="free",
                                            canon_gen=True)[0]


def test_sampling_draws_where_the_generator_lives():
    """A generator made for "cuda" reports device cuda while the tensors report cuda:0: the draw stays on the GPU
    (15.09.2026: comparing whole devices sent it to the CPU with a CUDA generator, and every sampled generation failed
    on the first GPU run). A CPU generator with accelerator weights draws on the CPU and returns to their device."""
    import torch
    from entropy_protocol.batch import _generator, _multinomial
    p = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])          # one choice per row: the draw is known
    assert _multinomial(p, _generator("cpu", 1)).tolist() == [1, 2]
    devs = ["cuda"] * torch.cuda.is_available() + ["mps"] * torch.backends.mps.is_available()
    for dev in devs:
        q = p.to(dev)
        for g in (_generator(dev, 1), torch.Generator().manual_seed(1)):
            out = _multinomial(q, g)
            assert out.device == q.device and out.tolist() == [1, 2]


def test_mps_cache_goes_back_without_changing_results(monkeypatch):
    """On MPS the allocator's cache of freed blocks goes back to the system after forward passes (HFBackend.tidy;
    without it a 1.5B model filled a 16 GB Mac): the call happens, and the tokens equal those with the cache kept."""
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    from entropy_protocol.backend_hf import HFBackend
    try:
        b = HFBackend("distilgpt2", device="mps")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    calls, empty = [], torch.mps.empty_cache
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: (calls.append(1), empty())[1])
    b.MPS_SLACK = float("inf")                                  # the cache kept
    kept = generate_batch(b, PROMPTS, [always_low] * len(PROMPTS), 40, 0.0, 0.9, 1, 3, seed=1, align="free")
    assert not calls
    b.MPS_SLACK = -1                                            # the cache returned at every check
    tidied = generate_batch(b, PROMPTS, [always_low] * len(PROMPTS), 40, 0.0, 0.9, 1, 3, seed=1, align="free")
    assert calls and tidied == kept
