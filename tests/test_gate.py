"""Confidence gate: checkpoints only where the verifier's view is not sure of the next token."""
import pytest

from entropy_protocol.record import Header, Record, dumps, loads


def _header(**kw):
    base = dict(model="m", weights_digest="11" * 32, tokenizer_digest="22" * 32, canon="c", context="bos+text", k=1,
                interval=5, tol=0.0, min_entries=8, indet_frac=0.2, text_digest="33" * 32, engine="e")
    return Header(**{**base, **kw})


def test_gate_is_optional_on_the_wire():
    plain = dumps(Record(_header(), ((1, 0),)))
    assert b"wm_gate" not in plain                                  # records without a gate keep their bytes
    gated = dumps(Record(_header(wm_gate=0.9), ((1, 0),)))
    assert b"wm_gate_pm" in gated and loads(gated).header.wm_gate == 0.9
    assert loads(plain).header.wm_gate == 0.0


pytest.importorskip("torch")


@pytest.fixture(scope="module")
def backend():
    from entropy_protocol.backend_hf import HFBackend
    try:
        return HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")


MASTER = bytes(range(32))
PROMPTS = ["The history of the printing press shows", "Our quarterly report highlights three trends:"]


@pytest.mark.model
def test_no_checkpoint_where_the_verifier_is_confident(backend):
    from entropy_protocol import crypto
    from entropy_protocol.canon import canonicalize
    from entropy_protocol.detect import detect
    from entropy_protocol.seal import SealParams, seal
    from entropy_protocol.verify import VERIFIED, verify
    gate = 0.5
    s = seal(backend, PROMPTS[0], MASTER, SealParams(max_new_tokens=160, min_new_tokens=160, gate=gate),
             doc_id="doc-gate", seed=3)
    rec = loads(crypto.decrypt_token(crypto.log_key(MASTER, s.doc_id), s.token))
    assert rec.header.wm_gate == gate and rec.entries
    stats = backend.position_stats(canonicalize(s.text), [c for c, _ in rec.entries], 1)
    assert all(st.top_sorted[0] < gate + 1e-4 for st in stats if st is not None)
    assert verify(backend, s.text, s.token, crypto.log_key(MASTER, s.doc_id)).verdict == VERIFIED
    d = detect(backend, s.text, crypto.wm_key(MASTER), gate=gate)
    assert d.gated > 0 and d.matched / max(1, d.checkpoints) > 0.8


@pytest.mark.model
def test_gate_batch_equals_single(backend):
    from entropy_protocol.batch import generate_batch
    from entropy_protocol.record import HIGH, LOW

    def alternate(j, gen):
        return (LOW, None, HIGH)[j % 3]

    batch = generate_batch(backend, PROMPTS, [alternate] * 2, 40, 0.0, 0.9, 1, 3, seed=1, align="free", gate=0.6)
    for p, (ids, enf) in zip(PROMPTS, batch):
        single = backend.generate(p, alternate, 40, 0.0, 0.9, 1, 3, seed=1, align="free", gate=0.6)
        assert (ids, enf) == single
    ungated = backend.generate(PROMPTS[0], alternate, 40, 0.0, 0.9, 1, 3, seed=1, align="free")
    assert len(batch[0][1]) < len(ungated[1])                        # the gate removed some checkpoints


def test_gate_needs_the_verifier_context():
    from entropy_protocol.seal import SealParams, check_params
    with pytest.raises(ValueError):
        check_params(SealParams(gate=0.9, align="prompt"))


@pytest.mark.model
def test_guard_batch_equals_single_and_removes_confident_checkpoints(backend):
    from entropy_protocol.batch import generate_batch
    from entropy_protocol.record import HIGH, LOW

    def alternate(j, gen):
        return (LOW, None, HIGH)[j % 3]

    batch = generate_batch(backend, PROMPTS, [alternate] * 2, 40, 0.0, 0.9, 1, 3, seed=1, align="free", guard=0.5)
    for p, (ids, enf) in zip(PROMPTS, batch):
        assert (ids, enf) == backend.generate(p, alternate, 40, 0.0, 0.9, 1, 3, seed=1, align="free", guard=0.5)
    assert len(batch[0][1]) < len(backend.generate(PROMPTS[0], alternate, 40, 0.0, 0.9, 1, 3, seed=1, align="free")[1])
