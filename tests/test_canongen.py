"""Canonical generation: sealed text is canonical (form v4) as generated."""
import pytest

pytest.importorskip("torch")

from entropy_protocol.canon import canonicalize  # noqa: E402
from entropy_protocol.canongen import FORBIDDEN, CanonGuard, _tail  # noqa: E402


def test_utf8_tail():
    q = chr(0x2019).encode("utf-8")              # right single quotation mark: E2 80 99
    assert _tail(b"ab") == b"" and _tail(b"a" + q) == b"" and _tail(b"a" + q[:2]) == q[:2] and _tail(q[:1]) == q[:1]


@pytest.fixture(scope="module", params=["distilgpt2", "Qwen/Qwen2.5-0.5B-Instruct"])
def tok(request):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(request.param, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{request.param} not available offline: {exc}")


def test_guard_bans_exactly_what_the_canonical_form_would_change(tok):
    g = CanonGuard(tok, len(tok))
    assert g.banned_any > 0
    for text in ("don" + chr(0x2019) + "t", "a" + chr(0x2014) + "b", "wait" + chr(0x2026), "x" + chr(0x00A0) + "y"):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        state, hit = 0, False
        for i in ids:
            hit |= bool(g.mask(state)[i])
            state = g.state_after(state, i)
        assert hit, text                         # some token of the text is banned in its state
    for text in ("don't", "a--b", "wait...", "plain text, 42.", "caf" + chr(0xE9), "emoji " + chr(0x1F600)):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        state = 0
        for i in ids:
            assert not bool(g.mask(state)[i]), (text, i)
            state = g.state_after(state, i)


MASTER = bytes(range(32))


@pytest.mark.model
def test_canonical_generation_produces_canonical_text():
    from entropy_protocol import crypto
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.batch import generate_batch
    from entropy_protocol.detect import detect
    from entropy_protocol.record import HIGH, LOW
    from entropy_protocol.seal import SealParams, seal
    try:
        b = HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    prompts = ["He said \u201cwe don\u2019t know\u201d \u2014 and then", "The minister\u2019s statement \u2026"]
    for i, p in enumerate(prompts):
        s = seal(b, p, MASTER, SealParams(max_new_tokens=120, min_new_tokens=120, canon_gen=True),
                 doc_id=f"cg{i}", seed=5)
        raw = s.text.encode("utf-8")
        assert not any(f in raw for f in FORBIDDEN), s.text
        assert canonicalize(s.text) == s.text.strip()
        d = detect(b, s.text, crypto.wm_key(MASTER))
        assert d.checkpoints and d.matched / d.checkpoints >= 0.75

    def alternate(j, gen):
        return (LOW, None, HIGH)[j % 3]

    batch = generate_batch(b, prompts, [alternate] * 2, 40, 0.0, 0.9, 1, 3, seed=1, align="free", canon_gen=True)
    for p, out in zip(prompts, batch):
        assert out == b.generate(p, alternate, 40, 0.0, 0.9, 1, 3, seed=1, align="free", canon_gen=True)



@pytest.fixture(scope="module")
def gpt2():
    from entropy_protocol.backend_hf import HFBackend
    try:
        return HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")


@pytest.mark.model
def test_first_token_is_read_as_the_verifier_reads_it(gpt2):
    """A completion starts with " that"; the canonical form strips the space. The verifier's view (its cache and the
    checkpoint contexts) takes "that": the generation-time view is exactly the verifier's pass over the text."""
    import torch
    from entropy_protocol.record import LOW
    b, g = gpt2, gpt2.canon_guard
    (that,), (sp_that,) = b.tok("that")["input_ids"], b.tok(" that")["input_ids"]
    assert int(g.lead[sp_that]) == that and int(g.lead[that]) == that
    assert all(int(g.lead[i]) == -1 for i in b.tok("\n\n")["input_ids"] + b.tok(" ")["input_ids"])

    def low_from_4(j, view):
        return LOW if j >= 4 else None

    ids, enforced = b.generate("The history of the printing press shows", low_from_4, 40, 0.0, 0.9, 1, 3, seed=1,
                               align="free", canon_gen=True)
    assert b.tok.decode(ids)[0].isspace() and len(enforced) > 25
    view = [int(g.lead[ids[0]])] + ids[1:]
    assert b.tok(canonicalize(b.tok.decode(ids)), add_special_tokens=False)["input_ids"][0] == view[0]
    with torch.no_grad():
        full = b.model(torch.tensor([[b.start_id] + view])).logits[0]
    assert all(int(full[j].argmax()) == view[j] for j in enforced)     # every LOW holds in the verifier's view


def test_no_stray_neighbours_of_a_banned_character(tok):
    """A byte-level tokenizer may split a curly quote as E2 80 + 9D. The guard never lets a token end on E2 80, and
    inside a character it allows only continuation bytes, so the text stays UTF-8 and never gets a dagger or an angle
    quote where the model meant a quotation mark."""
    g = CanonGuard(tok, len(tok))
    for i, tb in enumerate(g.bytes):
        if tb and _tail(tb) == bytes([0xE2, 0x80]):
            assert bool(g.mask(0)[i]), tok.convert_ids_to_tokens(i)
    lead_e2 = g.index.get(bytes([0xE2]))
    if lead_e2 is not None:
        for i, tb in enumerate(g.bytes):
            if tb and tb[0] < 0x80:
                assert bool(g.mask(lead_e2)[i])


def test_high_never_picks_a_prefix_of_the_first_choice(tok):
    import torch
    from entropy_protocol.backend_hf import HFBackend
    g = CanonGuard(tok, len(tok))
    whole = tok(" questionnaire", add_special_tokens=False)["input_ids"]
    part = tok(" question", add_special_tokens=False)["input_ids"]
    if len(whole) != 1 or len(part) != 1:
        pytest.skip("this tokenizer splits the example differently")
    assert part[0] in g.high_exclusion(whole)
    ref = torch.full((len(tok),), -10.0)
    ref[whole[0]], ref[part[0]] = 5.0, 4.0                                # the prefix is the runner-up
    gen = torch.Generator().manual_seed(0)
    pick = HFBackend._high_choice(ref, ref.clone(), 1, 3, 0.0, 0.0, set(), gen, exclude=g.high_exclusion(whole))
    assert pick not in (whole[0], part[0])


def test_replacement_character_is_never_written(tok):
    g = CanonGuard(tok, len(tok))
    ids = tok("don" + chr(0xFFFD) + "t", add_special_tokens=False)["input_ids"]
    state, hit = 0, False
    for i in ids:
        hit |= bool(g.mask(state)[i])
        state = g.state_after(state, i)
    assert hit


def test_when_every_choice_breaks_the_join_the_best_joinable_is_taken(gpt2):
    """GPT-2 reads "\\n\\nThe" as three tokens: after the two-newline token only a word with a space (or more
    whitespace) joins as the verifier reads it, however strongly the model wants the bare word."""
    import torch
    b, g = gpt2, gpt2.canon_guard
    view = b.tok("They left.\n\n", add_special_tokens=False)["input_ids"]
    assert b.tok.convert_ids_to_tokens(view[-1]) == chr(0x10A) * 2                 # the two-newline token
    (bare,), (spaced,) = b.tok("The", add_special_tokens=False)["input_ids"], b.tok(" The", add_special_tokens=False)["input_ids"]
    assert not g.joinable(view, bare) and g.joinable(view, spaced)
    scores = torch.full((len(b.tok),), -20.0)
    scores[bare], scores[spaced] = 5.0, 1.0
    assert g.first_joinable(view, scores, set(), b.stop_ids) == spaced
