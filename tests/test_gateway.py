"""The sealing gateway (OpenAI-compatible surface) and presentation typography."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from entropy_protocol.canon import canonicalize
from entropy_protocol.typography import smarten

LQ, RQ, APO, EM, EN, ELL = (chr(c) for c in (0x201C, 0x201D, 0x2019, 0x2014, 0x2013, 0x2026))
GL, GR, LOW9 = chr(0x00AB), chr(0x00BB), chr(0x201E)


def test_smart_typography_keeps_the_canonical_form():
    texts = {
        "en": 'He said "we don\'t know" -- and then left... A list:\n- one\n- "two"\n| a | b |\n|---|---|',
        "ru": 'Она сказала: "Он ответил "нет" и ушёл". Потом - тишина...',
        "uk": "П'ять хвилин - і все. \"Так\", - сказав він.",
        "de": 'Er sagte "Hallo" - dann ging er.',
        "fr": 'Il a dit "bonjour" -- puis... rien.',
    }
    for lang, t in texts.items():
        out = smarten(t, lang)
        assert out != t and canonicalize(out) == canonicalize(t), (lang, out)
    en = smarten(texts["en"], "en")
    assert LQ + "we don" + APO + "t know" + RQ + " " + EM + " and then left" + ELL in en
    assert "\n- one\n- " + LQ + "two" + RQ + "\n| a | b |\n|---|---|" in en      # list markers and table rules stay
    ru = smarten(texts["ru"], "ru")
    assert GL + "Он ответил " + LOW9 + "нет" in ru and ru.count(GR) == 1          # nested quotes, then outer close
    code = 'Use `print("x")` and\n```\nsay "hi" -- now...\n```\nthen "go".'
    out = smarten(code, "en")
    assert '`print("x")`' in out and 'say "hi" -- now...' in out and LQ + "go" + RQ in out


pytest.importorskip("torch")
MASTER = bytes(range(32))


def _post(port, path, body, key=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type"), exc.read()


@pytest.fixture(scope="module")
def gateway():
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.gateway import Gateway, serve
    from entropy_protocol.seal import SealParams
    try:
        b = HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    gw = Gateway(b, MASTER, SealParams(), api_key="test-key")      # the reference point: every checkpoint enforced
    srv = serve(gw, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_port
    srv.shutdown()


@pytest.mark.model
def test_gateway_seals_verifies_and_detects(gateway):
    port, key = gateway, "test-key"
    assert _post(port, "/v1/completions", {"prompt": "x"})[0] == 401                  # the API key is required
    body = {"prompt": 'The mayor told reporters: "', "max_tokens": 80, "seed": 3}
    status, _, raw = _post(port, "/v1/completions", body, key)
    assert status == 200
    r = json.loads(raw)
    text, token = r["choices"][0]["text"], r["entropy"]["token"]
    assert r["object"] == "text_completion" and r["usage"]["completion_tokens"] > 0 and r["entropy"]["entries"] > 0
    v = json.loads(_post(port, "/v1/verify", {"text": text, "token": token}, key)[2])
    assert v["verdict"] == "VERIFIED", v
    edited = json.loads(_post(port, "/v1/verify", {"text": text + " And one more sentence.", "token": token}, key)[2])
    assert edited["verdict"] == "MODIFIED"
    d = json.loads(_post(port, "/v1/detect", {"text": text, "alpha": 1e-2}, key)[2])
    assert d["checkpoints"] > 0 and d["matched"] == d["checkpoints"]
    smart = json.loads(_post(port, "/v1/completions", dict(body, entropy={"typography": "smart", "lang": "en"}),
                             key)[2])
    st, stoken = smart["choices"][0]["text"], smart["entropy"]["token"]
    assert json.loads(_post(port, "/v1/verify", {"text": st, "token": stoken}, key)[2])["verdict"] == "VERIFIED"
    status, ctype, raw = _post(port, "/v1/completions", dict(body, stream=True), key)
    assert status == 200 and ctype.startswith("text/event-stream") and raw.endswith(b"data: [DONE]\n\n")
    assert json.loads(raw.split(b"\n\n")[0][6:])["entropy"]["token"]
    for bad in ({"prompt": ""}, {"prompt": "x", "n": 2}, {"prompt": "x", "max_tokens": 0}):
        assert _post(port, "/v1/completions", bad, key)[0] == 400
    assert _post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, key)[0] == 400
    assert _post(port, "/v1/nothing", {}, key)[0] == 404


@pytest.mark.model
def test_gateway_chat_completion():
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.gateway import Gateway, serve
    try:
        b = HFBackend("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen2.5-0.5B-Instruct not available offline: {exc}")
    srv = serve(Gateway(b, MASTER), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        msgs = [{"role": "user", "content": 'Write two sentences with a quotation, using "quotes".'}]
        status, _, raw = _post(srv.server_port, "/v1/chat/completions",
                               {"messages": msgs, "max_tokens": 80, "temperature": 0.7, "seed": 1,
                                "entropy": {"typography": "smart", "lang": "en"}})
        assert status == 200
        r = json.loads(raw)
        content, token = r["choices"][0]["message"]["content"], r["entropy"]["token"]
        assert r["object"] == "chat.completion" and content
        v = json.loads(_post(srv.server_port, "/v1/verify", {"text": content, "token": token})[2])
        assert v["verdict"] == "VERIFIED", v
    finally:
        srv.shutdown()


def test_operating_points():
    from entropy_protocol.profiles import DEFAULT, NAMES, detect_gate, detect_gate_for, seal_overrides
    from entropy_protocol.seal import SealParams, check_params
    assert set(NAMES) == {"quality", "strong"} and DEFAULT == "quality"
    q = SealParams(**seal_overrides("quality"))
    assert q.guard == 0.9 and q.gate == 0.0 and detect_gate("quality") == 0.9
    check_params(q)
    assert SealParams(**seal_overrides("strong")) == SealParams() and detect_gate("strong") == 0.0
    for name in NAMES:          # the measured table and the rule for other parameters agree
        assert detect_gate(name) == detect_gate_for(SealParams(**seal_overrides(name)))
    with pytest.raises(ValueError):
        seal_overrides("fast")


def test_product_surfaces_default_to_quality():
    """entropy seal / detect / serve and the gateway use the quality operating point unless told otherwise (Serhii,
    13.09.2026); an explicit detector gate of 0 is honoured (it used to fall back to the profile's)."""
    import types

    from entropy_protocol.cli import build_parser
    from entropy_protocol.gateway import Gateway
    from entropy_protocol.seal import SealParams
    p = build_parser()
    assert p.parse_args(["seal", "--model", "m", "--key", "k", "--prompt", "x"]).profile == "quality"
    served = p.parse_args(["serve", "--model", "m", "--key", "k"])
    assert served.profile == "quality" and served.gate is None and served.guard is None
    d = p.parse_args(["detect", "--model", "m", "--text-file", "t", "--key", "k"])
    assert d.profile == "quality" and d.gate is None
    assert p.parse_args(["detect", "--model", "m", "--text-file", "t", "--key", "k", "--gate", "0"]).gate == 0.0
    fake = types.SimpleNamespace(model_id="m")
    gw = Gateway(fake, MASTER)
    assert (gw.params.guard, gw.params.gate, gw.detect_gate) == (0.9, 0.0, 0.9)
    assert gw.params == SealParams(guard=0.9)
    assert Gateway(fake, MASTER, SealParams()).detect_gate == 0.0
    assert Gateway(fake, MASTER, SealParams(gate=0.9, guard=0.9)).detect_gate == 0.9
    assert Gateway(fake, MASTER, SealParams(), detect_gate=0.8).detect_gate == 0.8
    assert served.max_footprint_gb is None and gw.max_footprint is None and gw.restart_pending is None
    assert p.parse_args(["serve", "--model", "m", "--key", "k", "--max-footprint-gb", "0"]).max_footprint_gb == 0.0


def test_phys_footprint_reads_this_process():
    """The gateway's memory budget reads the process footprint from the kernel (macOS); elsewhere there is none."""
    import sys

    from entropy_protocol.gateway import phys_footprint
    before = phys_footprint()
    if sys.platform != "darwin":
        assert before is None
        return
    import mmap                     # fresh pages: a bytearray may reuse a freed block malloc still holds (delta 0)
    block = mmap.mmap(-1, 64 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    before = phys_footprint()
    for i in range(0, 64 << 20, 4096):
        block[i] = 1
    assert before > 0 and phys_footprint() >= before + (48 << 20)
    block.close()


@pytest.mark.model
def test_gateway_over_its_memory_budget_answers_then_stops_for_a_restart():
    """Over the budget, the answer that crossed it is delivered in full; the server then stops serving (`entropy
    serve` re-executes itself) and a model request meanwhile gets 503 with Retry-After."""
    import sys
    if sys.platform != "darwin":
        pytest.skip("the footprint is read on macOS")
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.gateway import Gateway, RequestError, serve
    from entropy_protocol.seal import SealParams
    try:
        b = HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    gw = Gateway(b, MASTER, SealParams(), max_footprint=1)                  # any process is over one byte
    srv = serve(gw, "127.0.0.1", 0)
    loop = threading.Thread(target=srv.serve_forever, daemon=True)
    loop.start()
    status, _, raw = _post(srv.server_port, "/v1/completions", {"prompt": "The history of the printing press shows",
                                                                "max_tokens": 24, "seed": 2})
    r = json.loads(raw)
    assert status == 200 and r["choices"][0]["text"] and r["entropy"]["token"]
    loop.join(timeout=10)
    assert not loop.is_alive() and srv.restart and "budget" in srv.restart
    with pytest.raises(RequestError) as exc:
        gw.complete({"prompt": "x", "max_tokens": 4})
    assert exc.value.status == 503
    srv.server_close()


@pytest.mark.model
def test_gateway_default_quality_point_verifies_and_detects():
    """The gateway as it comes (quality): the record verifies and the detector reads with its gate of 0.9."""
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.gateway import Gateway, serve
    try:
        b = HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    gw = Gateway(b, MASTER)
    srv = serve(gw, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = json.loads(_post(srv.server_port, "/v1/completions", {"prompt": "The history of the printing press shows",
                                                                  "max_tokens": 160, "seed": 5})[2])
        text, token = r["choices"][0]["text"], r["entropy"]["token"]
        assert json.loads(_post(srv.server_port, "/v1/verify", {"text": text, "token": token})[2])["verdict"] == "VERIFIED"
        d = json.loads(_post(srv.server_port, "/v1/detect", {"text": text, "alpha": 1e-2})[2])
        assert d["gate"] == 0.9 and d["checkpoints"] > 0 and d["matched"] / d["checkpoints"] > 0.8
    finally:
        srv.shutdown()
