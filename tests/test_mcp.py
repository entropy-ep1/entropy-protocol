"""The MCP server (stdio JSON-RPC) answers the handshake and serves the three tools; a text arriving in a call is
measured, never obeyed, and a failed check is a tool error, not a dead server."""
import io
import json

import pytest

from entropy_protocol import crypto
from entropy_protocol.mcp import MAX_CHARS, PROTOCOL_VERSIONS, Verifier, handle, serve


def ask(v, method, params=None, rid=1):
    return handle({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}, v)


def bare():
    return Verifier("distilgpt2", backend_factory=lambda: pytest.fail("no model needed for this test"))


def test_handshake_and_tools():
    v = bare()
    r = ask(v, "initialize", {"protocolVersion": PROTOCOL_VERSIONS[0], "clientInfo": {"name": "t"}})
    assert r["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]
    assert r["result"]["capabilities"]["tools"] == {"listChanged": False}
    r = ask(v, "initialize", {"protocolVersion": "1999-01-01"})          # unknown revision: ours is offered
    assert r["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]
    tools = ask(v, "tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["verify", "detect", "operating_point"]
    assert all("inputSchema" in t and t["description"] for t in tools)
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, v) is None
    assert ask(v, "nonesuch")["error"]["code"] == -32601


def test_bad_arguments_and_missing_keys_are_reported_not_raised():
    v = bare()
    assert ask(v, "tools/call", {"name": "verify", "arguments": {"text": "  "}})["error"]["code"] == -32602
    long = {"name": "detect", "arguments": {"text": "x" * (MAX_CHARS + 1)}}
    assert ask(v, "tools/call", long)["error"]["code"] == -32602
    r = ask(v, "tools/call", {"name": "verify", "arguments": {"text": "hello", "token": "t"}})
    assert r["result"]["isError"] and "master key" in r["result"]["content"][0]["text"]
    assert ask(v, "tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == -32601


def test_serve_reads_and_writes_one_json_per_line():
    v = bare()
    inp = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
                      + "not json\n"
                      + json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled"}) + "\n"
                      + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "exit"}) + "\n")
    out = io.StringIO()
    serve(v, inp, out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [x.get("id") for x in lines] == [1, None, 2]                  # the notification produced nothing
    assert lines[1]["error"]["code"] == -32700


@pytest.mark.model
def test_verify_and_detect_through_the_server():
    pytest.importorskip("torch")
    from entropy_protocol.seal import SealParams, seal
    try:
        from entropy_protocol.backend_hf import HFBackend
        backend = HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    master = crypto.new_master_key()
    sealed = seal(backend, "The history of the printing press shows", master, SealParams(max_new_tokens=60), seed=7)
    v = Verifier("distilgpt2", master=master, wm_key=crypto.wm_key(master), backend_factory=lambda: backend)
    ok = ask(v, "tools/call", {"name": "verify", "arguments": {"text": sealed.text, "token": sealed.token}})
    assert ok["result"]["structuredContent"]["verdict"] == "VERIFIED"
    words = sealed.text.split()
    words[len(words) // 2] = "zebra"
    bad = ask(v, "tools/call", {"name": "verify", "arguments": {"text": " ".join(words), "token": sealed.token}})
    assert bad["result"]["structuredContent"]["verdict"] == "MODIFIED"
    det = ask(v, "tools/call", {"name": "detect", "arguments": {"text": sealed.text}})["result"]
    assert 0.0 <= det["structuredContent"]["p_value"] <= 1.0
    point = ask(v, "tools/call", {"name": "operating_point"})["result"]["structuredContent"]
    assert point["weights_digest"] and point["profile"] == "quality" and point["detect_gate"] > 0
