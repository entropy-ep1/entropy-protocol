"""Sealing gateway with an OpenAI-compatible surface (internal tool): completions come back sealed.

  POST /v1/chat/completions   OpenAI request -> OpenAI response, plus "entropy": {doc_id, token, entries, canon, ...}
  POST /v1/completions        prompt -> text completion, sealed the same way
  POST /v1/verify             {"text", "token"} -> the verdict against the record (the gateway holds the master key)
  POST /v1/detect             {"text", "localize"?, "alpha"?} -> record-free detection under the gateway's key
  GET  /v1/models             the served model and its operating point (gate, guard, detector gate)
  GET  /                      a local page: paste a text (and its token) and check it
One model, one request at a time (a lock); standard library only. Binds to 127.0.0.1 unless told otherwise; with an
API key, every /v1 request needs "Authorization: Bearer <key>". "stream": true is answered as one server-sent event
chunk and [DONE] (the record is written when the text is complete).
Operating point: profiles.DEFAULT (quality: guard 0.9 at sealing, detector gate 0.9) unless given.
Memory budget (macOS): over it after a model pass, the answer goes out, model requests get 503, and `entropy serve`
restarts itself (Gateway._model).
Typography: the sealed text is canonical (straight quotes, "--" for a long dash). A request may ask for
"entropy": {"typography": "smart", "lang": "ru"}; quotes, dashes and ellipses are then set for that language by
substitution only (typography.py), so canonical form v4 folds them back and the text still verifies.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hmac
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import crypto
from .batch import seal_batch
from .canon import CANON_ID, SUPPORTED, canonicalize
from .detect import detect
from .profiles import DEFAULT, detect_gate_for, seal_overrides
from .seal import SealParams
from .typography import smarten
from .verify import verify


class RequestError(Exception):
    def __init__(self, status: int, message: str, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.code = status, code


def phys_footprint() -> int | None:
    """This process's physical footprint in bytes (what top and Activity Monitor show as its memory), read from the
    kernel (macOS task_info, TASK_VM_INFO); None where that call does not exist."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_")
        buf, count = (ctypes.c_uint32 * 128)(), ctypes.c_uint32(128)
        if libc.task_info(task, 22, ctypes.byref(buf), ctypes.byref(count)) != 0 or count.value < 38:
            return None
        return int(ctypes.c_uint64.from_buffer(buf, 144).value)            # task_vm_info.phys_footprint
    except Exception:  # noqa: BLE001
        return None


class Gateway:
    def __init__(self, backend, master: bytes, params: SealParams | None = None, api_key: str | None = None,
                 model_name: str | None = None, max_tokens: int = 1024, detect_gate: float | None = None,
                 max_footprint: int | None = None):
        """``params``: default the product's operating point (profiles.DEFAULT, quality). ``detect_gate``: the
        record-free detector's confidence gate; default the one that goes with ``params`` (profiles.detect_gate_for).
        ``max_footprint``: a memory budget in bytes (see _model); None = none."""
        params = SealParams(**seal_overrides(DEFAULT)) if params is None else params
        self.backend, self.master, self.params, self.api_key = backend, master, params, api_key
        self.detect_gate = detect_gate_for(params) if detect_gate is None else detect_gate
        self.model_name = model_name or backend.model_id
        self.max_tokens = max_tokens
        self.k_wm = crypto.wm_key(master)
        self.lock = threading.Lock()
        self.max_footprint = max_footprint
        self.restart_pending: str | None = None      # why the gateway asks to be restarted

    @contextlib.contextmanager
    def _model(self):
        """The model, one request at a time. On MPS a process keeps a compiled graph for every tensor shape it has
        run, outside the allocator (14.09.2026: Phi-3 sealing 10 batches of new lengths +1.8 GB, the same batch 8
        times +0.13 GB), so a long-running gateway grows with the prompt and text lengths it sees. After each model
        pass, over ``max_footprint``, the gateway takes no more model requests (503, Retry-After) and asks to be
        restarted once this answer is out (`entropy serve` then re-executes itself: all memory back, same port)."""
        with self.lock:
            if self.restart_pending:
                raise RequestError(503, "the gateway is restarting to give its memory back; retry in 30 s",
                                   "server_restarting")
            yield
            if self.max_footprint:
                fp = phys_footprint()
                if fp is not None and fp > self.max_footprint:
                    self.restart_pending = (f"footprint {fp / 2 ** 30:.2f} GB over the budget of "
                                            f"{self.max_footprint / 2 ** 30:.2f} GB")

    # ----------------------------------------------------------------------------------------------- requests
    def _params(self, body: dict, prompt_len: int) -> SealParams:
        n = body.get("n", 1)
        if n != 1:
            raise RequestError(400, "only n = 1 is supported")
        max_new = body.get("max_completion_tokens", body.get("max_tokens", self.params.max_new_tokens))
        temperature = body.get("temperature", self.params.temperature)
        top_p = body.get("top_p", self.params.top_p)
        try:
            max_new, temperature, top_p = int(max_new), float(temperature), float(top_p)
        except (TypeError, ValueError):
            raise RequestError(400, "max_tokens, temperature and top_p must be numbers") from None
        if not 1 <= max_new <= self.max_tokens:
            raise RequestError(400, f"max_tokens must be between 1 and {self.max_tokens}")
        if prompt_len + max_new + 1 > self.backend.max_positions:
            raise RequestError(400, f"prompt ({prompt_len} tokens) and max_tokens exceed the model context "
                                    f"({self.backend.max_positions})")
        if temperature < 0 or not 0 < top_p <= 1:
            raise RequestError(400, "temperature must be >= 0 and top_p in (0, 1]")
        return dataclasses.replace(self.params, max_new_tokens=max_new, temperature=temperature, top_p=top_p,
                                   min_new_tokens=min(self.params.min_new_tokens, max_new))

    def _seal(self, ids: list[int], body: dict) -> dict:
        params = self._params(body, len(ids))
        seed = body.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise RequestError(400, "seed must be an integer")
        t0 = time.time()
        with self._model():
            (s,) = seal_batch(self.backend, [ids], self.master, params, seed=seed)
        opts = body.get("entropy") or {}
        text = s.text
        if opts.get("typography") == "smart":
            text = smarten(text, str(opts.get("lang", "en")))
        return {"text": text, "finish": "length" if s.tokens >= params.max_new_tokens else "stop",
                "usage": {"prompt_tokens": len(ids), "completion_tokens": s.tokens,
                          "total_tokens": len(ids) + s.tokens},
                "entropy": {"doc_id": s.doc_id, "token": s.token, "entries": s.entries, "enforced": s.enforced,
                            "canon": CANON_ID, "typography": "smart" if text != s.text else "canonical",
                            "seconds": round(time.time() - t0, 3)}}

    def chat(self, body: dict) -> dict:
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs or not all(
                isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                for m in msgs):
            raise RequestError(400, "messages must be a non-empty list of {role, content} with text content")
        tok = self.backend.tok
        if not getattr(tok, "chat_template", None):
            raise RequestError(400, f"{self.model_name} has no chat template; use /v1/completions")
        ids = list(tok.apply_chat_template([{"role": m["role"], "content": m["content"]} for m in msgs],
                                           add_generation_prompt=True, tokenize=True))
        r = self._seal(ids, body)
        return {"id": f"chatcmpl-{r['entropy']['doc_id']}", "object": "chat.completion", "created": int(time.time()),
                "model": self.model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]},
                             "finish_reason": r["finish"]}],
                "usage": r["usage"], "entropy": r["entropy"]}

    def complete(self, body: dict) -> dict:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RequestError(400, "prompt must be a non-empty string")
        r = self._seal(self.backend.tok(prompt)["input_ids"], body)
        return {"id": f"cmpl-{r['entropy']['doc_id']}", "object": "text_completion", "created": int(time.time()),
                "model": self.model_name,
                "choices": [{"index": 0, "text": r["text"], "finish_reason": r["finish"], "logprobs": None}],
                "usage": r["usage"], "entropy": r["entropy"]}

    def verify(self, body: dict) -> dict:
        text, token = body.get("text"), body.get("token")
        if not isinstance(text, str) or not isinstance(token, str):
            raise RequestError(400, "text and token must be strings")
        try:
            key = crypto.log_key(self.master, crypto.token_doc_id(token))
        except Exception:  # noqa: BLE001
            raise RequestError(400, "token is not a custody token") from None
        with self._model():
            out = verify(self.backend, text, token, key).to_dict()
        canon = out.get("decoding", {}).get("canon", CANON_ID)
        out["canonical_text"] = canonicalize(text, canon if canon in SUPPORTED else CANON_ID)   # ranges index it
        return out

    def detect(self, body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, str):
            raise RequestError(400, "text must be a string")
        alpha = float(body.get("alpha", 1e-6))
        with self._model():
            out = detect(self.backend, text, self.k_wm, alpha=alpha, localize=bool(body.get("localize", False)),
                         gate=self.detect_gate).to_dict()
        out["canonical_text"] = canonicalize(text, out.get("canon", CANON_ID))                   # segment indexes it
        return out

    def models(self) -> dict:
        point = {"gate": self.params.gate, "guard": self.params.guard, "detect_gate": self.detect_gate}
        return {"object": "list", "data": [{"id": self.model_name, "object": "model", "owned_by": "entropy-gateway",
                                            "entropy": {"weights": self.backend.weights_digest, "canon": CANON_ID,
                                                        "operating_point": point}}]}


PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entropy gateway: check a text</title>
<style>
:root{--ink:#1d1d1f;--mute:#6e6e73;--line:#d9d9de;--bg:#fbfbfd;--ok:#1a7f37;--bad:#c62828;--warn:#b26a00;--gray:#6e6e73}
body{font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:var(--ink);
background:var(--bg)}h1{font-size:1.35em;margin:0 0 .2em}p.sub{color:var(--mute);margin:0 0 1.2em}
label{font-weight:600;font-size:.9em}textarea,input{width:100%;box-sizing:border-box;font:13px/1.45 ui-monospace,
monospace;padding:.55em;margin:.3em 0 .9em;border:1px solid var(--line);border-radius:8px;background:#fff}
button{font:inherit;padding:.5em 1.1em;margin:0 .5em .5em 0;border:1px solid var(--line);border-radius:8px;
background:#fff;cursor:pointer}button.primary{background:var(--ink);color:#fff;border-color:var(--ink)}
.badge{display:inline-block;padding:.25em .8em;border-radius:999px;font-weight:700;color:#fff;margin:.6em 0}
.why{color:var(--mute);margin:.2em 0 .8em}.text{white-space:pre-wrap;background:#fff;border:1px solid var(--line);
border-radius:8px;padding:1em;font:14px/1.6 Georgia,serif}mark.chg{background:#ffd7d5}mark.seg{background:#d7f5dd}
u.div{text-decoration:underline wavy var(--warn)}span.del{color:var(--bad);font-weight:700}
.legend{color:var(--mute);font-size:.85em;margin:.5em 0}details{margin-top:1em}pre{background:#f1f1f4;padding:1em;
border-radius:8px;overflow:auto;white-space:pre-wrap;font-size:12px}
</style>
<h1>Check a text</h1>
<p class="sub">Paste a text. With its token, verification tells whether it is exactly the sealed text and where it
changed. Without a token, detection tells whether it was sealed by this gateway at all.</p>
<label for="t">Text</label><textarea id="t" rows="9"></textarea>
<label for="k">Token (from the response that delivered the text)</label><input id="k" autocomplete="off">
<label for="a">API key (only if the gateway requires one)</label><input id="a" type="password" autocomplete="off">
<button class="primary" onclick="go('verify')">Verify with the token</button>
<button onclick="go('detect')">Detect without the token</button>
<div id="res"></div>
<script>
const COLORS = {VERIFIED: 'var(--ok)', VERIFIED_TEXT: 'var(--ok)', MODIFIED: 'var(--bad)', INDETERMINATE: 'var(--warn)', NOT_VERIFIABLE: 'var(--gray)'};
const LABELS = {VERIFIED_TEXT: 'VERIFIED: TEXT'};
const WHY = {VERIFIED: 'This is the sealed text: every character matches and the model check holds.',
  VERIFIED_TEXT: 'The visible text is the sealed one: a rendered copy (e.g. out of a chat window), every word matches. Formatting (Markdown markup, line breaks, list numbering) is not checked.',
  MODIFIED: 'The text differs from the sealed one. Changed passages are marked; a red caret marks deleted text.',
  INDETERMINATE: 'Same text, but the model check could not be confirmed on this machine (numerics outside the calibrated set).',
  NOT_VERIFIABLE: 'The token does not open with this gateway key, or it is not a token.'};
function esc(s) { return s.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c])); }
function render(text, marks) {           // marks: [start, end, kind]; kind: chg | seg | div | del (point)
  const cuts = new Map();
  for (const [a, b, kind] of marks) {
    if (kind === 'del') { (cuts.get(a) || cuts.set(a, []).get(a)).push('<span class="del" title="deleted text">&#9660;</span>'); continue; }
    const [o, c] = kind === 'div' ? ['<u class="div">', '</u>'] : ['<mark class="' + kind + '">', '</mark>'];
    (cuts.get(a) || cuts.set(a, []).get(a)).push(o); (cuts.get(b) || cuts.set(b, []).get(b)).unshift(c);
  }
  let out = '', last = 0;
  for (const i of [...cuts.keys()].sort((x, y) => x - y)) { out += esc(text.slice(last, i)) + cuts.get(i).join(''); last = i; }
  return out + esc(text.slice(last));
}
async function go(kind) {
  const res = document.getElementById('res');
  const body = kind === 'verify' ? {text: t.value, token: k.value.trim()} : {text: t.value, localize: true};
  const h = {'Content-Type': 'application/json'}; if (a.value) h['Authorization'] = 'Bearer ' + a.value;
  res.innerHTML = '<p class="why">Checking...</p>';
  let j;
  try { const r = await fetch('/v1/' + kind, {method: 'POST', headers: h, body: JSON.stringify(body)}); j = await r.json(); }
  catch (e) { res.innerHTML = '<p class="why">Request failed: ' + esc(String(e)) + '</p>'; return; }
  if (j.error) { res.innerHTML = '<p class="why">Error: ' + esc(j.error.message) + '</p>'; return; }
  let head, marks = [], shown = null, legend = 'Shown in canonical form (typographic quotes and dashes folded, spacing normalized).';
  if (kind === 'verify') {
    head = '<span class="badge" style="background:' + (COLORS[j.verdict] || 'var(--gray)') + '">' + esc(LABELS[j.verdict] || j.verdict) + '</span>'
         + '<p class="why">' + esc(WHY[j.verdict] || j.reason || '') + '</p>';
    const size = rs => (rs || []).reduce((n, [x, y]) => n + y - x, 0);
    const tv = j.text_view;
    if (tv && size(tv.changed_ranges) < size(j.changed_ranges)) {     // a rendered copy: its visible text shows the change
      shown = tv.text; legend = 'Shown as visible text: a rendered copy, compared without markup and line breaks.';
      for (const [x, y] of tv.changed_ranges) marks.push([x, y, 'chg']);
      for (const x of tv.deletion_points) marks.push([x, x, 'del']);
    } else {
      for (const [x, y] of j.changed_ranges || []) marks.push([x, y, 'chg']);
      for (const [x, y] of j.divergence_ranges || []) marks.push([x, y, 'div']);
      for (const x of j.deletion_points || []) marks.push([x, x, 'del']);
    }
  } else {
    const p = Math.min(j.p_value, j.segment_p_corrected == null ? 1 : j.segment_p_corrected);
    const [label, color, note] = j.detected ? ['SEALED BY THIS GATEWAY', 'var(--ok)', 'The whole text carries the mark.']
      : j.segment_detected ? ['SEALED PASSAGE FOUND', 'var(--ok)', 'A passage carries the mark (highlighted); the rest does not.']
      : p <= 1e-2 ? ['WEAK SIGNAL', 'var(--warn)', 'The mark is there, but the text is too short to decide at alpha = '
                     + Number(j.alpha).toExponential(0) + ' (about 150-200 tokens are needed).']
      : ['NO SIGNAL', 'var(--gray)', 'Not sealed by this gateway, or edited beyond recognition.'];
    head = '<span class="badge" style="background:' + color + '">' + label + '</span>'
         + '<p class="why">' + note + ' p = ' + Number(j.p_value).toExponential(2) + ' over ' + j.checkpoints
         + ' checkpoints (' + j.matched + ' matched)' + (j.segment_p_corrected != null ? '; best passage p = '
         + Number(j.segment_p_corrected).toExponential(2) + ' after correction' : '') + '.</p>';
    if (j.segment_detected && j.segment) marks.push([j.segment[0], j.segment[1], 'seg']);
  }
  res.innerHTML = head + '<div class="text">' + render(shown != null ? shown : (j.canonical_text || ''), marks) + '</div>'
    + '<p class="legend">' + legend + '</p>'
    + '<details><summary>Full result</summary><pre>' + esc(JSON.stringify(j, null, 1)) + '</pre></details>';
}
</script>"""


def make_handler(gw: Gateway):
    routes = {"/v1/chat/completions": gw.chat, "/v1/completions": gw.complete, "/v1/verify": gw.verify,
              "/v1/detect": gw.detect}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):             # quiet: requests carry texts and tokens
            pass

        def _send(self, status: int, payload, ctype: str = "application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if status == 503:
                self.send_header("Retry-After", "30")
            self.end_headers()
            self.wfile.write(data)
            if gw.restart_pending and not getattr(self.server, "restart", None):
                # the answer that crossed the memory budget is out: stop serving; `entropy serve` restarts itself
                self.server.restart = gw.restart_pending
                self.close_connection = True
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def _error(self, status: int, message: str, code: str = "invalid_request_error"):
            self._send(status, {"error": {"message": message, "type": code, "code": status}})

        def _authorized(self) -> bool:
            if not gw.api_key:
                return True
            got = self.headers.get("Authorization", "")
            return got.startswith("Bearer ") and hmac.compare_digest(got[7:].encode(), gw.api_key.encode())

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/v1/models":
                if not self._authorized():
                    return self._error(401, "missing or wrong API key", "authentication_error")
                return self._send(200, gw.models())
            return self._error(404, "not found", "not_found_error")

        def do_POST(self):
            fn = routes.get(self.path.split("?")[0])
            if fn is None:
                return self._error(404, "not found", "not_found_error")
            if not self._authorized():
                return self._error(401, "missing or wrong API key", "authentication_error")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 4_000_000:
                    raise RequestError(413, "request too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise RequestError(400, "the request body must be a JSON object")
                out = fn(body)
            except RequestError as exc:
                return self._error(exc.status, str(exc), exc.code)
            except json.JSONDecodeError:
                return self._error(400, "the request body is not JSON")
            except Exception as exc:  # noqa: BLE001
                return self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
            if body.get("stream") and fn in (gw.chat, gw.complete):
                chunk = dict(out, object=out["object"] + ".chunk")
                if fn == gw.chat:
                    chunk["choices"] = [{"index": 0, "delta": {"role": "assistant",
                                                               "content": out["choices"][0]["message"]["content"]},
                                         "finish_reason": out["choices"][0]["finish_reason"]}]
                data = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()
                return self._send(200, data, "text/event-stream")
            return self._send(200, out)

    return Handler


def serve(gw: Gateway, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """A running server (call .serve_forever(), or run it in a thread and .shutdown() it). When .serve_forever()
    returns with .restart set, the gateway went over its memory budget (Gateway._model): the caller restarts it."""
    srv = ThreadingHTTPServer((host, port), make_handler(gw))
    srv.restart = None
    return srv
