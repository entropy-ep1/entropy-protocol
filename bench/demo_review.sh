#!/bin/bash
# The whole product on one machine, one command: seal through the gateway, verify the text, a rendered copy and an
# edited copy, detect without the record, the verifier as an assistant's tool, and an evidence package checked
# offline. A small model on the CPU, so it runs anywhere; nothing leaves this machine.
# Usage: bash bench/demo_review.sh [dir] [model] [port]
set -uo pipefail
cd "$(dirname "$0")/.."
D=${1:-/tmp/entropy-demo}; M=${2:-Qwen/Qwen2.5-0.5B-Instruct}; PORT=${3:-8099}
export PYTHONPATH=$PWD/src PYTHONWARNINGS=ignore
rm -rf "$D"; mkdir -p "$D"
e() { python3 -m entropy_protocol.cli "$@"; }
say() { printf "\n\033[1m%s\033[0m\n" "$*"; }

say "1. Keys: the issuer's master key, and signing keys for the issuer, the verifier and the log"
e keygen --out "$D/master.key" > /dev/null
for n in issuer verifier log; do e signkey --out "$D/$n.pem" > /dev/null; done
echo "   $D: master.key, issuer.pem, verifier.pem, log.pem"

say "2. Sealing gateway (OpenAI-compatible) on http://127.0.0.1:$PORT"
nohup python3 -m entropy_protocol.cli serve --model "$M" --key "$D/master.key" --port "$PORT" \
  > "$D/gateway.log" 2>&1 &
GW=$!
trap 'kill $GW 2>/dev/null' EXIT
for _ in $(seq 240); do curl -s -m 1 "http://127.0.0.1:$PORT/v1/models" -o /dev/null && break; sleep 0.5; done
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import json, sys
d = json.load(sys.stdin)['data'][0]
print('   model', d['id'], '| operating point', d['entropy']['operating_point'])"

say "3. A sealed answer, asked in Markdown (the way a chat answers)"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "'"$M"'", "max_tokens": 220, "messages": [{"role": "user",
  "content": "In three short bullet points with bold lead-ins, say why receipts beat guesses for AI text."}]}' \
  > "$D/answer.json"
python3 - "$D" <<'PY'
import json, sys
d = json.load(open(sys.argv[1] + "/answer.json"))
open(sys.argv[1] + "/text.txt", "w").write(d["choices"][0]["message"]["content"])
open(sys.argv[1] + "/token.txt", "w").write(d["entropy"]["token"])
print("   text:", d["choices"][0]["message"]["content"].replace("\n", " ")[:90] + "…")
print("   record:", d["entropy"]["entries"], "checkpoints, doc", d["entropy"]["doc_id"])
PY

say "4. The verifier: the text as delivered, a copy rendered without Markdown, one edited word"
python3 - "$D" "$PORT" <<'PY'
import json, sys, urllib.request
sys.path.insert(0, "src")
from entropy_protocol import plain
d, port = sys.argv[1], sys.argv[2]
text, token = open(d + "/text.txt").read(), open(d + "/token.txt").read()
visible = plain.render_visible(text)          # exactly what a reader copies once the Markdown is rendered
words = text.split()
edited = text.replace(words[len(words) // 2], "zebra", 1)
cases = [("as delivered", text), ("one word changed", edited)]
cases.insert(1, ("rendered copy", visible) if visible.strip() != text.strip()
             else ("rendered copy (no markup in this answer)", visible))
for name, cand in cases:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/verify", method="POST",
                                 data=json.dumps({"text": cand, "token": token}).encode(),
                                 headers={"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    print(f"   {name:34s} -> {r['verdict']:14s} matched {r['matched']}/{r['entries']}"
          + (f", changed ranges {r['changed_ranges'][:2]}" if r.get("changed_ranges") else ""))
PY

say "5. Detection without the record: the exact p-value (the sign needs ~400 tokens at the quality point)"
python3 - "$D" "$PORT" <<'PY'
import json, sys, urllib.request
d, port = sys.argv[1], sys.argv[2]
human = ("The committee met on Tuesday in the small hall behind the library, and the minutes were taken by hand "
         "because the recorder had been lent to the school for the week.")
for name, cand in (("sealed text", open(d + "/text.txt").read()), ("human text", human)):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/detect", method="POST",
                                 data=json.dumps({"text": cand}).encode(),
                                 headers={"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    print(f"   {name:12s} -> p = {r['p_value']:.3g} over {r['checkpoints']} checkpoints"
          f" (detected at 1e-6: {r['detected']})")
PY

say "6. The verifier as a tool for an assistant (MCP over stdio)"
python3 - "$D" "$M" <<'PY'
import json, subprocess, sys
d, model = sys.argv[1], sys.argv[2]
msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "verify", "arguments": {"text": open(d + "/text.txt").read(),
                                                    "token": open(d + "/token.txt").read()}}},
        {"jsonrpc": "2.0", "id": 4, "method": "exit"}]
p = subprocess.run([sys.executable, "-m", "entropy_protocol.cli", "mcp", "--model", model,
                    "--key", d + "/master.key"],
                   input="\n".join(json.dumps(m) for m in msgs), capture_output=True, text=True)
for line in p.stdout.splitlines():
    r = json.loads(line).get("result", {})
    if "tools" in r:
        print("   tools:", ", ".join(t["name"] for t in r["tools"]))
    if "structuredContent" in r:
        print("   verify through the tool ->", r["structuredContent"]["verdict"])
PY

say "7. Statements, the transparency log, and an evidence package checked offline"
TOKEN=$(cat "$D/token.txt")
e commit --key "$D/master.key" --token "$TOKEN" --sign "$D/issuer.pem" --issuer "demo-host" \
  --out "$D/gen.cose" > /dev/null
e register --log "$D/log.jsonl" --log-key "$D/log.pem" --statement "$D/gen.cose" \
  --receipt-out "$D/gen_rc.cose" > /dev/null
e verify --model "$M" --text-file "$D/text.txt" --token "$TOKEN" --key "$D/master.key" \
  --sign "$D/verifier.pem" --verifier-id "demo-verifier" --generation "$D/gen.cose" \
  --statement-out "$D/ver.cose" > /dev/null
e register --log "$D/log.jsonl" --log-key "$D/log.pem" --statement "$D/ver.cose" \
  --receipt-out "$D/ver_rc.cose" > /dev/null
e evidence --out "$D/package.cbor" --text-file "$D/text.txt" --token "$TOKEN" \
  --generation "$D/gen.cose" --gen-receipt "$D/gen_rc.cose" \
  --verification "$D/ver.cose" --ver-receipt "$D/ver_rc.cose" \
  --issuer-pub "$D/issuer.pem" --verifier-pub "$D/verifier.pem" --log-pub "$D/log.pem" \
  --note "demo run on $(date -u +%Y-%m-%d)" > /dev/null
e evidence --check "$D/package.cbor" --out-dir "$D/package" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print('   package checks:', sum(1 for c in r['checks'] if c['ok']), 'passed,',
      sum(1 for c in r['checks'] if c['ok'] is False), 'failed,',
      sum(1 for c in r['checks'] if c['ok'] is None), 'left to the reader')
print('   files:', ', '.join(r['written'][:4]) + ', …')"

say "Open http://127.0.0.1:$PORT in a browser: paste the text and the token, then change a word."
echo "   files: $D · gateway log: $D/gateway.log · package: $D/package/README.md"
echo "   the gateway stays up until you press Ctrl+C"
wait $GW
