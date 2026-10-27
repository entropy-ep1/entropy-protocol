#!/bin/bash
# Attested sealing end to end on one Azure confidential H100 (ARCHITECTURE §7d).
# Here (the authority's side): the policy and a fresh authority; there (the guest): the sealing service, which loads
# Qwen2.5-7B-Instruct, generates its keys in memory and enrolls twice - a statement key, a detection key - through a
# file conduit over SSH (an untrusted transport: the guest pins the authority's root and rrp_id in its own command
# line). The authority appraises each request with live collateral and certifies the key only when every check
# holds. Then the guest seals, delivers each record to the verifier's public key, signs generation statements, and
# detects sealed and unsealed texts; a separate verifier process checks every text as sealed, with one word changed
# and after ordinary transport. Everything is fetched, the resource group deleted and its absence confirmed.
# Evidence: bench/results/attested/run-<stamp>/ (tracked, except the authority's root key and the verifier's key);
# the operator log (it names the operator's address) goes to bench/gpu/infra/ (ignored).
# Usage: bash bench/attested/run_attested_azure.sh [texts] [controls]
set -euo pipefail
N=${1:-20}
NC=${2:-10}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
INFRA="$REPO/bench/gpu/infra"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$REPO/bench/results/attested/run-$STAMP"
LOG="$INFRA/attested_run_$STAMP.log"
NAME="atr$(date -u +%H%M)"
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15
     -o ServerAliveCountMax=8 -o ConnectTimeout=20 -o LogLevel=ERROR)
SCP=(scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -q)
PY="$REPO/.venv/bin/python"
CAL=runs_gpu/a1/Qwen_Qwen2.5-7B-Instruct/calibration_A_cal_fp16only.json
TOKENIZER=f884026e05f6dfffe68b72d580b588006d98b3bc0a128bb1ac425c76f7fc438c
RRP="urn:ep1:authority:reference-$STAMP"
TARGET="urn:ep1:sealer:azure-cgpu:$STAMP"
VERIFIER="urn:ep1:verifier:reference-$STAMP"
mkdir -p "$RUN/conduit" "$RUN/guest"
exec > >(tee -a "$LOG") 2>&1
RG=""

cleanup() {
  set +e
  if [ -n "$RG" ]; then
    echo "cleanup: deleting $RG ($(date -u +%H:%M:%SZ))"
    az group delete -n "$RG" --yes </dev/null
    for _ in $(seq 1 30); do
      az group show -n "$RG" >/dev/null 2>&1 </dev/null || { echo "cleanup: $RG is gone"; return; }
      sleep 20
    done
    echo "cleanup: $RG STILL EXISTS - delete it by hand"
  fi
}
trap cleanup EXIT

remote_wait() {   # remote_wait <file under /home/ubuntu> <minutes>
  local end=$(( $(date +%s) + $2 * 60 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    "${SSH[@]}" "ubuntu@$IP" "test -s /home/ubuntu/$1" </dev/null && return 0
    "${SSH[@]}" "ubuntu@$IP" "pgrep -f 'attested[.]service seal' >/dev/null" </dev/null || {
      echo "the sealing service is not running; its log:"; "${SSH[@]}" "ubuntu@$IP" 'tail -40 /home/ubuntu/seal.log' </dev/null
      return 1; }
    sleep 5
  done
  echo "timeout waiting for $1"; return 1
}

echo "0. the code that ships is the committed tree"
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo "uncommitted changes"; exit 1; }
COMMIT=$(git -C "$REPO" rev-parse HEAD)
echo "$COMMIT" > "$RUN/commit.txt"
TREE=$(mktemp -d)
git -C "$REPO" archive --format=tar HEAD src | tar -C "$TREE" -xf -

echo "1. policy and authority (here)"
(cd "$REPO" && PYTHONPATH="$TREE/src" "$PY" -W ignore bench/attested/make_policy.py \
   --capture bench/results/attested/capture-20261027T162455Z --cal "$CAL" --tokenizer "$TOKENIZER" \
   --out "$RUN/policy.json" >/dev/null)
PYTHONPATH="$REPO/src" "$PY" -W ignore -m entropy_protocol.attested.authority init --dir "$RUN/authority" \
  --rrp-id "$RRP" --name "Entropy Protocol reference authority $STAMP" --policy "$RUN/policy.json"
ROOT_SHA=$(shasum -a 256 "$RUN/authority/root.der" | cut -d' ' -f1)
"$PY" -I - "$REPO/runs_gpu/a1/Qwen_Qwen2.5-7B-Instruct/A_c4_plain.jsonl" "$RUN/controls.jsonl" "$NC" <<'EOF'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()][: int(sys.argv[3])]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for r in rows:   # unsealed Qwen2.5-7B-Instruct texts from phase A: the detector must not fire on them
        f.write(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n")
EOF

echo "2. machine ($(date -u +%H:%M:%SZ))"
bash "$INFRA/launch_azure.sh" "$NAME" 3 westeurope </dev/null
LINE=$(awk -F'\t' -v n="$NAME" '$1 == n' "$INFRA/instances.tsv" | tail -1)
RG=$(echo "$LINE" | cut -f2 | cut -d/ -f1)
IP=$(echo "$LINE" | cut -f5)
[ -n "$RG" ] && [ -n "$IP" ] || { echo "no ledger line for $NAME"; exit 1; }
echo "machine up: $RG ($(date -u +%H:%M:%SZ))"

echo "3. guest bootstrap"
"${SCP[@]}" "$REPO/bench/attested/guest_bootstrap.sh" "ubuntu@$IP:"
"${SSH[@]}" "ubuntu@$IP" 'bash guest_bootstrap.sh' </dev/null
git -C "$REPO" archive --format=tar HEAD src | "${SSH[@]}" "ubuntu@$IP" 'mkdir -p ep && tar -C ep -xf -'
"${SCP[@]}" "$REPO/$CAL" "ubuntu@$IP:ep/cal.json"
"${SCP[@]}" "$REPO/data/prose_offline.jsonl" "$RUN/controls.jsonl" "ubuntu@$IP:ep/data/"
"${SCP[@]}" "$RUN/authority/root.der" "ubuntu@$IP:io/root.der"

echo "4. the sealing service (guest, root: the vTPM needs it)"
ENVS="HF_HOME=/home/ubuntu/hf PYTHONPATH=/home/ubuntu/ep/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore"
"${SSH[@]}" "ubuntu@$IP" "sudo env $ENVS nohup /home/ubuntu/venv/bin/python -m entropy_protocol.attested.service seal \
  --io /home/ubuntu/io --out /home/ubuntu/out --model Qwen/Qwen2.5-7B-Instruct --cal /home/ubuntu/ep/cal.json \
  --prompts /home/ubuntu/ep/data/prose_offline.jsonl --n $N --controls /home/ubuntu/ep/data/controls.jsonl \
  --n-controls $NC --verifier-pub /home/ubuntu/io/verifier.pub --rrp-id '$RRP' --root-sha256 $ROOT_SHA \
  --target '$TARGET' > /home/ubuntu/seal.log 2>&1 &" </dev/null
remote_wait io/ready.json 20
"${SCP[@]}" "ubuntu@$IP:io/ready.json" "$RUN/guest/ready.json"
echo "model loaded: $(cat "$RUN/guest/ready.json" | head -c 300)"

echo "5. enrollments through the conduit"
for role in seal detect; do
  PYTHONPATH="$REPO/src" "$PY" -W ignore -m entropy_protocol.attested.authority initiate --dir "$RUN/authority" \
    --ttl 900 --out "$RUN/conduit/initiation-$role.cbor"
  "${SCP[@]}" "$RUN/conduit/initiation-$role.cbor" "ubuntu@$IP:io/"
  remote_wait "io/request-$role.cbor" 10
  "${SCP[@]}" "ubuntu@$IP:io/request-$role.cbor" "$RUN/conduit/"
  echo "appraising the $role request ($(date -u +%H:%M:%SZ))"
  PYTHONPATH="$REPO/src" "$PY" -W ignore -m entropy_protocol.attested.authority enroll --dir "$RUN/authority" \
    --request "$RUN/conduit/request-$role.cbor" --ear-out "$RUN/conduit/ear-$role.cose" \
    --cert-out "$RUN/conduit/cert-$role.der" || { echo "the authority refused the $role key"; exit 1; }
  "${SCP[@]}" "$RUN/conduit/ear-$role.cose" "$RUN/conduit/cert-$role.der" "ubuntu@$IP:io/"
done
remote_wait io/done.json 60
echo "sealing done ($(date -u +%H:%M:%SZ)): $("${SSH[@]}" "ubuntu@$IP" 'cat /home/ubuntu/io/done.json' </dev/null)"

echo "6. the verifier (separate process, its own keys)"
"${SSH[@]}" "ubuntu@$IP" "sudo env $ENVS /home/ubuntu/venv/bin/python -m entropy_protocol.attested.service verify \
  --out /home/ubuntu/out --model Qwen/Qwen2.5-7B-Instruct --verifier-key /home/ubuntu/verifier/verifier.key \
  --verifier-id '$VERIFIER' > /home/ubuntu/verify.log 2>&1" </dev/null || echo "verifier rc=$?"

echo "7. fetch"
"${SSH[@]}" "ubuntu@$IP" 'sudo tar -C /home/ubuntu -czf - out io seal.log verify.log pip_freeze.txt' </dev/null \
  | tar -C "$RUN/guest" -xzf -
mkdir -p "$RUN/verifier"
"${SSH[@]}" "ubuntu@$IP" 'cat /home/ubuntu/verifier/verifier.key' </dev/null > "$RUN/verifier/verifier.key"
chmod 600 "$RUN/verifier/verifier.key"
"$PY" -I - "$RUN" <<'EOF'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1]); g = run / "guest"
m = json.loads((g / "out/manifest.json").read_text())
v = json.loads((g / "out/verification.json").read_text()) if (g / "out/verification.json").exists() else []
from collections import Counter
print("sealed", len(m["sealed"]), "| detections", Counter((d["kind"], d["detected"]) for d in m["detections"]))
print("verdicts", Counter((r["kind"], r["verdict"]) for r in v))
EOF
echo "done ($(date -u +%H:%M:%SZ)): $RUN"
