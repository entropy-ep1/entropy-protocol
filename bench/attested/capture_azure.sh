#!/bin/bash
# Capture raw attestation evidence on one Azure confidential H100 guest (ARCHITECTURE §7d), for the appraisal code
# and its tests: launch (bench/gpu/infra/launch_azure.sh: CVM image, Microsoft's onboarding package pinned by
# SHA-256, the 595 driver; auto-shutdown + a watcher that deletes the resource group at the cap), install
# tpm2-tools and the capture's Python dependencies, run `entropy_protocol.attested.capture` as root, run Microsoft's
# onboarding attestation step (NVIDIA's local GPU verifier) for an independent opinion, fetch everything, check
# every file against the guest's manifest, delete the resource group and confirm it is gone.
# Evidence goes to bench/results/attested/capture-<stamp>/ (tracked); the operator log, which names the operator's
# address, goes to bench/gpu/infra/ (ignored).
# Usage: bash bench/attested/capture_azure.sh [rounds]
set -euo pipefail
ROUNDS=${1:-3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
INFRA="$REPO/bench/gpu/infra"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$REPO/bench/results/attested/capture-$STAMP"
LOG="$INFRA/attested_capture_$STAMP.log"
NAME="att$(date -u +%H%M)"
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15
     -o ServerAliveCountMax=8 -o ConnectTimeout=20)
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
RG=""

cleanup() {
  set +e
  if [ -n "$RG" ]; then
    echo "cleanup: deleting $RG"
    az group delete -n "$RG" --yes </dev/null
    for _ in $(seq 1 30); do
      az group show -n "$RG" >/dev/null 2>&1 </dev/null || { echo "cleanup: $RG is gone"; return; }
      sleep 20
    done
    echo "cleanup: $RG STILL EXISTS - delete it by hand"
  fi
}
trap cleanup EXIT

echo "launch $NAME ($(date -u +%H:%M:%SZ))"
bash "$INFRA/launch_azure.sh" "$NAME" 2 westeurope </dev/null
LINE=$(awk -F'\t' -v n="$NAME" '$1 == n' "$INFRA/instances.tsv" | tail -1)
RG=$(echo "$LINE" | cut -f2 | cut -d/ -f1)
IP=$(echo "$LINE" | cut -f5)
[ -n "$RG" ] && [ -n "$IP" ] || { echo "no ledger line for $NAME"; exit 1; }
echo "machine up: $RG ($(date -u +%H:%M:%SZ))"

echo "tools"
"${SSH[@]}" "ubuntu@$IP" 'sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q tpm2-tools python3-venv >/dev/null && \
  python3 -m venv /home/ubuntu/cap && /home/ubuntu/cap/bin/pip -q install "cbor2>=5.6" "cryptography>=43.0.3" \
  "nvidia-ml-py==12.550.52" && /home/ubuntu/cap/bin/pip freeze' </dev/null > "$OUT/guest_pip_freeze.txt"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$REPO" -czf - src | "${SSH[@]}" "ubuntu@$IP" 'mkdir -p /home/ubuntu/ep && tar -C /home/ubuntu/ep -xzf -'

echo "capture ($ROUNDS rounds)"
"${SSH[@]}" "ubuntu@$IP" "sudo env PYTHONPATH=/home/ubuntu/ep/src /home/ubuntu/cap/bin/python -m \
  entropy_protocol.attested.capture --out /root/cap --rounds $ROUNDS" </dev/null || echo "capture rc=$? (see MANIFEST errors)"

echo "NVIDIA's local GPU verifier through Microsoft's onboarding step 2 (independent opinion)"
"${SSH[@]}" "ubuntu@$IP" 'cd cgpu-onboarding-package && ls && timeout 900 sudo bash step-2-attestation.sh 2>&1 | tail -120' \
  </dev/null > "$OUT/onboarding_step2_attestation.txt" 2>&1 || echo "step 2 rc=$?"

echo "fetch"
"${SSH[@]}" "ubuntu@$IP" 'sudo tar -C /root/cap -czf - .' </dev/null | tar -C "$OUT" -xzf -
/usr/local/bin/python3 -I - "$OUT" <<'EOF'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1]); m = json.loads((out / "MANIFEST.json").read_text())
bad = [n for n, h in m["files"].items() if not n.endswith("_seconds")
       and hashlib.sha256((out / n).read_bytes()).hexdigest() != h]
print(f"fetched {len(m['files'])} entries; mismatches: {bad or 'none'}; errors on the guest: {m['errors'] or 'none'}")
sys.exit(1 if bad else 0)
EOF
echo "done ($(date -u +%H:%M:%SZ)): $OUT"
