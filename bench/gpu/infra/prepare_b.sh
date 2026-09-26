#!/bin/bash
# Prepare a phase B machine while the shared corpus is still being sealed elsewhere: wait for SSH, bootstrap,
# push the code, fetch the model's weights ahead of time. Usage: bash prepare_b.sh <name> <bootstrap script>
set -uo pipefail
cd "$(dirname "$0")"
NAME=${1:?name}; BOOT=${2:?bootstrap script}
IP=$(awk -F'\t' -v n="$NAME" '$1 == n {ip = $5} END {print ip}' instances.tsv)
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -o BatchMode=yes "ubuntu@$IP")
stamp() { date -u +%H:%M:%SZ; }
end=$(( $(date +%s) + 900 ))
until "${SSH[@]}" true 2>/dev/null; do [ "$(date +%s)" -gt "$end" ] && { echo "$(stamp) $NAME: no ssh in 15 min"; exit 1; }; sleep 10; done
echo "$(stamp) $NAME: ssh up ($IP)"
# a GCP deep-learning image may still be installing its driver on first boot
end=$(( $(date +%s) + 900 ))
until "${SSH[@]}" 'nvidia-smi -L' >/dev/null 2>&1; do [ "$(date +%s)" -gt "$end" ] && { echo "$(stamp) $NAME: no driver in 15 min"; exit 1; }; sleep 15; done
"${SSH[@]}" 'nvidia-smi -L'
"${SSH[@]}" 'bash -s' < "$BOOT" 2>&1 | tail -3
bash sync.sh up "$NAME"
"${SSH[@]}" 'source ~/ep.env && python3 -c "from huggingface_hub import snapshot_download as s; s(\"Qwen/Qwen2.5-7B-Instruct\")" >/dev/null 2>&1 && du -sh $HF_HOME' 2>&1 | tail -1
echo "$(stamp) $NAME: ready for phase B"
