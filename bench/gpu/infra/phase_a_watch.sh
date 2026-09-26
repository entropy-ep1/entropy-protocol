#!/bin/bash
# Watch phase A on machine "a1" (ledger instances.tsv): results down to this Mac every 20 min; when the run prints
# "PHASE A DONE", a last download, then the machine is terminated (it would also terminate itself at its cap). Stops
# too if the machine is gone (spot interruption, cap) - whatever was downloaded stays in runs_gpu/a1/.
# Usage: bash phase_a_watch.sh
set -u
cd "$(dirname "$0")"
stamp() { date "+%H:%M:%S"; }
IP=$(awk -F'\t' '$1 == "a1" {ip = $5} END {print ip}' instances.tsv)
ID=$(awk -F'\t' '$1 == "a1" {id = $2} END {print id}' instances.tsv)
[ -n "$IP" ] || { echo "no a1 in the ledger"; exit 1; }
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
state() { aws --region us-east-2 ec2 describe-instances --instance-ids "$ID" \
          --query "Reservations[0].Instances[0].State.Name" --output text 2>/dev/null; }
while true; do
  st=$(state)
  if [ "$st" != "running" ]; then echo "$(stamp) machine $ID is $st - stopping the watch"; exit 2; fi
  bash sync.sh down a1 > /dev/null 2>&1 && echo "$(stamp) results down ($(du -sh ../../../runs_gpu/a1 2>/dev/null | cut -f1))"
  last=$("${SSH[@]}" 'tail -2 ep/logs/phase_a.log' 2>/dev/null)
  echo "$(stamp) $last" | tail -1
  if echo "$last" | grep -q "PHASE A DONE"; then
    bash sync.sh down a1 && bash terminate.sh a1 && echo "$(stamp) PHASE A DONE: results down, machine terminated"
    exit 0
  fi
  sleep 1200
done
