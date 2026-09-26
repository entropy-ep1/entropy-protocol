#!/bin/bash
# End a machine's run: wait until its log prints the given marker (or the machine is gone, or the deadline passes),
# download the results, check that every remote result file arrived with its size, and only then terminate the machine.
# (15.09.2026: replaces phase_a_watch.sh for runs with extra steps; the size check follows the rule "destructive
# steps only after the copy is verified" - a download that succeeds but is incomplete must not end the machine.)
# Usage: bash finish_when.sh <name> <remote log> <marker regex> [deadline, e.g. 17:05Z]
set -u
cd "$(dirname "$0")"
NAME=${1:?name}; LOG=${2:?remote log}; MARK=${3:?marker}; DEADLINE=${4:-}
stamp() { date -u "+%H:%M:%SZ"; }
IP=$(awk -F'\t' -v n="$NAME" '$1 == n {ip = $5} END {print ip}' instances.tsv)
ID=$(awk -F'\t' -v n="$NAME" '$1 == n {id = $2} END {print id}' instances.tsv)
[ -n "$IP" ] || { echo "no $NAME in the ledger"; exit 1; }
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
CLOUD=$(awk -F'\t' -v n="$NAME" '$1 == n {c = $8} END {print (c == "" ? "aws" : c)}' instances.tsv)
state() {                            # "running" while the machine is up, anything else when it is gone
  case $CLOUD in
    aws)   aws --region us-east-2 ec2 describe-instances --instance-ids "$ID" \
             --query "Reservations[0].Instances[0].State.Name" --output text 2>/dev/null;;
    gcp)   IFS=/ read -r P Z V <<< "$ID"
           gcloud compute instances describe "$V" --project "$P" --zone "$Z" --format "value(status)" 2>/dev/null \
             | sed 's/RUNNING/running/';;
    azure) IFS=/ read -r G V <<< "$ID"
           az vm get-instance-view -g "$G" -n "$V" --query "instanceView.statuses[?starts_with(code,'PowerState')].displayStatus | [0]" \
             -o tsv 2>/dev/null | sed 's/VM running/running/';;
  esac; }
dl=0; [ -n "$DEADLINE" ] && dl=$(date -u -j -f "%H:%MZ" "$DEADLINE" +%s 2>/dev/null || echo 0)
while true; do
  st=$(state)
  [ "$st" = "running" ] || { echo "$(stamp) machine $ID is $st: nothing to finish"; exit 2; }
  if "${SSH[@]}" "grep -qE '$MARK' $LOG" 2>/dev/null; then echo "$(stamp) marker seen in $LOG"; break; fi
  if [ "$dl" -gt 0 ] && [ "$(date -u +%s)" -ge "$dl" ]; then echo "$(stamp) deadline $DEADLINE reached"; break; fi
  sleep 60
done
bash sync.sh down "$NAME" || { echo "$(stamp) download failed: machine left running (its cap still applies)"; exit 3; }
LOCAL=../../../runs_gpu/$NAME
missing=$("${SSH[@]}" 'cd ep && find runs logs -type f -printf "%p\t%s\n"' 2>/dev/null | while IFS=$'\t' read -r p s; do
  if [ "${p#runs/}" != "$p" ]; then l="$LOCAL/${p#runs/}"; else l="$LOCAL/logs/${p#logs/}"; fi
  [ -f "$l" ] && [ "$(stat -f %z "$l")" = "$s" ] || echo "$p"
done)
if [ -n "$missing" ]; then
  echo "$(stamp) NOT terminating: $(echo "$missing" | wc -l | tr -d ' ') remote files missing or different locally:"
  echo "$missing" | head -20
  exit 4
fi
echo "$(stamp) every remote file is here ($(du -sh "$LOCAL" | cut -f1)); terminating $NAME"
bash terminate.sh "$NAME" && echo "$(stamp) $NAME terminated"
