#!/bin/bash
# A long run's safety net: pull the machine's results every 30 minutes, so the hard cap that terminates it can never
# take more than half an hour of results with it. Stops after three failed pulls in a row (the machine is gone).
# Usage: bash sync_every.sh <name> [hours]
set -u
cd "$(dirname "$0")"
NAME=${1:?name}; HOURS=${2:-40}; fails=0
for i in $(seq 1 $((HOURS * 2))); do
  sleep 1800
  if bash sync.sh down "$NAME" > /dev/null 2>&1; then
    fails=0; echo "$(date -u +%H:%MZ) pulled $NAME ($(du -sh "../../../runs_gpu/$NAME" | cut -f1))"
  else
    fails=$((fails + 1)); echo "$(date -u +%H:%MZ) pull failed ($fails)"
    [ "$fails" -ge 3 ] && { echo "three failures: $NAME is gone"; exit 0; }
  fi
done
