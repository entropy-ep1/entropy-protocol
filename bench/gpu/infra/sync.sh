#!/bin/bash
# Code up / results down between this Mac and an evaluation machine (by ledger name).
# Usage: bash sync.sh up|down <name>   ·   bash sync.sh runs-up <name> <path> [FROM=<machine>]
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$DIR/../../.." && pwd)
MODE=${1:?up|down|runs-up}; NAME=${2:?name}; WHAT=${3:-}
IP=$(awk -F'\t' -v n="$NAME" '$1 == n {ip = $5} END {print ip}' "$DIR/instances.tsv")
[ -n "$IP" ] || { echo "no machine named $NAME in the ledger"; exit 1; }
SSH="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new"
if [ "$MODE" = up ]; then
  rsync -az -e "$SSH" --exclude runs --exclude data --exclude __pycache__ --exclude .pytest_cache \
        --exclude 'bench/results' --exclude 'bench/logs' --exclude runs_gpu "$ROOT/" "ubuntu@$IP:ep/"
  echo "code -> $NAME ($IP)"
elif [ "$MODE" = runs-up ]; then
  # one result file (or directory) from this Mac to the machine, e.g. the shared calibration corpus of phase B:
  # bash sync.sh runs-up b2 Qwen_Qwen2.5-7B-Instruct/B_cal.jsonl   (paths are relative to runs_gpu/<from>/ and ep/runs/)
  FROM=${FROM:-a1}; [ -n "$WHAT" ] || { echo "usage: sync.sh runs-up <name> <path under runs_gpu/\$FROM>"; exit 1; }
  SRC="$ROOT/runs_gpu/$FROM/$WHAT"
  [ -e "$SRC" ] || { echo "no $SRC"; exit 1; }
  # shellcheck disable=SC2086 - $SSH carries its options and must split into words here
  $SSH "ubuntu@$IP" "mkdir -p ep/runs/$(dirname "$WHAT")"
  rsync -az -e "$SSH" "$SRC" "ubuntu@$IP:ep/runs/$WHAT"
  echo "$WHAT -> $NAME ($IP)"
else
  mkdir -p "$ROOT/runs_gpu/$NAME"
  rsync -az -e "$SSH" "ubuntu@$IP:ep/runs/" "$ROOT/runs_gpu/$NAME/"
  rsync -az -e "$SSH" "ubuntu@$IP:ep/logs/" "$ROOT/runs_gpu/$NAME/logs/" 2>/dev/null || true
  echo "results <- $NAME ($IP): $(du -sh "$ROOT/runs_gpu/$NAME" | cut -f1)"
fi
