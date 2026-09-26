#!/bin/bash
# Memory watchdog for the unattended bridge runs on this 16 GB Mac.
# 13.09.2026, 17:25: a GSM8K job on the frozen copy (batch 16; that code scores full [batch x length x vocabulary]
# logits) held a 19 GB footprint with a full 10 GB swap and 15 GB free disk: the kind of state that ended in the
# morning's kernel panic. Every 15 s: if the largest Python process of the bench has a footprint above FOOT_MAX_GB,
# or swap use is above SWAP_MAX_GB, or the data volume has less than DISK_MIN_GB free, stop that process and log why.
# The run's wrapper then records rc=143 for that job and moves on; the machine stays up.
FOOT_MAX_GB=${FOOT_MAX_GB:-13}; SWAP_MAX_GB=${SWAP_MAX_GB:-12}; DISK_MIN_GB=${DISK_MIN_GB:-5}
LOG="$(cd "$(dirname "$0")/.." && pwd)/bench/logs/mem_watchdog.log"
stamp() { date "+%F %T"; }
tomb() { awk '{s=$1; n=s; gsub(/[^0-9.]/, "", n); if (s ~ /G/) n*=1024; else if (s ~ /K/) n/=1024;
               else if (s !~ /M/) n/=1048576; print int(n)}'; }
echo "$(stamp) start (pid $$): footprint > ${FOOT_MAX_GB} GB, swap > ${SWAP_MAX_GB} GB, free disk < ${DISK_MIN_GB} GB" >> "$LOG"
while true; do
  line=$(top -l 1 -o mem -n 15 -stats pid,mem,command 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && $3 ~ /^Python/ {print $1, $2; exit}')
  swap_mb=$(sysctl -n vm.swapusage | awk '{print $6}' | tomb)
  disk_gb=$(df -g /System/Volumes/Data | awk 'NR==2 {print $4}')
  if [ -n "$line" ]; then
    pid=${line%% *}; foot_mb=$(echo "${line##* }" | tomb)
    why=""
    [ "$foot_mb" -gt $((FOOT_MAX_GB * 1024)) ] && why="footprint ${foot_mb} MB"
    [ "$swap_mb" -gt $((SWAP_MAX_GB * 1024)) ] && why="$why swap ${swap_mb} MB"
    [ "$disk_gb" -lt "$DISK_MIN_GB" ] && why="$why free disk ${disk_gb} GB"
    if [ -n "$why" ]; then
      cmd=$(ps -p "$pid" -o command= 2>/dev/null)
      if echo "$cmd" | grep -q "bench/\|pytest"; then
        kill "$pid" && echo "$(stamp) STOPPED pid $pid ($why): ${cmd:0:220}" >> "$LOG"
        sleep 30
      else
        echo "$(stamp) over the limit ($why) but the largest Python (pid $pid) is not a bench job: ${cmd:0:120}" >> "$LOG"
      fi
    fi
  fi
  sleep 15
done
