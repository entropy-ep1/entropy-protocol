#!/bin/bash
# Phase A as soon as the account's GPU limits allow it. 14.09.2026 ~08:25, right after Serhii reported the four quotas
# approved and said "да запускай" (g6e.2xlarge, Ohio, spot or else on-demand, cap 4 h): spot was refused with
# MaxSpotInstanceCountExceeded and on-demand with VcpuLimitExceeded ("current vCPU limit of 0"), i.e. the new limits
# were not in effect in us-east-2 yet. 09:00-11:02: 9 tries, spot later refused for capacity (InsufficientInstanceCapacity),
# on-demand still at a limit of 0; 11:04 spot refused for capacity in all three zones. Serhii approved g6e.xlarge
# too (same GPU). Tries spot for both types in each Ohio zone, then on-demand, every 15 min for up to 4 h (a refused launch creates
# and costs nothing); once a machine runs: bootstrap, code up, phase A started on the machine in the background.
# Usage: bash phase_a_when_ready.sh
set -u
cd "$(dirname "$0")"
stamp() { date "+%H:%M:%S"; }
running() { aws --region us-east-2 ec2 describe-instances --filters Name=tag:Project,Values=entropy-eval \
            Name=instance-state-name,Values=pending,running --query "Reservations[].Instances[].InstanceId" --output text; }
# approved by Serhii on 14.09, spot, cap 4 h: g6e.2xlarge and g6e.xlarge (L40S 48 GB, the plan's GPU), then p5.4xlarge
# (H100 80 GB), g5.2xlarge (A10G 24 GB), g6.2xlarge (L4 24 GB; phase A skips its fp32 steps on 24 GB cards);
# on-demand only g6e.2xlarge ($2.24/h, approved as the fallback of the first launch)
SPOT_TYPES=${SPOT_TYPES:-"g6e.2xlarge g6e.xlarge p5.4xlarge g5.2xlarge g6.2xlarge"}
ONDEMAND_TYPES=${ONDEMAND_TYPES:-"g6e.2xlarge"}
launched=0
TRIES=${TRIES:-16}                            # every 15 min: 16 = 4 h, 48 = 12 h
for try in $(seq 1 "$TRIES"); do
  if [ -n "$(running)" ]; then        # never a second machine (e.g. after a launch whose wait timed out)
    grep -q "^a1	" instances.tsv 2>/dev/null && { launched=1; break; }
    echo "$(stamp) an entropy-eval machine exists but is not in the ledger: $(running) - stopping here"; exit 1
  fi
  for type in $SPOT_TYPES; do                 # spot capacity is per zone; cheapest zones first (14.09 prices)
    for az in us-east-2a us-east-2c us-east-2b; do
      [ -n "$(running)" ] && break 2
      bash launch.sh a1 "$type" 4 spot us-east-2 "$az" 2>&1 && { launched=1; break 2; }
    done
  done
  [ "$launched" = 1 ] && break
  for type in $ONDEMAND_TYPES; do             # on-demand (15.09: G on-demand limit 8 vCPU = one g6e.2xlarge)
    for az in us-east-2a us-east-2c us-east-2b; do
      [ -n "$(running)" ] && break 2
      bash launch.sh a1 "$type" 4 ondemand us-east-2 "$az" 2>&1 && { launched=1; break 2; }
    done
  done
  [ "$launched" = 1 ] && break
  echo "$(stamp) try $try: no spot capacity for $SPOT_TYPES, no on-demand limit for $ONDEMAND_TYPES; next in 15 min"
  [ "$try" -lt "$TRIES" ] && sleep 900
done
[ "$launched" = 1 ] || { echo "$(stamp) NOT LAUNCHED after $TRIES tries: no spot capacity in any Ohio zone and no on-demand limit"; exit 1; }
IP=$(awk -F'\t' '$1 == "a1" {ip = $5} END {print ip}' instances.tsv)
SSH=(ssh -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
for i in $(seq 60); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
echo "$(stamp) ssh up on $IP"
"${SSH[@]}" 'bash -s' < bootstrap.sh 2>&1 | tail -4
bash sync.sh up a1
"${SSH[@]}" 'cd ep && mkdir -p logs && (nohup bash bench/gpu/run_phase_a.sh > logs/phase_a.log 2>&1 &)'
sleep 60
"${SSH[@]}" 'tail -4 ep/logs/phase_a.log'
echo "$(stamp) PHASE A STARTED on $IP"
# results down every 20 min; at "PHASE A DONE" a last download and the machine is terminated
exec bash phase_a_watch.sh
