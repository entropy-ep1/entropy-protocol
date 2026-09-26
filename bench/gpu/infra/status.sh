#!/bin/bash
# Evaluation machines that are running now, their age against the cap, and the money spent so far (estimate:
# hours x the current spot price, or x the on-demand price table below).
# Usage: bash status.sh [region]
set -euo pipefail
REGION=${1:-us-east-2}
od_price() {                        # on-demand $/h, us-east-2 (bash 3.2 has no associative arrays)
  case $1 in p5.4xlarge) echo 6.88;; g6e.xlarge) echo 1.861;; g6e.2xlarge) echo 2.242;; g6e.4xlarge) echo 3.004;;
    g5.xlarge) echo 1.006;; g6.xlarge) echo 0.805;; g6e.12xlarge) echo 10.493;; *) echo 0;; esac
}
aws --region "$REGION" ec2 describe-instances \
  --filters Name=tag:Project,Values=entropy-eval Name=instance-state-name,Values=pending,running,stopping \
  --query "Reservations[].Instances[].[Tags[?Key=='Name']|[0].Value,InstanceId,InstanceType,InstanceLifecycle,LaunchTime,Tags[?Key=='CapHours']|[0].Value,PublicIpAddress]" \
  --output text | while read -r NAME ID TYPE LIFE LAUNCH CAP IP; do
    AGE=$(python3 -c "import datetime as d; t=d.datetime.fromisoformat('$LAUNCH'.replace('Z','+00:00')); print(round((d.datetime.now(d.timezone.utc)-t).total_seconds()/3600,2))")
    if [ "$LIFE" = "spot" ]; then
      PRICE=$(aws --region "$REGION" ec2 describe-spot-price-history --instance-types "$TYPE" --product-descriptions Linux/UNIX \
              --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" --query "max(SpotPriceHistory[].SpotPrice)" --output text)
    else
      PRICE=$(od_price "$TYPE")
    fi
    COST=$(python3 -c "print(round($AGE*$PRICE,2))")
    echo -e "$NAME\t$ID\t$TYPE\t$([ "$LIFE" = spot ] && echo spot || echo ondemand)\t${AGE}h of ${CAP}h\t\$$PRICE/h\t~\$$COST\t$IP"
  done
