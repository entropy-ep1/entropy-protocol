#!/bin/bash
# Launch one GPU machine of the evaluation program with a hard time cap (the machine terminates itself).
# Usage: bash launch.sh <name> <instance type> <hours cap> [spot|ondemand] [region] [availability zone]
#   e.g. bash launch.sh a1 g6e.2xlarge 6 spot us-east-2 us-east-2c   (spot capacity is per zone)
# Creates on first use (free): key pair "ep-eval" from ~/.ssh/id_ed25519.pub, security group "ep-eval" (SSH from
# this Mac's current public IP only). Never touches other key pairs, groups or instances.
set -euo pipefail
NAME=${1:?name}; TYPE=${2:?instance type}; HOURS=${3:?hours cap}; MARKET=${4:-spot}; REGION=${5:-us-east-2}; AZ=${6:-}
DIR=$(cd "$(dirname "$0")" && pwd); LEDGER="$DIR/instances.tsv"
AWS=(aws --region "$REGION")

if ! "${AWS[@]}" ec2 describe-key-pairs --key-names ep-eval >/dev/null 2>&1; then
  "${AWS[@]}" ec2 import-key-pair --key-name ep-eval --public-key-material "fileb://$HOME/.ssh/id_ed25519.pub" >/dev/null
  echo "key pair ep-eval imported"
fi
VPC=$("${AWS[@]}" ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text)
SG=$("${AWS[@]}" ec2 describe-security-groups --filters Name=group-name,Values=ep-eval Name=vpc-id,Values="$VPC" \
     --query "SecurityGroups[0].GroupId" --output text)
if [ "$SG" = "None" ]; then
  SG=$("${AWS[@]}" ec2 create-security-group --group-name ep-eval --description "Entropy eval: SSH from operator IP" \
       --vpc-id "$VPC" --query GroupId --output text)
  echo "security group ep-eval created: $SG"
fi
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '\n')
"${AWS[@]}" ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true

read -r AMI ROOTDEV < <("${AWS[@]}" ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)*" "Name=state,Values=available" \
  --query "sort_by(Images,&CreationDate)[-1].[ImageId,RootDeviceName]" --output text)

UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
# hard cap: power off after ${HOURS} h; shutdown behaviour is "terminate", so nothing keeps running or billing
shutdown -h +$((HOURS * 60)) "ep-eval time cap"
echo "ep-eval cap ${HOURS}h set at \$(date -u)" > /var/log/ep-eval-cap.log
EOF
MARKET_OPTS=()
[ -n "$AZ" ] && MARKET_OPTS+=(--placement "AvailabilityZone=$AZ")
if [ "$MARKET" = "spot" ]; then
  MARKET_OPTS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
fi
ID=$("${AWS[@]}" ec2 run-instances --instance-type "$TYPE" --image-id "$AMI" --key-name ep-eval --security-group-ids "$SG" \
  --instance-initiated-shutdown-behavior terminate ${MARKET_OPTS[@]+"${MARKET_OPTS[@]}"} \
  --block-device-mappings "DeviceName=$ROOTDEV,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ep-eval-$NAME},{Key=Project,Value=entropy-eval},{Key=CapHours,Value=$HOURS}]" \
  --user-data "file://$UD" --query "Instances[0].InstanceId" --output text)
rm -f "$UD"
"${AWS[@]}" ec2 wait instance-running --instance-ids "$ID"
IP=$("${AWS[@]}" ec2 describe-instances --instance-ids "$ID" --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$ID" "$TYPE" "$MARKET" "$IP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOURS" aws >> "$LEDGER"
echo "launched $NAME $ID $TYPE ($MARKET) ip $IP, cap ${HOURS}h (AMI $AMI)"
echo "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new ubuntu@$IP"
