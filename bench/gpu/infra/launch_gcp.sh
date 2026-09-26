#!/bin/bash
# Launch one GCP GPU machine of the evaluation program with a hard cap the platform itself enforces:
# --max-run-duration with --instance-termination-action=DELETE deletes the instance and its boot disk when the cap
# runs out, whether or not this Mac is awake.
# Usage: bash launch_gcp.sh <name> <machine type> <accelerator> <hours cap> [zone] [project]
#   e.g. bash launch_gcp.sh b-l4 g2-standard-8 nvidia-l4 8 us-central1-a rats-probe
# SSH: this Mac's key as INSTANCE metadata (user ubuntu). Never touches project metadata, firewall rules or other
# instances: the project's default network already admits SSH, and the key is the only way in.
# The project allows one GPU across all regions (GPUS_ALL_REGIONS = 1): one GCP machine at a time.
set -euo pipefail
NAME=${1:?name}; TYPE=${2:?machine type}; ACC=${3:?accelerator}; HOURS=${4:?hours cap}
ZONE=${5:-us-central1-a}; PROJECT=${6:-rats-probe}
DIR=$(cd "$(dirname "$0")" && pwd); LEDGER="$DIR/instances.tsv"
VM="ep-eval-$NAME"
PUB=$(cat "$HOME/.ssh/id_ed25519.pub")
IP=$(gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" --machine-type "$TYPE" \
  --accelerator "type=$ACC,count=1" --maintenance-policy TERMINATE --provisioning-model STANDARD \
  --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
  --boot-disk-size 200GB --boot-disk-type pd-balanced \
  --metadata "ssh-keys=ubuntu:$PUB,install-nvidia-driver=True" \
  --labels "project=entropy-eval,caphours=$HOURS" \
  --max-run-duration "${HOURS}h" --instance-termination-action DELETE \
  --format "value(networkInterfaces[0].accessConfigs[0].natIP)")
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$PROJECT/$ZONE/$VM" "$TYPE+$ACC" ondemand "$IP" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOURS" gcp >> "$LEDGER"
echo "launched $NAME $VM ($TYPE, $ACC) in $ZONE ip $IP, cap ${HOURS}h enforced by GCP (delete on expiry)"
echo "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new ubuntu@$IP"
