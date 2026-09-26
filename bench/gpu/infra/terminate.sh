#!/bin/bash
# Terminate one evaluation machine by ledger name (after its results were synced down). Only our own resources are
# accepted: AWS instances tagged Project=entropy-eval, GCP instances labelled project=entropy-eval, Azure resource
# groups tagged Project=entropy-eval (the whole group goes: VM, disk, NIC, IP, NSG).
# Usage: bash terminate.sh <name> [aws region]
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd); NAME=${1:?name}; REGION=${2:-us-east-2}
ROW=$(awk -F'\t' -v n="$NAME" '$1 == n {r = $0} END {print r}' "$DIR/instances.tsv")
[ -n "$ROW" ] || { echo "no machine named $NAME in the ledger"; exit 1; }
ID=$(echo "$ROW" | cut -f2); CLOUD=$(echo "$ROW" | cut -f8); CLOUD=${CLOUD:-aws}
case $CLOUD in
  aws)
    TAG=$(aws --region "$REGION" ec2 describe-instances --instance-ids "$ID" \
          --query "Reservations[0].Instances[0].Tags[?Key=='Project']|[0].Value" --output text)
    [ "$TAG" = "entropy-eval" ] || { echo "refusing: $ID is not an entropy-eval instance"; exit 1; }
    aws --region "$REGION" ec2 terminate-instances --instance-ids "$ID" \
      --query "TerminatingInstances[0].CurrentState.Name" --output text;;
  gcp)
    IFS=/ read -r P Z V <<< "$ID"
    L=$(gcloud compute instances describe "$V" --project "$P" --zone "$Z" --format "value(labels.project)" 2>/dev/null || true)
    [ "$L" = "entropy-eval" ] || { echo "refusing: $V is not an entropy-eval instance (or already gone)"; exit 1; }
    gcloud compute instances delete "$V" --project "$P" --zone "$Z" --quiet;;
  azure)
    IFS=/ read -r G V <<< "$ID"
    T=$(az group show -n "$G" --query "tags.Project" -o tsv 2>/dev/null || true)
    [ "$T" = "entropy-eval" ] || { echo "refusing: $G is not an entropy-eval group (or already gone)"; exit 1; }
    az group delete -n "$G" --yes && echo "deleted $G";;
  *) echo "unknown cloud $CLOUD"; exit 1;;
esac
