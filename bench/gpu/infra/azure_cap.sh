#!/bin/bash
# The second lock on an Azure evaluation machine (launch_azure.sh starts it under caffeinate): after the cap, delete
# the machine's whole resource group - VM, disk, NIC, IP, NSG - unless it is already gone. Refuses any group that is
# not tagged Project=entropy-eval, so it can never reach another project's resources.
# Usage: bash azure_cap.sh <resource group> <hours>
set -u
RG=${1:?resource group}; HOURS=${2:?hours}
sleep $((HOURS * 3600))
TAG=$(az group show -n "$RG" --query "tags.Project" -o tsv 2>/dev/null) || { echo "$(date -u) $RG already gone"; exit 0; }
[ "$TAG" = "entropy-eval" ] || { echo "$(date -u) refusing: $RG is not tagged Project=entropy-eval"; exit 1; }
echo "$(date -u) cap reached: deleting $RG"
az group delete -n "$RG" --yes --no-wait && echo "$(date -u) delete requested for $RG"
