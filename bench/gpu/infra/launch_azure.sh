#!/bin/bash
# Launch the evaluation program's Hopper machine: one Azure Standard_NCC40ads_H100_v5 (a Confidential VM - AMD
# SEV-SNP guest with an H100 NVL 94 GB in confidential-computing mode) in West Europe, brought to a working CUDA
# driver the way Microsoft's onboarding package does it. The attestation step is skipped: the evaluation only needs
# CUDA. The procedure (image, package pinned by SHA-256, the 595 driver pin) is the one proven on this subscription
# on 16.09.2026; nothing of that run's resource groups, keys or network rules is used here.
#
# The cap needs two locks, because on Azure a shutdown from inside the guest does NOT stop the bill (the VM stays
# allocated): (1) Azure auto-shutdown deallocates the VM at the cap time; (2) azure_cap.sh, started here in the
# background under caffeinate, deletes the whole resource group at the cap unless it is already gone.
# Usage: bash launch_azure.sh <name> <hours cap> [location]
set -euo pipefail
NAME=${1:?name}; HOURS=${2:?hours cap}; LOC=${3:-westeurope}
DIR=$(cd "$(dirname "$0")" && pwd); LEDGER="$DIR/instances.tsv"
STAMP=$(date -u +%Y%m%dt%H%M%Sz)
RG="ep-eval-$NAME-$STAMP"; VM="ep-eval-$NAME"; ADMIN=ubuntu
SIZE=Standard_NCC40ads_H100_v5
IMAGE="Canonical:ubuntu-24_04-lts:cvm:24.04.202607310"      # the version Microsoft's onboarding V4.4.1 pins
PKG_URL="https://github.com/Azure/az-cgpu-onboarding/releases/download/V4.4.1/cgpu-onboarding-package.tar.gz"
PKG_SHA="455ac1c1318d857c36dbae3432e023aebcb983dbcbd0ab1c3973af3a43eb480e"
KEY="$HOME/.ssh/id_ed25519"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15
     -o ServerAliveCountMax=8 -o ConnectTimeout=20)

PKG=$(mktemp -d)/cgpu-onboarding-package.tar.gz
curl -fsSL "$PKG_URL" -o "$PKG"
echo "$PKG_SHA  $PKG" | shasum -a 256 -c - >/dev/null

MYIP=$(curl -s https://api.ipify.org)
CAP_AT=$(date -u -v+"${HOURS}"H +%H%M)                       # HHMM UTC, the auto-shutdown clock time
az group create -n "$RG" -l "$LOC" --tags Project=entropy-eval CapHours="$HOURS" >/dev/null
# the second lock starts before the VM exists, so no path leaves a machine without it
nohup caffeinate -i bash "$DIR/azure_cap.sh" "$RG" "$HOURS" > "$DIR/azure_cap_$RG.log" 2>&1 &
echo "cap watcher pid $! (deletes $RG in ${HOURS}h unless gone)"

IP=$(az vm create -g "$RG" -n "$VM" -l "$LOC" --image "$IMAGE" --size "$SIZE" \
  --security-type ConfidentialVM --os-disk-security-encryption-type DiskWithVMGuestState \
  --enable-secure-boot true --enable-vtpm true --os-disk-size-gb 200 \
  --public-ip-sku Standard --admin-username "$ADMIN" --ssh-key-values "@$KEY.pub" \
  --nsg-rule SSH --tags Project=entropy-eval --query publicIpAddress -o tsv)
NSG=$(az network nsg list -g "$RG" --query "[0].name" -o tsv)
RULE=$(az network nsg rule list -g "$RG" --nsg-name "$NSG" --query "[?destinationPortRange=='22'].name" -o tsv | head -1)
az network nsg rule update -g "$RG" --nsg-name "$NSG" -n "$RULE" --source-address-prefixes "$MYIP/32" >/dev/null
az vm auto-shutdown -g "$RG" -n "$VM" --time "$CAP_AT" >/dev/null
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$RG/$VM" "$SIZE" ondemand "$IP" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HOURS" azure >> "$LEDGER"
echo "vm $VM at $IP in $RG; ssh from $MYIP only; auto-shutdown (deallocate) at ${CAP_AT}Z"

wait_ssh() { local end=$(( $(date +%s) + $1 * 60 )); while [ "$(date +%s)" -lt "$end" ]; do
  "${SSH[@]}" -o BatchMode=yes "$ADMIN@$IP" true >/dev/null 2>&1 && return 0; sleep 10; done; return 1; }
wait_ssh 10 || { echo "no ssh after 10 min"; exit 1; }
scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -q "$PKG" "$ADMIN@$IP:"
"${SSH[@]}" "$ADMIN@$IP" 'tar -xzf cgpu-onboarding-package.tar.gz'

echo "step 0: kernel, then reboot"
BOOTED=$("${SSH[@]}" "$ADMIN@$IP" 'uptime -s')
"${SSH[@]}" "$ADMIN@$IP" 'cd cgpu-onboarding-package && sudo bash step-0-prepare-kernel.sh 2>&1 | tail -3; sudo systemctl reboot' || true
for _ in $(seq 1 60); do sleep 10
  NOW=$("${SSH[@]}" -o BatchMode=yes "$ADMIN@$IP" 'uptime -s' 2>/dev/null || true)
  [ -n "$NOW" ] && [ "$NOW" != "$BOOTED" ] && break; done
echo "step 1: the NVIDIA open driver, userspace pinned to the version the signed kernel modules require"
# Ubuntu's signed 595 server modules are built against one userspace version; noble-updates carries a newer one
# that apt would pick and fail on (16.09.2026). The X driver is a strict "=" dependency, pinned by name too.
"${SSH[@]}" "$ADMIN@$IP" 'MOD=$(apt-cache show linux-modules-nvidia-595-server-open-$(uname -r) 2>/dev/null | sed -n "s/.*nvidia-kernel-common-595-server (<= \([^)]*\)).*/\1/p" | head -1); \
  V=$(apt-cache madison nvidia-kernel-common-595-server | awk "{print \$3}" | sort -V | while read x; do dpkg --compare-versions "$x" le "${MOD:-999}" && echo "$x"; done | tail -1); \
  echo "pinning 595 userspace ${V:-?}"; \
  [ -n "$V" ] && printf "Package: nvidia-*-595-server* libnvidia-*-595-server* xserver-xorg-video-nvidia-595-server nvidia-kernel-common-595-server nvidia-kernel-source-595-server-open nvidia-driver-595-server-open nvidia-utils-595-server nvidia-compute-utils-595-server nvidia-firmware-595-server*\nPin: version %s\nPin-Priority: 1001\n" "$V" | sudo tee /etc/apt/preferences.d/nvidia-595-userspace.pref >/dev/null'
"${SSH[@]}" "$ADMIN@$IP" 'cd cgpu-onboarding-package && sudo bash step-1-install-gpu-driver.sh 2>&1 | tail -8'
wait_ssh 5 || true
"${SSH[@]}" "$ADMIN@$IP" 'nvidia-smi -L && nvidia-smi conf-compute -q' || { echo "no working NVIDIA driver after step 1"; exit 1; }
echo "ready: ssh -i $KEY -o StrictHostKeyChecking=accept-new $ADMIN@$IP   (next: bootstrap_pip.sh)"
