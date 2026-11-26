#!/usr/bin/env bash
# Operator-side helper (your laptop). Requires terraform, jq, rsync, ssh-agent with your key.
#   bash deploy/yandex/stand.sh up        # create VMs, write inventory.json + placement.json
#   bash deploy/yandex/stand.sh push      # copy the project to the coordinator
#   bash deploy/yandex/stand.sh ssh       # ssh -A to the coordinator
#   bash deploy/yandex/stand.sh fetch DIR # download a study folder from the coordinator
#   bash deploy/yandex/stand.sh down      # destroy everything
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$HERE"
coordinator() { jq -r .coordinator_public_ip "$HERE/inventory.json"; }
case "${1:-}" in
  up)
    terraform init
    terraform apply
    terraform output -json > tf_output.json
    python3 inventory.py tf_output.json --roles "${OCP_WORKER_ROLES:-16}" --out "$HERE"
    ;;
  push)
    rsync -az --delete --exclude .venv --exclude results --exclude 'study_*' --exclude .git \
          --exclude 'deploy/yandex/.terraform*' --exclude 'deploy/yandex/terraform.tfstate*' \
          -e "ssh -o StrictHostKeyChecking=accept-new" "$ROOT/" "ubuntu@$(coordinator):~/adaptive_planning/"
    ;;
  ssh)
    exec ssh -A -o StrictHostKeyChecking=accept-new "ubuntu@$(coordinator)"
    ;;
  fetch)
    mkdir -p "$ROOT/results/cloud"
    rsync -az "ubuntu@$(coordinator):~/adaptive_planning/${2:?study folder}" "$ROOT/results/cloud/"
    ;;
  down)
    terraform destroy
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1;;
esac
