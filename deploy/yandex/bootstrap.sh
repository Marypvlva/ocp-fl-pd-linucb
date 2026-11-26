#!/usr/bin/env bash
# Run ON THE COORDINATOR from the project root (ssh -A ubuntu@<coordinator_public_ip>):
#   bash deploy/yandex/bootstrap.sh deploy/yandex/inventory.json
# Copies the project to every VM, builds the Python 3.13 environment from the
# lock file, and records clock synchronization (chrony) of every host.
set -euo pipefail
INV=${1:-deploy/yandex/inventory.json}
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ControlMaster=auto
          -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=30m)
mapfile -t HOSTS < <(jq -r '([.coordinator] + .workers + .edges) | unique | .[]' "$INV")
echo "hosts: ${HOSTS[*]}"

echo "== waiting for cloud-init"
for h in "${HOSTS[@]}"; do
  until ssh "${SSH_OPTS[@]}" "ubuntu@$h" test -f /var/lib/cloud/instance/stand-ready; do sleep 5; done
done

echo "== stand-internal SSH key (the study keeps running after you disconnect the agent)"
[ -f ~/.ssh/stand ] || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/stand
for h in "${HOSTS[@]}"; do
  ssh "${SSH_OPTS[@]}" "ubuntu@$h" "grep -qF '$(cat ~/.ssh/stand.pub)' ~/.ssh/authorized_keys || echo '$(cat ~/.ssh/stand.pub)' >> ~/.ssh/authorized_keys"
done

echo "== copying the project"
for h in "${HOSTS[@]}"; do
  [ "$h" = "$(jq -r .coordinator "$INV")" ] && continue
  rsync -az --delete -e "ssh ${SSH_OPTS[*]}" --exclude .venv --exclude results \
        --exclude 'study_*' --exclude .git ./ "ubuntu@$h:~/adaptive_planning/" &
done
wait

echo "== building environments"
for h in "${HOSTS[@]}"; do
  ssh "${SSH_OPTS[@]}" "ubuntu@$h" 'set -e; cd ~/adaptive_planning
    ~/.local/bin/uv venv -q --python 3.13 .venv
    ~/.local/bin/uv pip install -q --python .venv/bin/python -r requirements-lock.txt
    ~/.local/bin/uv pip install -q --python .venv/bin/python --no-deps -e .
    .venv/bin/python -c "import adaptive_planning.node, adaptive_planning.distributed"' &
done
wait

echo "== clocks (deadlines are compared across hosts; offsets must be << rpc timeout)"
for h in "${HOSTS[@]}"; do
  printf '%s ' "$h"
  ssh "${SSH_OPTS[@]}" "ubuntu@$h" "chronyc tracking | sed -n 's/^System time *: //p'"
done | tee deploy/yandex/clock_check.txt
echo "bootstrap done"
