"""Turn `terraform output -json` into inventory.json and placement.json.

  terraform output -json > tf_output.json
  python inventory.py tf_output.json --roles 9

placement.json maps every role to a private address: worker roles w0..w{N-1}
round-robin over the worker VMs (so neighbouring clients sit in different
zones), e0/e1 on the primary-lane edges, e2/e3 on the backup-lane edges. It also
carries the SSH settings the launcher uses from the coordinator.
"""

import argparse
import json
from pathlib import Path

SSH_OPTIONS = [
    "-i", "~/.ssh/stand",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.ssh/cm-%r@%h:%p",
    "-o", "ControlPersist=30m",
    "-o", "ServerAliveInterval=15",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tf_output")
    parser.add_argument("--roles", type=int, default=9, help="worker roles (live sampler: <= 9)")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    data = json.loads(Path(args.tf_output).read_text())
    inventory = data["inventory"]["value"]
    inventory["coordinator_public_ip"] = data["coordinator_public_ip"]["value"]
    workers, edges = inventory["workers"], inventory["edges"]
    if len(edges) != 4 or not workers:
        raise SystemExit("Inventory needs four edges and at least one worker VM")
    roles = {f"w{i}": workers[i % len(workers)] for i in range(args.roles)}
    roles.update({f"e{i}": edges[i] for i in range(4)})
    placement = dict(
        roles=roles,
        ssh=dict(ssh_user="ubuntu", ssh_options=SSH_OPTIONS, remote_dir="~/adaptive_planning",
                 remote_python=".venv/bin/python"),
        zones=inventory["zones"],
    )
    out = Path(args.out)
    (out / "inventory.json").write_text(json.dumps(inventory, indent=1))
    (out / "placement.json").write_text(json.dumps(placement, indent=1))
    print(f"coordinator {inventory['coordinator_public_ip']}; {len(workers)} worker VMs, 4 edges, "
          f"{args.roles} worker roles")


if __name__ == "__main__":
    main()
