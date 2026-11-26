"""Create a new-version plan while preserving the scientific settings of an old plan."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/collector_v3/kit"))
from distributed_live import sources, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("old_plan", type=Path)
    p.add_argument("new_directory", type=Path)
    a = p.parse_args()
    raw = a.old_plan.read_bytes()
    recorded = a.old_plan.with_name("plan.sha256").read_text().strip()
    if sha256(raw).hexdigest() != recorded:
        raise ValueError("Original plan hash mismatch")
    new = json.loads(raw)
    new.update(
        source_hashes=sources(ROOT),
        commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        created=time.time(),
        parent_plan_sha256=recorded,
        parent_software="v0.6.0",
        control_protocol="0.6.1-pool2-total-deadline-sqlite-totals",
    )
    a.new_directory.mkdir(parents=True, exist_ok=False)
    dump(a.new_directory / "plan.json", new)
    (a.new_directory / "plan.sha256").write_text(
        sha256((a.new_directory / "plan.json").read_bytes()).hexdigest() + "\n"
    )
    print(a.new_directory / "plan.json")


if __name__ == "__main__":
    main()
