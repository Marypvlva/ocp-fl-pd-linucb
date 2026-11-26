"""Run paired training experiments under several localhost pacing profiles."""

from .paths import ROOT
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import numpy as np

from . import tcp_closed_loop_pilot as experiment
from .tcp_network import PROFILES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tcp_network_pilot.json")
    parser.add_argument("--profiles", choices=tuple(PROFILES), nargs="+", default=list(PROFILES))
    parser.add_argument("--output-root", type=Path)
    args, forwarded = parser.parse_known_args(argv)
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("Duplicate profiles")
    settings = json.loads(args.config.read_text())
    order_seed = settings.get("seed", 2026) + 30_000
    profiles = [str(p) for p in np.random.default_rng(order_seed).permutation(args.profiles)]
    root = args.output_root or ROOT / "results" / (
        "tcp_network_sweep_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    root.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "running",
        "profiles": profiles,
        "profile_order_seed": order_seed,
        "completed": [],
    }
    summaries = []
    try:
        for profile in profiles:
            folder = experiment.main(
                [
                    "--config",
                    str(args.config),
                    *forwarded,
                    "--network-profile",
                    profile,
                    "--output",
                    str(root / profile),
                ]
            )
            with (folder / "summary.csv").open() as stream:
                summaries.extend(csv.DictReader(stream))
            status["completed"].append(profile)
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        experiment.append_csv(root / "summary.csv", summaries)
        (root / "sweep.json").write_text(json.dumps(status, indent=2))
    print(f"Sweep saved: {root}", flush=True)
    return root


if __name__ == "__main__":
    main()
