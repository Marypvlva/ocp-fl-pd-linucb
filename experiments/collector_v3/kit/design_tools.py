"""Generate candidate grids and resource estimates, without executing experiments."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from program import validate_scenario


def grid(base, kind):
    result = {}
    values = {
        "budget": [30000000, 27000000, 24000000, 21000000, 18000000, 15000000],
        "horizon": [120, 250, 500, 1000],
        "quotas": [0.05, 0.10, 0.20, 0.40, 1.0],
    }
    if kind not in values:
        raise ValueError("Unknown grid")
    for value in values[kind]:
        scenario = deepcopy(base)
        scenario["id"] = base["id"] + "_" + kind + "_" + str(value)
        if kind == "budget":
            scenario["budget_bytes"] = value
        elif kind == "horizon":
            old = scenario.get("horizon", 120)
            scenario["horizon"] = value
            if "budget_bytes" in scenario:
                scenario["budget_bytes"] = int(scenario["budget_bytes"] * value / old)
            for event in scenario.get("events", []):
                event["start"] = int((event["start"] - 1) * value / old) + 1
                event["end"] = int(event["end"] * value / old)
        else:
            scenario["mirror_fraction"] = value
            scenario["site_fraction"] = value
        validate_scenario(scenario)
        result[str(value)] = scenario
    return result


def costs(seconds_per_round, bytes_per_round, hourly_rates, confirm_blocks):
    if (
        seconds_per_round <= 0
        or bytes_per_round <= 0
        or len(hourly_rates) != 3
        or min(hourly_rates) < 0
    ):
        raise ValueError(
            "Positive measured time/disk and three nonnegative hourly VM tariffs required"
        )
    stages = [
        dict(stage="A_existing_data", scenarios=2, policies=7, blocks=8, horizon=120, new_rounds=0),
        dict(
            stage="B_local_integration", scenarios=1, policies=3, blocks=2, horizon=2, new_rounds=12
        ),
        dict(stage="C0_actions", scenarios=1, policies=1, blocks=2, horizon=40, new_rounds=80),
        dict(
            stage="C1_two_window_pilots",
            scenarios=2,
            policies=3,
            blocks=2,
            horizon=120,
            new_rounds=1440,
        ),
        dict(
            stage="C2_one_proof_service_pilot",
            scenarios=1,
            policies=3,
            blocks=2,
            horizon=120,
            new_rounds=720,
        ),
        dict(
            stage="D_exploratory_ceiling",
            scenarios=2,
            policies=6,
            blocks=8,
            horizon=120,
            new_rounds=11520,
        ),
        dict(
            stage="E_confirmation_ceiling",
            scenarios=1,
            policies=2,
            blocks=confirm_blocks,
            horizon=250,
            new_rounds=500 * confirm_blocks,
        ),
    ]
    for r in stages:
        cloud = r["stage"] not in ("A_existing_data", "B_local_integration")
        hours = r["new_rounds"] * seconds_per_round / 3600
        r.update(
            estimated_active_wall_hours=hours if cloud else 0,
            estimated_result_bytes=r["new_rounds"] * bytes_per_round,
            estimated_active_cloud_cost=hours * sum(hourly_rates) if cloud else 0,
            scope="Measured mean full-cycle input; add episode startup, diagnosis, idle time, backups, safety margin and egress. This is not a probabilistic upper bound.",
        )
    return stages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("grid")
    q.add_argument("base", type=Path)
    q.add_argument("--kind", choices=["budget", "horizon", "quotas"], required=True)
    q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("cost")
    q.add_argument("--seconds-per-round", type=float, required=True)
    q.add_argument("--bytes-per-round", type=int, required=True)
    q.add_argument("--hourly-rates", nargs=3, type=float, required=True)
    q.add_argument("--confirm-blocks", type=int, default=24)
    a = p.parse_args()
    if a.command == "grid":
        a.out.mkdir(parents=True, exist_ok=False)
        for name, value in grid(json.loads(a.base.read_text()), a.kind).items():
            (a.out / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    else:
        if not 3 <= a.confirm_blocks <= 24:
            raise ValueError(
                "Confirmation ceiling is 24 blocks; reconsider scope if power requires more"
            )
        print(
            json.dumps(
                costs(a.seconds_per_round, a.bytes_per_round, a.hourly_rates, a.confirm_blocks),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
