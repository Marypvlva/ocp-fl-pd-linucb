"""Audit complete seed blocks and report paired differences without filling missing runs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_planning.randomized_check import Challenge, encoded_message


def audit(output):
    plan = json.loads((output / "plan.json").read_text())
    if (
        sha256((output / "plan.json").read_bytes()).hexdigest()
        != (output / "plan.sha256").read_text().strip()
    ):
        raise ValueError("Plan digest")
    totals = {m: [] for m in plan["modes"]}
    walls = {m: [] for m in plan["modes"]}
    source_rosters = {}
    initial = None
    missing = []
    diagnostic_errors = []
    for seed in plan["seeds"]:
        for mode in plan["order"][str(seed)]:
            path = output / f"{seed}_{mode}"
            if not (path / "complete.json").exists():
                missing.append(path.name)
                continue
            if (path / "control-log-fetch.json").exists():
                fetched = json.loads((path / "control-log-fetch.json").read_text())
                if fetched["errors"]:
                    diagnostic_errors.append(dict(episode=path.name, errors=fetched["errors"]))
            keys = json.loads((path / "worker-keys.json").read_text())
            rows = [json.loads(p.read_text()) for p in sorted(path.glob("round-*.json"))]
            if len(rows) != plan["horizon"]:
                raise ValueError("Incomplete round log")
            bill = 0
            for rd, row in enumerate(rows, 1):
                if row["round"] != rd:
                    raise ValueError("Round order")
                roster = row["selected"]
                key = (seed, rd)
                if key in source_rosters and source_rosters[key] != roster:
                    raise ValueError("Paired roster differs")
                source_rosters[key] = roster
                if initial is not None and initial != row["initial_model_hash"]:
                    raise ValueError("Initial model differs")
                initial = row["initial_model_hash"]
                bill += row["telemetry"]["app_tx_bytes"]
                ledger = row["byte_ledger"]
                if (
                    bill != ledger["spent"]
                    or ledger["reserved"]
                    or bill > plan["bytes_per_episode"]
                ):
                    raise ValueError("Byte ledger mismatch")
                if not row["accepted"] and row["previous_model_hash"] != row["model_hash"]:
                    raise ValueError("Rejected update modified model")
                if set(row["node_grant_checks"]) != {
                    *[f"w{i}" for i in range(plan["population"])],
                    *[f"e{i}" for i in range(4)],
                }:
                    raise ValueError("Missing lease check")
                valid = set()
                for receipt in row["metrics"].get("collector_receipts", []):
                    ch = Challenge(
                        bytes.fromhex(receipt["session"]),
                        receipt["epoch"],
                        receipt["round"],
                        bytes.fromhex(receipt["candidate_hash"]),
                        receipt["dimension"],
                        tuple(bytes.fromhex(x) for x in receipt["seeds"]),
                    )
                    worker = receipt["worker"]
                    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(keys[str(worker)]))
                    pub.verify(
                        bytes.fromhex(receipt["signature"]),
                        encoded_message(ch, worker, receipt["tags"]),
                    )
                    if (
                        worker not in roster
                        or receipt["gateway"] not in row["gateways"]
                        or receipt["elapsed_ms"] > row["window_ms"]
                    ):
                        raise ValueError("Receipt context")
                    valid.add(worker)
                if row["accepted"] and valid != set(roster):
                    raise ValueError("Accepted without full receipt coverage")
            totals[mode].append(sum(r["accepted"] for r in rows))
            walls[mode].extend(r["total_round_s"] for r in rows)
    if missing:
        return dict(complete=False, missing=missing, control_log_fetch_errors=diagnostic_errors)
    rng = np.random.default_rng(734128)
    n = len(plan["seeds"])
    indices = rng.integers(n, size=(10000, n))
    effects = {}
    for mode in plan["modes"]:
        delta = np.array(totals["joint"]) - totals[mode] if "joint" in totals else np.zeros(n)
        effects[mode] = dict(
            accepted=totals[mode],
            mean=float(np.mean(totals[mode])),
            joint_minus=float(delta.mean()),
            conditional_seed_ci=np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            p50_p95_total_s=np.quantile(walls[mode], [0.5, 0.95]).tolist(),
        )
    return dict(
        complete=True,
        control_log_fetch_errors=diagnostic_errors,
        blocks=n,
        results=effects,
        scope=plan["paired_scope"],
        caution="Seed/time-block uncertainty is conditional on these hosts and sequential observation periods. Pilot size does not establish confirmatory power.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = audit(a.output)
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["complete"] else 2)
