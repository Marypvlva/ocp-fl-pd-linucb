"""Locally frozen mechanism and resource study. All outputs are append-only per job."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HISTORICAL = ROOT / "research_workspace/outputs/af_latent_20261119/study3"
sys.path.append(str(HISTORICAL))
from sim import execute
from adaptive_planning.cli import source_hashes
from adaptive_planning.remedy_world import WorldSpec, make_world

AF = dict(
    kind="pd",
    reward_model="factored",
    attribution="latent",
    alpha=0.05,
    ridge=10.0,
    dual="resolve",
    base_context=False,
)
POLICIES = dict(
    AF_L=AF,
    filter_only=dict(AF, alpha=0.0, residual_learning=False),
    residual_greedy=dict(AF, alpha=0.0),
    gradient=dict(AF, dual="gradient"),
    no_price=dict(AF, alpha=0.0, dual="gradient", eta=0.0),
    one_price=dict(AF, secondary_price=False),
    old_greedy=dict(kind="plugin", pacing="dual", attributed=True),
)
N16 = dict(
    population=16, sample_size=4, group_sizes=(3, 3, 3), repairable=(0.25, 0.85), proof=(0.25, 0.85)
)
CELLS = dict(
    S1=dict(),
    D2=dict(traffic_ratio=1.15, budget_ratio=1.5),
    S10=dict(population=16, sample_size=4, group_sizes=(3, 3, 3), repairable=(0.25, 0.85)),
    tight=dict(N16, budget_ratio=1.10, traffic_ratio=1.0475),
    medium=dict(N16, budget_ratio=1.25, traffic_ratio=1.11875),
    loose=dict(N16, budget_ratio=3.0, traffic_ratio=1.95),
    exogenous=dict(N16, budget_ratio=1.25, traffic_ratio=1.11875),
    noisy_upload=dict(
        N16, budget_ratio=1.25, traffic_ratio=1.11875, full_update_failure=0.3, backup_failure=0.3
    ),
)


def hashes():
    files = source_hashes(ROOT)
    for folder in (ROOT / "research/study4", HISTORICAL):
        for f in folder.glob("*.py"):
            files[str(f.relative_to(ROOT))] = sha256(f.read_bytes()).hexdigest()
    return files


def world_for(cell, seed, plan):
    w = make_world(WorldSpec(**plan["cells"][cell]), seed)
    if cell == "exogenous":
        # Preserve the exact fault sequence and its run lengths, decouple its
        # original memberships from the current (independently random) roster.
        shift = w.spec.horizon // 2
        w.indicators = w.indicators[shift:] + w.indicators[:shift]
        w.faults = w.faults[shift:] + w.faults[:shift]
    return w


def job(args):
    out, cell, seed = args
    plan = json.loads((out / "plan.json").read_text())
    destination = out / "jobs" / f"{cell}_{seed}.json"
    if destination.exists():
        return
    w = world_for(cell, seed, plan)
    unrestricted = sum(
        any(w.accepted(t, a) for a in ("F+P", "F+B", "R+P", "R+B")) for t in range(w.spec.horizon)
    )
    oracle = w.oracle2()
    # Counterfactual oracles remove just one quota, retaining the same trace.
    oracle_credit = w.oracle()
    from dataclasses import replace

    traffic_only = replace(w, spec=replace(w.spec, budget_ratio=3.0))
    oracle_traffic = traffic_only.oracle2() if w.traffic_budget is not None else unrestricted
    rows = []
    for name, spec in plan["policies"].items():
        result = execute(spec, w, delay=plan["delay"])
        assert result["accepted"] <= oracle <= unrestricted
        rows.append(
            dict(
                cell=cell,
                seed=seed,
                policy=name,
                oracle=oracle,
                unrestricted=unrestricted,
                oracle_credit=oracle_credit,
                oracle_traffic=oracle_traffic,
                **result,
            )
        )
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(rows))
    temp.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "freeze":
        out.mkdir(parents=True, exist_ok=True)
        plan = dict(
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            cells=CELLS,
            policies=POLICIES,
            seeds=list(range(610000, 610128)),
            delay=2,
            primary_cells=["S1", "D2", "S10"],
            primary="AF_L-filter_only, paired mean across three cells within seed; two-sided Wilcoxon .05, positive mean required. Practical superiority requires lower 95% mean bootstrap bound >1 accepted round per 480.",
            secondary="Per-cell AF_L-filter_only Holm across 8 cells. Residual_greedy-filter_only and AF_L-residual_greedy descriptive mechanism contrasts. All other policy contrasts exploratory.",
            sample_size="Development differences sd max 3.464 (8 seeds); normal approximation (z_(1-.05/6)+z_.8)^2 sd^2 /1^2 approximately 126, rounded to128. Approximation is for mean differences, Wilcoxon power may differ.",
            bootstrap="10000 seed-block resamples, RNG 20261120. No pooling with study3 or development.",
            sources=hashes(),
        )
        (out / "plan.json").open("x").write(json.dumps(plan, indent=2))
        with zipfile.ZipFile(out / "frozen_sources.zip", "x", zipfile.ZIP_DEFLATED) as z:
            for name in plan["sources"]:
                z.write(ROOT / name, name)
        return
    plan = json.loads((out / "plan.json").read_text())
    if hashes() != plan["sources"]:
        raise ValueError("Frozen study sources changed")
    (out / "jobs").mkdir(exist_ok=True)
    jobs = [(out, c, s) for c in plan["cells"] for s in plan["seeds"]]
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(job, jobs, chunksize=1), 1):
            if i % 32 == 0:
                print(i, len(jobs), round(time.time() - start, 1), flush=True)


if __name__ == "__main__":
    main()
