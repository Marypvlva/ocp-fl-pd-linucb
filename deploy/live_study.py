"""Paired live study on the TCP stand (local processes or Yandex Cloud VMs).

Every (cell, seed) is one frozen latent fault world (remedy_world). All policies
replay the same world, the same OCP cohorts (virtual genesis + per-seed commit
seeds) and the same model seed; only the planner differs. Policy order is
randomized per seed. Each episode is a full ocp_experiment run: commit-reveal
ceremony in the local EVM, BLS beacon verification, worker gradients, edge
aggregation, projection or signed-update verification over TCP.

Usage
  python deploy/live_study.py plan  --cells D2 S11 --seeds 8 --horizon 160 --out study_live
  python deploy/live_study.py run   --out study_live [--placement placement.json]
  python deploy/live_study.py summarize --out study_live

`plan` freezes plan.json (cells, seeds, policies, world parameters, source
hashes). `run` is resumable: finished episodes are skipped. With --placement
the roles run on the hosts of placement.json (RemoteCluster, real edge outages).
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_planning.remedy_world import ACTION_COSTS, ACTION_TRAFFIC, WorldSpec, make_world  # noqa: E402

AF_L = dict(
    kind="pd",
    reward_model="factored",
    attribution="latent",
    alpha=0.05,
    ridge=10.0,
    dual="resolve",
    base_context=False,
)
POLICIES = {
    "AF_L": AF_L,
    "filter_only": dict(AF_L, alpha=0.0, residual_learning=False),
    "residual_greedy": dict(AF_L, alpha=0.0),
    "greedy_dual": dict(kind="plugin", pacing="dual", attributed=True),
    "rescue_forecast_orig": dict(kind="plugin", pacing="quantile", attributed=False),
    "PD_original": dict(kind="pd", attributed=False, implied_feedback=False, alpha=1.0, eta=0.05),
}
# Cells of study 3 that fit the live sampler (<= 9 built-in EVM accounts).
CELLS = {
    "joint_tight": dict(
        repairable=(0.25, 0.85), proof=(0.25, 0.85), budget_ratio=1.10, traffic_ratio=1.0475
    ),
    "joint_medium": dict(
        repairable=(0.25, 0.85), proof=(0.25, 0.85), budget_ratio=1.25, traffic_ratio=1.11875
    ),
    "S1_base": dict(),
    "S2_proof_heavy": dict(proof=(0.2, 0.85)),
    "S4_n8_k2": dict(population=8, sample_size=2, group_sizes=(2, 2, 2)),
    "S11_backup_drift": dict(repairable=(0.25, 0.85), backup_regime=(0.5, 0.98, 0.95)),
    "D2_traffic115_credit150": dict(traffic_ratio=1.15, budget_ratio=1.5),
    "D6_backup_drift": dict(
        traffic_ratio=1.15,
        budget_ratio=1.5,
        repairable=(0.25, 0.85),
        backup_regime=(0.5, 0.98, 0.95),
    ),
    "D7_backup_and_upload_drift": dict(
        traffic_ratio=1.2,
        budget_ratio=1.5,
        repairable=(0.2, 0.85),
        proof=(0.2, 0.85),
        backup_regime=(0.5, 0.98, 0.95),
        upload_regime=(0.5, 0.98, 0.95),
    ),
}
GENESIS = 2524608000
LOG_LOCK = threading.Lock()
SEED0 = 720000


def world_for(cell, seed, horizon, cells=None):
    spec = dict((CELLS if cells is None else cells)[cell])
    for key in ("proof", "repairable", "common", "group_sizes", "backup_regime", "upload_regime"):
        if key in spec:
            spec[key] = tuple(spec[key])
    return make_world(WorldSpec(horizon=horizon, **spec), seed)


def config_for(plan, cell, seed, policy, placement=None):
    world = world_for(cell, seed, plan["horizon"], plan["cells"])
    spec = dict(plan["policies"][policy])
    config = dict(
        workers=[world.spec.population],
        sample_size=world.spec.sample_size,
        hidden=8,
        rounds=plan["horizon"],
        model_seeds=[seed],
        learning_rate=0.5,
        target_loss=0.05,
        budget_credits=world.budget,
        action_costs=ACTION_COSTS,
        available_actions=list(ACTION_COSTS),
        pd_alpha=spec.pop("alpha", 0.1),
        pd_ridge=spec.pop("ridge", 1.0),
        pd_eta=spec.pop("eta", 0.05),
        pd_feedback_delay=plan["delay"],
        pd_reward="success",
        pd_reward_reference_seconds=0.1,
        conditions=["remedy_world"],
        remedy_traces={str(seed): world.trace_config()},
        deadline=plan["deadline_s"],
        rpc_timeout=plan["rpc_timeout_s"],
        network_profile="unlimited",
        stage=f"live_{cell}_{policy}",
        stop_at_target=False,
        trace_genesis=GENESIS,
        protocol_feedback_schema=3,
        af_planner=spec,
    )
    if world.traffic_budget is not None:
        config.update(traffic_costs=ACTION_TRAFFIC, traffic_budget=world.traffic_budget)
    if placement is not None:
        ssh = placement.get("ssh", {}) if isinstance(placement, dict) else {}
        roles = {"roles": placement["roles"]} if isinstance(placement, dict) else placement
        config["distributed"] = dict(placement=roles, edge_outage=plan["edge_outage"], **ssh)
    return config, world


def deploy_hashes():
    return {
        str(f.relative_to(ROOT)): sha256(f.read_bytes()).hexdigest()
        for f in sorted((ROOT / "deploy").rglob("*.py"))
    }


def plan_digest(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_plan(plan):
    from adaptive_planning.cli import source_hashes

    if plan_digest(plan) != plan.get("plan_sha256"):
        raise ValueError("Frozen plan changed")
    if plan["source_sha256"] != source_hashes() or plan["deploy_sha256"] != deploy_hashes():
        raise ValueError("Sources differ from the frozen live plan")


def require_complete(out, plan):
    missing = []
    for cell, seed, policy in episodes(plan):
        folder = out / "runs" / cell / str(seed) / policy
        if not (folder / "summary.csv").exists() or not (folder / "rounds.json").exists():
            missing.append(str(folder.relative_to(out)))
            continue
        metadata = json.loads((folder / "metadata.json").read_text())
        expected_config, _ = config_for(plan, cell, seed, policy, plan.get("placement"))
        if metadata.get("config") != expected_config:
            raise ValueError(f"Episode configuration differs from plan: {folder}")
        rows = json.loads((folder / "rounds.json").read_text())
        if len(rows) != plan["horizon"] or [r["attempt"] for r in rows] != list(
            range(1, plan["horizon"] + 1)
        ):
            raise ValueError(f"Incomplete or duplicate rounds in {folder}")
    if missing:
        raise ValueError(f"Missing {len(missing)} planned episodes: {missing[:4]}")


def cmd_plan(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from adaptive_planning.cli import source_hashes

    plan = dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        cells={c: CELLS[c] for c in args.cells},
        seeds=list(range(args.seed0, args.seed0 + args.seeds)),
        horizon=args.horizon,
        delay=2,
        policies={p: POLICIES[p] for p in args.policies},
        deadline_s=args.deadline,
        rpc_timeout_s=args.rpc_timeout,
        edge_outage=args.edge_outage,
        genesis=GENESIS,
        order_rng=args.order_seed,
        primary_rival="filter_only",
        hypothesis="Each cell: AF_L - filter_only, two-sided Wilcoxon Holm over cells, positive mean required. Other rivals are exploratory. No efficacy claim from acceptance checks.",
        source_sha256=source_hashes(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        placement=json.loads(Path(args.placement).read_text())
        if getattr(args, "placement", None)
        else None,
    )
    if (
        not plan["cells"]
        or args.seeds < 1
        or args.horizon < 1
        or "AF_L" not in plan["policies"]
        or "filter_only" not in plan["policies"]
    ):
        raise ValueError("Plan requires positive size and both primary policies")
    plan["deploy_sha256"] = deploy_hashes()
    plan["plan_sha256"] = plan_digest(plan)
    with (out / "plan.json").open("x") as f:
        json.dump(plan, f, indent=2, default=list)
    print(
        f"plan: {len(plan['cells'])} cells x {len(plan['seeds'])} seeds x {len(plan['policies'])} policies"
    )


def episodes(plan):
    rng = random.Random(plan["order_rng"])
    for cell in plan["cells"]:
        for seed in plan["seeds"]:
            order = list(plan["policies"])
            rng.shuffle(order)
            for policy in order:
                yield cell, seed, policy


def run_episode(out, plan, placement, env, n, total, cell, seed, policy, stop_on_error):
    folder = out / "runs" / cell / str(seed) / policy
    if (folder / "summary.csv").exists():
        return "skipped"
    if folder.exists():
        # Retain failed attempts for diagnosis before safely retrying.
        folder.rename(folder.with_name(folder.name + ".failed-" + str(time.time_ns())))
    folder.parent.mkdir(parents=True, exist_ok=True)
    config, _ = config_for(plan, cell, seed, policy, placement)
    path = out / "configs" / f"{cell}_{seed}_{policy}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1))
    start = time.time()
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "adaptive_planning.ocp_experiment",
            "--config",
            str(path),
            "--output",
            str(folder),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    status = "ok" if done.returncode == 0 else "failed"
    with LOG_LOCK, (out / "progress.log").open("a") as log:
        log.write(
            json.dumps(
                dict(
                    n=n,
                    of=total,
                    cell=cell,
                    seed=seed,
                    policy=policy,
                    status=status,
                    wall_s=round(time.time() - start, 1),
                    utc=datetime.now(timezone.utc).isoformat(),
                )
            )
            + "\n"
        )
    if done.returncode:
        (out / "errors").mkdir(exist_ok=True)
        (out / "errors" / f"{cell}_{seed}_{policy}.txt").write_text(done.stderr[-20000:])
        if stop_on_error:
            raise RuntimeError(done.stderr[-3000:])
    return status


def cmd_run(args):
    from concurrent.futures import ThreadPoolExecutor

    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    check_plan(plan)
    placement = json.loads(Path(args.placement).read_text()) if args.placement else None
    if placement != plan.get("placement"):
        raise ValueError("Placement differs from the frozen live plan")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    if placement is not None:
        env["ADAPTIVE_CLOCK"] = "wall"
    todo = list(episodes(plan))
    # Episodes of one seed stay adjacent, so concurrent load is shared by the paired policies.
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [
            pool.submit(
                run_episode, out, plan, placement, env, n, len(todo), *item, args.stop_on_error
            )
            for n, item in enumerate(todo, 1)
        ]
        statuses = [f.result() for f in futures]
    print({k: statuses.count(k) for k in set(statuses)})
    require_complete(out, plan)


def episode_record(plan, cell, seed, policy, folder):
    """Live outcome per round plus the outcome predicted by the frozen world."""
    from adaptive_planning.ocp_experiment import remedy_fault

    config, world = config_for(plan, cell, seed, policy)
    rounds = json.loads((folder / "rounds.json").read_text())
    accepted = predicted = agree = rescued = credits = traffic = 0
    for row in rounds:
        action, selected = row["action"], tuple(row["certificate"]["selected"])
        live = bool(row["accepted"])
        fault = remedy_fault(config, row["attempt"], selected, seed)
        E, S = 0 in fault.failed_edges, 2 in fault.failed_edges
        M, U = bool(fault.missing_projection), bool(fault.missing_full_update)
        lane = (not E) or (action[0] == "R" and not S)
        expect = lane and ((not U) if action[-1] == "B" else (not M))
        fp_ok = (not E) and (not M)
        accepted += live
        predicted += expect
        agree += live == expect
        rescued += live and not fp_ok
        credits += ACTION_COSTS[action]
        traffic += ACTION_TRAFFIC[action]
    return dict(
        cell=cell,
        seed=seed,
        policy=policy,
        rounds=len(rounds),
        accepted=accepted,
        predicted=predicted,
        fidelity=agree / max(1, len(rounds)),
        rescued=rescued,
        credits=credits,
        traffic=traffic,
        budget=world.budget,
        traffic_budget=world.traffic_budget,
    )


def cmd_summarize(args):
    import numpy as np
    from scipy.stats import wilcoxon

    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    check_plan(plan)
    require_complete(out, plan)
    records = []
    for cell, seed, policy in episodes(plan):
        folder = out / "runs" / cell / str(seed) / policy
        if (folder / "summary.csv").exists():
            records.append(episode_record(plan, cell, seed, policy, folder))
    (out / "episodes.json").write_text(json.dumps(records, indent=1))
    rng = np.random.default_rng(20261119)
    table, pvals = [], []
    for cell in plan["cells"]:
        by = {(r["seed"], r["policy"]): r for r in records if r["cell"] == cell}
        for rival in [p for p in plan["policies"] if p != "AF_L"]:
            seeds = [s for s in plan["seeds"] if (s, "AF_L") in by and (s, rival) in by]
            if len(seeds) < 3:
                continue
            d = np.array(
                [by[s, "AF_L"]["accepted"] - by[s, rival]["accepted"] for s in seeds], float
            )
            boot = rng.choice(d, (10000, len(d))).mean(1)
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            table.append(
                dict(
                    cell=cell,
                    rival=rival,
                    n=len(seeds),
                    mean=d.mean(),
                    lo=np.percentile(boot, 2.5),
                    hi=np.percentile(boot, 97.5),
                    wins=int((d > 0).sum()),
                    losses=int((d < 0).sum()),
                    p=p,
                )
            )
            if rival == plan["primary_rival"]:
                pvals.append(p)
            else:
                table[-1]["exploratory"] = True
    primary_rows = [row for row in table if row["rival"] == plan["primary_rival"]]
    order = np.argsort(pvals)
    m, running = len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        primary_rows[i]["p_holm"] = running
    fidelity = np.mean([r["fidelity"] for r in records]) if records else float("nan")
    result = dict(
        episodes=len(records),
        mean_fidelity=fidelity,
        min_fidelity=min((r["fidelity"] for r in records), default=None),
        comparisons=table,
    )
    (out / "summary.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"episodes={len(records)} fidelity={fidelity:.4f}")
    for row in table:
        print(
            f"{row['cell']:28s} AF_L-{row['rival']:22s} n={row['n']:2d} "
            f"{row['mean']:+6.2f} [{row['lo']:+6.2f},{row['hi']:+6.2f}] "
            f"W/L={row['wins']}/{row['losses']} p={row.get('p_holm', row['p']):.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--placement", help="Freeze the exact deployment placement with the plan")
    p.add_argument(
        "--cells", nargs="+", default=["joint_tight", "joint_medium", "D7_backup_and_upload_drift"]
    )
    p.add_argument(
        "--policies", nargs="+", default=["AF_L", "filter_only", "residual_greedy", "greedy_dual"]
    )
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--seed0", type=int, default=SEED0)
    p.add_argument("--horizon", type=int, default=240)
    p.add_argument("--deadline", type=float, default=3.0)
    p.add_argument("--rpc-timeout", type=float, default=2.0)
    p.add_argument("--edge-outage", choices=["refuse", "silent"], default="refuse")
    p.add_argument("--order-seed", type=int, default=20261119)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--placement")
    r.add_argument("--stop-on-error", action="store_true")
    r.add_argument("--parallel", type=int, default=1)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    dict(plan=cmd_plan, run=cmd_run, summarize=cmd_summarize)[args.command](args)


if __name__ == "__main__":
    main()
