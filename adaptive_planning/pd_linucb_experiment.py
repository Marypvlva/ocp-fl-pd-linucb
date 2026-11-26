"""Integrate the article's budgeted bandit with actual verified TCP training."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import uuid

from . import tcp_dynamic_experiment as dynamic
from .availability_experiment import make_trace as availability_trace
from .cli import source_hashes
from .paths import ROOT
from .pd_linucb_planner import CONTEXT_NAMES
from .tcp_closed_loop_pilot import append_csv, capture_sources
from .tcp_network import PROFILES
from .network_telemetry import TELEMETRY_MODES
from .tcp_runtime import ACTIONS, Cluster, make_model
from .training import make_workers

import numpy as np

POLICIES = (*ACTIONS, "linucb", "pd_linucb")


def make_trace(config, repeat, workers):
    if config.pd_environment == "switching_network":
        return dynamic.make_trace(
            config.seed + repeat,
            config.rounds,
            workers,
            config.scenario,
            config.phase_min,
            config.phase_max,
        )
    return availability_trace(
        config.seed + repeat,
        config.rounds,
        workers,
        config.pd_environment,
        config.phase_min,
        config.phase_max,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tcp_pd_linucb_eval.json")
    parser.add_argument("--budgets", nargs="+", type=float)
    parser.add_argument("--planner-only", action="store_true")
    own, forwarded = parser.parse_known_args(argv)
    args = dynamic.parse_args(["--config", str(own.config), *forwarded])
    budgets = own.budgets or args.pd_budgets or [args.budget_credits]
    policies = ("pd_linucb",) if own.planner_only else POLICIES
    if (
        set(args.policies) != set(policies)
        or args.budget_credits is None
        or len(set(budgets)) != len(budgets)
    ):
        parser.error("Require PD/LinUCB, all four fixed actions, explicit distinct budgets")
    for budget in budgets:
        try:
            dynamic.create_pd_planner(
                "pd_linucb", argparse.Namespace(**{**vars(args), "budget_credits": budget})
            )
        except (ValueError, TypeError) as error:
            parser.error(str(error))
    if len(budgets) * len(args.policies) * args.repeats * (args.rounds + 1) >= 65536:
        parser.error("Too many rounds for one dynamic cluster")
    folder = args.output or ROOT / "results" / (
        "tcp_pd_linucb_"
        + args.stage
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    meta = dict(
        status="running",
        family="pd_linucb_budget",
        protocol_revision=2,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        budgets=budgets,
        policies=list(policies),
        mode="training" if own.planner_only else "comparative_pilot",
        planned_runs=len(args.workers) * args.repeats * len(policies) * len(budgets),
        primary_definition=(
            "single-PD training, no comparative claim"
            if own.planner_only
            else "PD minus eta=0 LinUCB, total verified-update reward under the smallest known-credit budget; pilot only"
        ),
        source_sha256=source_hashes(),
        context_names=list(CONTEXT_NAMES),
        network_profiles={k: v.to_dict() for k, v in PROFILES.items()},
        sensor=asdict(TELEMETRY_MODES[args.telemetry_mode]),
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "matplotlib", "scikit-learn", "cryptography")
        },
        resource="deterministic allocation credits, NOT actual wire bytes or elapsed seconds",
        reward_delivery="fixed artificial delay in selection steps, both zero and positive labels; model verification/update remains synchronous",
        fixed_policy_definition="singleton-action baseline stops when its own action is unaffordable; other cheaper actions are not substituted",
        warmup_definition="one equal pre-trial F+B outside measured T and bandit allocation budget",
    )
    metadata = folder / "metadata.json"
    metadata.write_text(json.dumps(meta, indent=2))
    capture_sources(folder)
    rows, summaries = [], []
    try:
        for count in args.workers:
            groups, validation, _ = make_workers(count)
            with Cluster(
                groups, hidden=args.hidden, rpc_timeout=args.rpc_timeout, dynamic_network=True
            ) as cluster:
                for repeat in range(args.repeats):
                    trace = make_trace(args, repeat, count)
                    (folder / f"trace_w{count}_r{repeat}.json").write_text(
                        json.dumps([asdict(t) for t in trace], indent=2)
                    )
                    cases = [
                        (index, budget, policy)
                        for index, budget in enumerate(budgets)
                        for policy in policies
                    ]
                    order = np.random.default_rng(args.seed + 20000 + repeat).permutation(
                        len(cases)
                    )
                    for position, index in enumerate(order):
                        budget_index, budget, policy = cases[int(index)]
                        case = f"budget{budget_index}"
                        config = argparse.Namespace(
                            **{**vars(args), "budget_credits": budget, "journal_namespace": case}
                        )
                        warm = cluster.run_round(
                            make_model(
                                dynamic.model_seed_for(args, repeat),
                                args.hidden,
                                args.learning_rate,
                            ),
                            "F+B",
                            deadline_s=args.deadline,
                        )
                        if not warm.accepted:
                            raise RuntimeError(f"Warmup failed: {warm.errors}")
                        rr, ss = dynamic.run_policy(
                            cluster, policy, trace, validation, config, repeat, position, folder
                        )
                        for row in rr:
                            row.update(condition=case, budget_credits=budget)
                        ss.update(condition=case, budget_credits=budget)
                        rows.extend(rr)
                        summaries.append(ss)
                        append_csv(
                            folder
                            / f"rounds_w{count}_r{repeat}_{case}_{policy.replace('+', '')}.csv",
                            rr,
                        )
                        print(
                            f"workers={count} repeat={repeat} budget={budget} policy={policy}: updates={ss['accepted_updates']} "
                            f"attempts={ss['attempts']} spent={ss['spent_credits']} stop={ss['stop_reason']}",
                            flush=True,
                        )
        meta["status"] = "completed"
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        append_csv(folder / "rounds.csv", rows)
        append_csv(folder / "summary.csv", summaries)
        metadata.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {folder}", flush=True)
    return folder


if __name__ == "__main__":
    main()
