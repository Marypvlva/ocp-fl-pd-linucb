"""Actual TCP training with network changes between rounds and causal schedulers."""

from .paths import ROOT
from .cli import source_hashes
import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
import uuid

import numpy as np

from .adaptive_scheduler import (
    Observation,
    NetworkOnlinePlanner,
    FrozenNetworkPlanner,
    network_rule,
    predicted_seconds,
)
from .tcp_closed_loop_pilot import (
    OnlinePlanner,
    append_csv,
    capture_sources,
    make_scenario,
    simple_rule,
)
from .tcp_network import PROFILES
from .tcp_runtime import ACTIONS, Cluster, Fault, make_model
from .network_telemetry import NetworkTelemetry, TELEMETRY_MODES
from .aligned_scheduler import AlignedNetworkPlanner
from .freshness_scheduler import FreshnessMixturePlanner
from .availability_scheduler import AvailabilityPlanner
from .pd_linucb import CreditBudget
from .pd_linucb_planner import DEFAULT_COSTS, PDLinUCBPlanner, outcome_reward

POLICIES = (
    *ACTIONS,
    "rule",
    "online",
    "network_rule",
    "network_online",
    "network_prior",
    "network_aligned",
    "network_freshness",
    "availability",
    "pd_linucb",
    "linucb",
)
MATCHED_POLICIES = ("network_aligned", "network_freshness")
PD_POLICIES = ("pd_linucb", "linucb")
DEFAULT_POLICIES = (*ACTIONS, "rule", "online", "network_rule", "network_online", "network_prior")
LEARNER_POLICIES = (
    "network_online",
    "network_prior",
    *MATCHED_POLICIES,
    "availability",
    *PD_POLICIES,
)


def create_pd_planner(policy, config):
    return PDLinUCBPlanner(
        budget=config.budget_credits,
        horizon=config.rounds,
        costs=config.action_costs,
        alpha=config.pd_alpha,
        ridge=config.pd_ridge,
        eta=config.pd_eta if policy == "pd_linucb" else 0.0,
        feedback_delay=config.pd_feedback_delay,
        reward_mode=config.pd_reward,
        reward_reference_seconds=config.pd_reward_reference_seconds,
    )


@dataclass(frozen=True)
class DynamicStep:
    profile: str
    fault: Fault


def make_trace(seed, rounds, workers, scenario, phase_min=4, phase_max=8):
    if not 1 <= phase_min <= phase_max:
        raise ValueError("Invalid phase lengths")
    faults = make_scenario(seed + 1000, rounds, workers, scenario)
    rng = np.random.default_rng(seed + 2000)
    profiles = ("edge_limited", "verifier_limited")
    current, left = int(rng.integers(0, 2)), 0
    trace = []
    for i in range(rounds):
        if left == 0:
            if trace:
                current = 1 - current
            left = int(rng.integers(phase_min, phase_max + 1))
        trace.append(DynamicStep(profiles[current], faults[i]))
        left -= 1
    return trace


def observation_for(profile, workers, dimension, edge_trouble=False, projection_missing=False):
    return Observation(
        workers,
        dimension,
        profile.model_mbps,
        profile.edge_mbps,
        profile.verifier_mbps,
        profile.latency_ms,
        edge_trouble,
        projection_missing,
    )


def model_seed_for(config, repeat):
    return config.model_seeds[repeat] if config.model_seeds else config.model_seed


def observed_issues(result):
    # Observable protocol outcomes only. The injection object is never consulted.
    edge = any(error.startswith(("e0:", "e1:", "e2:", "e3:")) for error in result.errors)
    edge |= any(
        "Подмена агрегированной суммы" in error or "Некорректная агрегированная" in error
        for error in result.errors
    )
    projection = any("Missing projection participant" in error for error in result.errors)
    return edge, projection


def run_policy(cluster, policy, trace, validation, config, repeat, order, folder):
    model_seed = model_seed_for(config, repeat)
    model = make_model(model_seed, config.hidden, config.learning_rate)
    loss, _ = model.score(*validation)
    initial_loss = loss
    legacy = OnlinePlanner(config.seed + 10_000 + repeat, config.switch_seconds)
    planner_type = (
        AvailabilityPlanner
        if policy == "availability"
        else FreshnessMixturePlanner
        if policy == "network_freshness"
        else AlignedNetworkPlanner
        if policy == "network_aligned"
        else FrozenNetworkPlanner
        if policy == "network_prior"
        else NetworkOnlinePlanner
    )
    adaptive = (
        create_pd_planner(policy, config)
        if policy in PD_POLICIES
        else planner_type(config.seed + 11_000 + repeat, config.switch_seconds)
    )
    ledger = (
        adaptive.ledger
        if policy in PD_POLICIES
        else CreditBudget(config.action_costs, config.budget_credits, config.rounds)
        if getattr(config, "budget_credits", None) is not None
        else None
    )
    previous, previous_profile = None, None
    legacy_observed = (False, False, False)
    edge_trouble = projection_missing = False
    sensor = NetworkTelemetry(TELEMETRY_MODES[config.telemetry_mode], config.seed + 3000 + repeat)
    accepted = switches = profile_changes = 0
    rows = []
    stop_reason = "max_rounds"
    start = perf_counter()
    reached = 0.0 if loss <= config.target_loss else None
    namespace = getattr(config, "journal_namespace", "")
    journal = (
        folder
        / f"journal_{namespace + '_' if namespace else ''}w{len(cluster.groups)}_r{repeat}_{config.telemetry_mode}_{policy.replace('+', '')}.jsonl"
    )
    with journal.open("x", encoding="utf-8") as stream:
        for attempt, step in enumerate(trace, 1):
            if reached is not None and config.stop_at_target:
                break
            if ledger:
                reason = ledger.stop_reason()
                if not reason and policy in ACTIONS and policy not in ledger.feasible_actions():
                    reason = "fixed_action_unaffordable"
                if reason:
                    stop_reason = reason
                    break
            cycle_start = perf_counter()
            profile = PROFILES[step.profile]
            sensor_start = perf_counter()
            reading = sensor.observe(profile, attempt)
            sensor_s = perf_counter() - sensor_start
            observation = observation_for(
                reading, len(cluster.groups), cluster.dimension, edge_trouble, projection_missing
            )
            select_start = perf_counter()
            decision = {}
            if policy in (*MATCHED_POLICIES, *PD_POLICIES):
                adaptive.begin_round(observation, reading.capture_attempt, attempt)
            if policy in ACTIONS:
                action = policy
            elif policy == "rule":
                action = simple_rule(legacy_observed, len(cluster.groups))
            elif policy == "online":
                action = legacy.choose(legacy_observed, previous)
            elif policy == "network_rule":
                action = network_rule(observation, previous, config.switch_seconds)
            else:
                action = adaptive.choose(observation, previous)
                decision = adaptive.last_decision.copy()
            if ledger and policy not in PD_POLICIES:
                ledger.commit(action)
            selection_s = perf_counter() - select_start
            switched = previous is not None and previous != action
            profile_changed = previous_profile is not None and step.profile != previous_profile
            predicted = predicted_seconds(action, observation, previous, config.switch_seconds)
            result = cluster.run_round(
                model,
                action,
                deadline_s=config.deadline,
                fault=step.fault,
                switched=switched,
                switch_s=config.switch_seconds,
                network_profile=profile,
            )
            accepted += int(result.accepted)
            switches += int(switched)
            profile_changes += int(profile_changed)
            t = perf_counter()
            if result.accepted:
                loss, _ = model.score(*validation)
            validation_s = perf_counter() - t
            quality_time = perf_counter() - start
            if loss <= config.target_loss and reached is None:
                reached = quality_time
            t = perf_counter()
            telemetry = cluster.collect_metrics(cluster.round_id)
            telemetry_s = perf_counter() - t
            cycle_s = perf_counter() - cycle_start
            if policy == "online":
                legacy.observe(
                    legacy_observed,
                    action,
                    result.accepted,
                    cycle_s,
                    result.metrics["switch_wall_s"],
                )
            elif policy in ("network_online", "network_prior", "availability"):
                adaptive.observe(
                    observation, action, result.accepted, cycle_s, result.metrics["switch_wall_s"]
                )
            elif policy in (*MATCHED_POLICIES, *PD_POLICIES):
                adaptive.observe_round(
                    action, result.accepted, cycle_s, result.metrics["switch_wall_s"]
                )
            row = dict(
                workers=len(cluster.groups),
                policy=policy,
                repeat=repeat,
                order=order,
                stage=config.stage,
                seed=config.seed,
                model_seed=model_seed,
                learning_rate=config.learning_rate,
                telemetry_mode=config.telemetry_mode,
                sensor_processing_wall_s=sensor_s,
                telemetry_source_attempt=reading.capture_attempt,
                telemetry_age_rounds=attempt - reading.capture_attempt,
                telemetry_source_profile=reading.source_profile,
                attempt=attempt,
                profile=step.profile,
                profile_changed=int(profile_changed),
                network_observation=json.dumps(asdict(observation), sort_keys=True),
                predicted_action_seconds=predicted,
                decision=json.dumps(decision),
                action_cost_credits=float(ledger.costs[action]) if ledger else "",
                spent_credits=float(ledger.spent) if ledger else "",
                remaining_credits=float(ledger.remaining) if ledger else "",
                bandit_reward=outcome_reward(
                    result.accepted, cycle_s, config.pd_reward, config.pd_reward_reference_seconds
                )
                if ledger
                else "",
                feedback_updates=adaptive.feedback_count() if policy in LEARNER_POLICIES else "",
                cost_feedback_updates=adaptive.cost_feedback_count()
                if policy in MATCHED_POLICIES
                else "",
                pending_cost_outcomes=len(adaptive.pending) if policy in MATCHED_POLICIES else "",
                cost_feedback_source_attempt=adaptive.last_cost_capture
                if policy in MATCHED_POLICIES
                else "",
                planner_statistics=json.dumps(adaptive.stats, sort_keys=True)
                if policy in LEARNER_POLICIES
                else "",
                accepted_updates=accepted,
                validation_loss=loss,
                target_loss=config.target_loss,
                selection_wall_s=selection_s,
                validation_wall_s=validation_s,
                telemetry_wall_s=telemetry_s,
                cycle_wall_s=cycle_s,
                quality_observed_wall_s=quality_time,
                elapsed_wall_s=perf_counter() - start,
                errors=json.dumps(result.errors, ensure_ascii=False),
                injected_fault=json.dumps(asdict(step.fault), sort_keys=True),
                traffic_is_lower_bound=int(
                    bool(result.metrics["timeouts"] or telemetry["telemetry_missing_nodes"])
                ),
                **result.metrics,
                **telemetry,
            )
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            edge_trouble, projection_missing = observed_issues(result)
            legacy_observed = (
                not result.accepted,
                result.metrics["round_wall_s"] > 0.5 * config.deadline,
                action.endswith("P") and not result.accepted,
            )
            previous, previous_profile = action, step.profile
            if result.metrics["update_bound_exceeded"]:
                stop_reason = "update_out_of_range"
                break
    if reached is not None and config.stop_at_target:
        stop_reason = "target_reached"
    elif ledger and stop_reason == "max_rounds" and ledger.steps >= ledger.horizon:
        stop_reason = "horizon_exhausted"
    summary = dict(
        workers=len(cluster.groups),
        policy=policy,
        repeat=repeat,
        order=order,
        stage=config.stage,
        seed=config.seed,
        model_seed=model_seed,
        learning_rate=config.learning_rate,
        telemetry_mode=config.telemetry_mode,
        initial_loss=initial_loss,
        final_loss=loss,
        target_loss=config.target_loss,
        reached=int(reached is not None),
        t_epsilon_wall_s=reached,
        observed_wall_s=perf_counter() - start,
        accepted_updates=accepted,
        attempts=len(rows),
        success_fraction=accepted / len(rows) if rows else 1.0,
        switches=switches,
        profile_changes=profile_changes,
        app_tx_bytes=sum(r["app_tx_bytes"] for r in rows),
        candidate_rejections=sum(r["candidate_rejections"] for r in rows),
        verification_attempts=sum(r["verification_attempts"] for r in rows),
        timeout_count=sum(r["timeouts"] for r in rows),
        stop_reason=stop_reason,
        budget_credits=float(ledger.budget) if ledger else "",
        spent_credits=float(ledger.spent) if ledger else "",
        remaining_credits=float(ledger.remaining) if ledger else "",
        bandit_reward_sum=sum(r["bandit_reward"] for r in rows) if ledger else "",
        observed_rewards=adaptive.feedback_count() if policy in PD_POLICIES else "",
        pending_rewards=len(adaptive.engine.pending) if policy in PD_POLICIES else "",
        startup_wall_s=cluster.startup_s,
        traffic_is_lower_bound=int(any(r["traffic_is_lower_bound"] for r in rows)),
    )
    return rows, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[12])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(DEFAULT_POLICIES))
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=43026)
    parser.add_argument("--model-seed", type=int, default=2026)
    parser.add_argument("--model-seeds", nargs="+", type=int)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--telemetry-mode", choices=tuple(TELEMETRY_MODES), default="exact")
    parser.add_argument("--hidden", type=int, default=670)
    parser.add_argument("--target-loss", type=float, default=0.22)
    parser.add_argument("--deadline", type=float, default=1.5)
    parser.add_argument("--rpc-timeout", type=float, default=1.2)
    parser.add_argument("--switch-seconds", type=float, default=0.025)
    parser.add_argument("--scenario", choices=("clean", "faults"), default="faults")
    parser.add_argument("--phase-min", type=int, default=4)
    parser.add_argument("--phase-max", type=int, default=8)
    parser.add_argument(
        "--stage",
        choices=("smoke", "development", "evaluation", "ablation", "robustness"),
        default="evaluation",
    )
    parser.add_argument(
        "--primary-contrast",
        choices=("network_online - network_rule", "network_online - network_prior"),
        default="network_online - network_rule",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-credits", type=float)
    parser.add_argument("--pd-budgets", type=float, nargs="+")
    parser.add_argument("--action-costs", type=json.loads, default=DEFAULT_COSTS.copy())
    parser.add_argument("--pd-alpha", type=float, default=1.0)
    parser.add_argument("--pd-ridge", type=float, default=1.0)
    parser.add_argument("--pd-eta", type=float, default=0.05)
    parser.add_argument("--pd-feedback-delay", type=int, default=0)
    parser.add_argument("--pd-reward", choices=("success", "throughput"), default="success")
    parser.add_argument("--pd-reward-reference-seconds", type=float, default=0.1)
    parser.add_argument(
        "--pd-environment",
        choices=("switching_network", "persistent_delayed", "persistent_fast", "healthy"),
        default="switching_network",
    )
    parser.add_argument(
        "--continue-after-target", dest="stop_at_target", action="store_false", default=True
    )
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        settings = json.loads(pre.config.read_text())
        allowed = {a.dest for a in parser._actions} - {"config", "help"}
        if set(settings) - allowed:
            parser.error("Unknown config keys")
        parser.set_defaults(**settings)
    args = parser.parse_args(argv)
    if (
        not isinstance(args.action_costs, dict)
        or set(args.action_costs) != set(ACTIONS)
        or type(args.stop_at_target) is not bool
    ):
        parser.error("All four action costs and a boolean stop_at_target are required")
    if any(p in PD_POLICIES for p in args.policies) and args.budget_credits is None:
        parser.error("PD-LinUCB requires an explicit known-cost budget")
    if args.budget_credits is not None:
        if any(p not in (*ACTIONS, *PD_POLICIES) for p in args.policies):
            parser.error("Budgeted trials support fixed actions, pd_linucb and linucb")
        try:
            create_pd_planner("pd_linucb", args)
        except (ValueError, TypeError) as error:
            parser.error(str(error))
    if (
        not args.workers
        or not all(type(n) is int and 2 <= n <= 16 for n in args.workers)
        or len(set(args.workers)) != len(args.workers)
        or not args.policies
        or len(set(args.policies)) != len(args.policies)
        or any(p not in POLICIES for p in args.policies)
        or not 1 <= args.rounds <= 1000
        or not 1 <= args.repeats <= 40
        or not 1 <= args.hidden <= 2000
        or not 0 < args.target_loss < 10
        or not 0 < args.deadline <= 60
        or not 0 < args.rpc_timeout <= 30
        or not 0 <= args.switch_seconds <= 5
        or not 1 <= args.phase_min <= args.phase_max <= 100
        or args.seed < 0
        or args.model_seed < 0
        or not np.isfinite(args.learning_rate)
        or not 0 < args.learning_rate <= 2
        or args.telemetry_mode not in TELEMETRY_MODES
        or (
            args.model_seeds is not None
            and (
                len(args.model_seeds) != args.repeats
                or any(type(s) is not int or s < 0 for s in args.model_seeds)
            )
        )
        or args.rounds * args.repeats * len(args.policies) + args.repeats * len(args.policies)
        >= 65536
    ):
        parser.error("Invalid experiment limits")
    return args


def main(argv=None):
    args = parse_args(argv)
    from .training import make_workers

    root = ROOT
    folder = args.output or root / "results" / (
        "tcp_dynamic_"
        + args.stage
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    meta = dict(
        status="running",
        protocol_revision=2,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        python=platform.python_version(),
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "matplotlib", "scikit-learn", "cryptography")
        },
        source_sha256=source_hashes(),
        primary_contrast=args.primary_contrast,
        network_profiles={k: v.to_dict() for k, v in PROFILES.items()},
        time_definition="physical localhost T_epsilon; includes prior validation, telemetry and journal writes; excludes startup and warmup",
        observation_definition="current advertised profile and past protocol outcomes, never future phases/faults",
        trace_index="attempt; changes only between rounds",
        primary_ci="paired percentile bootstrap of mean difference; descriptive, no power guarantee",
    )
    meta_path = folder / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    capture_sources(folder)
    all_rows, summaries = [], []
    try:
        for count in args.workers:
            groups, validation, _ = make_workers(count)
            with Cluster(
                groups, hidden=args.hidden, rpc_timeout=args.rpc_timeout, dynamic_network=True
            ) as cluster:
                for repeat in range(args.repeats):
                    trace = make_trace(
                        args.seed + repeat,
                        args.rounds,
                        count,
                        args.scenario,
                        args.phase_min,
                        args.phase_max,
                    )
                    (folder / f"trace_w{count}_r{repeat}.json").write_text(
                        json.dumps([asdict(step) for step in trace], indent=2)
                    )
                    policies = np.random.default_rng(args.seed + 20000 + repeat).permutation(
                        args.policies
                    )
                    for order, policy in enumerate(policies):
                        warm = cluster.run_round(
                            make_model(
                                model_seed_for(args, repeat), args.hidden, args.learning_rate
                            ),
                            "F+B",
                            deadline_s=args.deadline,
                        )
                        if not warm.accepted:
                            raise RuntimeError(f"Warmup failed: {warm.errors}")
                        rows, summary = run_policy(
                            cluster, str(policy), trace, validation, args, repeat, order, folder
                        )
                        all_rows.extend(rows)
                        summaries.append(summary)
                        append_csv(
                            folder / f"rounds_w{count}_r{repeat}_{policy.replace('+', '')}.csv",
                            rows,
                        )
                        print(
                            f"workers={count} repeat={repeat} policy={policy}: accepted="
                            f"{summary['accepted_updates']}/{summary['attempts']} "
                            f"T_epsilon={summary['t_epsilon_wall_s']} switches={summary['switches']} "
                            f"profile_changes={summary['profile_changes']} loss={summary['final_loss']:.6f}",
                            flush=True,
                        )
        meta["status"] = "completed"
    except BaseException as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        append_csv(folder / "rounds.csv", all_rows)
        append_csv(folder / "summary.csv", summaries)
        meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {folder}", flush=True)
    return folder


if __name__ == "__main__":
    main()
