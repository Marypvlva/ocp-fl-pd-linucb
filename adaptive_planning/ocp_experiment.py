"""Joint foundation: OCP-FL selects clients; PD-LinUCB selects execution mode."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import uuid


from .cli import source_hashes
from .ocp_chain import OcpChain
from .ocp_randomness import _verify_bls
from .pd_linucb_planner import PDLinUCBPlanner
from .tcp_closed_loop_pilot import append_csv, capture_sources
from .tcp_dynamic_experiment import observed_issues
from .adaptive_scheduler import ACTIONS, Observation
from .pd_linucb import PrimalDualLinUCB
from .tcp_network import PROFILES
from .tcp_runtime import Cluster, Fault, flatten_model, make_model
from .training import make_workers
from .paths import ROOT
from .protocol_feedback import feedback_from_round

CONDITIONS = (
    "healthy",
    "non_reveal",
    "forged_schedule",
    "worker_dropout",
    "poisoning",
    "targeted_edge_failure",
    "intermittent_targeted_edge_failure",
    "irregular_targeted_edge_failure",
    "mixed_availability",
    "remedy_world",
)


def cluster_for(config, groups, chain):
    """Local multi-process cluster, or roles on separate hosts ("distributed" key)."""
    common = dict(
        hidden=config["hidden"],
        rpc_timeout=config["rpc_timeout"],
        selection_authority=chain,
        network=PROFILES[config["network_profile"]],
        app_byte_budget=config.get("app_byte_budget"),
    )
    if "distributed" not in config:
        return Cluster(groups, **common)
    from .distributed import RemoteCluster, load_placement

    spec = dict(config["distributed"])
    placement = load_placement(spec.pop("placement"), len(groups))
    return RemoteCluster(groups, placement=placement, **common, **spec)


def make_groups(population, condition):
    groups, validation, test = make_workers(population)
    if condition == "poisoning":
        groups = [(x, 9 - y if i == 0 else y) for i, (x, y) in enumerate(groups)]
    return groups, validation, test


def af_planner_for(config):
    """Planners of the attributed-factored study (see attributed_pd_linucb)."""
    from .attributed_pd_linucb import AttributedPDLinUCBPlanner
    from .remedy_controls import PlugInRemedyControl, ReactiveRemedyControl

    spec = dict(config["af_planner"])
    kind = spec.pop("kind")
    common = dict(
        budget=config["budget_credits"],
        horizon=config["rounds"],
        population=max(config["workers"]),
        feedback_delay=config["pd_feedback_delay"],
    )
    if "traffic_budget" in config:
        common.update(
            secondary_costs=config["traffic_costs"], secondary_budget=config["traffic_budget"]
        )
    actions = tuple(config.get("available_actions", ACTIONS))
    if kind == "pd":
        return AttributedPDLinUCBPlanner(
            costs=config["action_costs"],
            actions=actions,
            alpha=config["pd_alpha"],
            ridge=config["pd_ridge"],
            eta=config["pd_eta"],
            **common,
            **spec,
        )
    if kind == "plugin":
        return PlugInRemedyControl(costs=config["action_costs"], actions=actions, **common, **spec)
    if kind == "reactive":
        return ReactiveRemedyControl(
            costs=config["action_costs"], actions=actions, **common, **spec
        )
    raise ValueError("Unknown attributed-study planner")


def planner_for(config):
    if "af_planner" in config:
        return af_planner_for(config)
    planner = PDLinUCBPlanner(
        budget=config["budget_credits"],
        horizon=config["rounds"],
        costs=config["action_costs"],
        alpha=config["pd_alpha"],
        ridge=config["pd_ridge"],
        eta=config["pd_eta"],
        feedback_delay=config["pd_feedback_delay"],
        reward_mode=config["pd_reward"],
        reward_reference_seconds=config["pd_reward_reference_seconds"],
        action_schedule=config.get("action_schedule"),
        available_actions=config.get("available_actions"),
        selection_rule=config.get("selection_rule", "ucb"),
        roster_population=max(config["workers"]) if config.get("roster_context") else None,
        fund_horizon=config.get("fund_horizon", False),
    )

    if action := config.get("fixed_action"):
        if action not in planner.engine.actions:
            raise ValueError("Fixed action is unavailable")
        planner.engine = PrimalDualLinUCB(
            planner.engine.dimension,
            {action: config["action_costs"][action]},
            config["budget_credits"],
            config["rounds"],
            config["pd_alpha"],
            config["pd_ridge"],
            eta=0,
        )
        planner.ledger = planner.engine.ledger
    return planner


def round_fault(condition, attempt, selected, population, *, outages=None, affected_clients=None):
    if condition == "mixed_availability":
        if outages is None or not 1 <= attempt <= len(outages):
            raise ValueError("Mixed faults require a frozen trace")
        kind = outages[attempt - 1]
        if kind not in ("healthy", "edge", "shared_edge", "participant"):
            raise ValueError("Unknown mixed fault")
        if kind == "participant":
            # Required proof from an OCP-selected worker cannot be replaced by an edge replica.
            return Fault(missing_projection=(selected[0],))
        affected = range(population // 3) if affected_clients is None else affected_clients
        if kind in ("edge", "shared_edge") and not set(selected).isdisjoint(affected):
            # Nodes 0 and 2 provide the same lane. Shared outages defeat both copies.
            return Fault(failed_edges=(0,) if kind == "edge" else (0, 2))
        return Fault()
    if condition == "worker_dropout" and attempt % 3 == 0:
        return Fault(missing_compute=(selected[0],))
    # An outage targets the primary aggregator when an affected client is invited.
    outage = condition == "targeted_edge_failure" or (
        condition == "intermittent_targeted_edge_failure" and ((attempt - 1) // 4) % 2 == 1
    )
    if condition == "irregular_targeted_edge_failure":
        if outages is None or not 1 <= attempt <= len(outages):
            raise ValueError("Irregular faults require a frozen trace")
        outage = outages[attempt - 1]
    if outage and any(i < population // 3 for i in selected):
        return Fault(failed_edges=(0,))
    return Fault()


def remedy_fault(config, attempt, selected, seed):
    """Fault of the multi-remedy world for the actually certified roster."""
    trace = config["remedy_traces"][str(seed)]
    t = attempt - 1
    groups_r, groups_c, groups_m = trace["schedule"][t]
    members = set(selected)
    hit_r = trace["repairable"][t] and not members.isdisjoint(groups_r)
    hit_c = trace["common"][t] and not members.isdisjoint(groups_c)
    missing = tuple(i for i in selected if trace["proof"][t] and i in groups_m)
    backup_failed = hit_c or (hit_r and trace["backup_noise"][t])
    edges = (0, 2) if backup_failed else (0,) if hit_r else ()
    upload_noise = trace.get("upload_noise")
    missing_full = missing if upload_noise is not None and upload_noise[t] else ()
    return Fault(failed_edges=edges, missing_projection=missing, missing_full_update=missing_full)


def configured_fault(config, condition, attempt, selected, population, seed):
    if condition == "remedy_world":
        return remedy_fault(config, attempt, selected, seed)
    key = str(seed)
    trace = config.get("fault_traces", {}).get(key)
    groups = config.get("affected_clients", {})
    if condition == "mixed_availability" and trace[attempt - 1] == "shared_edge":
        groups = config.get("shared_affected_clients", groups)
    return round_fault(
        condition, attempt, selected, population, outages=trace, affected_clients=groups.get(key)
    )


def primary_edge_error(errors):
    return any(error.startswith(("e0:", "e1:")) for error in errors)


def backup_edge_error(errors):
    return any(error.startswith(("e2:", "e3:")) for error in errors)


def projection_missing(errors):
    return any("Missing projection participant" in error for error in errors)


def validate_config(config):
    required = {
        "workers",
        "sample_size",
        "hidden",
        "rounds",
        "model_seeds",
        "learning_rate",
        "target_loss",
        "budget_credits",
        "action_costs",
        "pd_alpha",
        "pd_ridge",
        "pd_eta",
        "pd_feedback_delay",
        "pd_reward",
        "pd_reward_reference_seconds",
        "conditions",
        "deadline",
        "rpc_timeout",
        "network_profile",
        "stage",
        "stop_at_target",
    }
    if not required <= config.keys() or config.keys() - required - {
        "fixed_action",
        "trace_genesis",
        "action_schedule",
        "available_actions",
        "fault_traces",
        "affected_clients",
        "shared_affected_clients",
        "roster_context",
        "fund_horizon",
        "selection_rule",
        "af_planner",
        "remedy_traces",
        "traffic_costs",
        "traffic_budget",
        "app_byte_budget",
        "protocol_feedback_schema",
        "distributed",
    }:
        raise ValueError("Unknown or missing joint configuration keys")
    if (
        not config["workers"]
        or len(set(config["workers"])) != len(config["workers"])
        or any(type(n) is not int or not 2 <= n <= 16 for n in config["workers"])
    ):
        raise ValueError("Worker counts must be distinct integers in 2..16")
    if type(config["sample_size"]) is not int or not 1 <= config["sample_size"] <= min(
        config["workers"]
    ):
        raise ValueError("Invalid sample size")
    if (
        type(config["hidden"]) is not int
        or not 1 <= config["hidden"] <= 2000
        or type(config["rounds"]) is not int
        or not 1 <= config["rounds"] <= 1000
    ):
        raise ValueError("Invalid model/horizon")
    if (
        not config["model_seeds"]
        or len(set(config["model_seeds"])) != len(config["model_seeds"])
        or any(type(s) is not int or s < 0 for s in config["model_seeds"])
    ):
        raise ValueError("Invalid model seeds")
    if (
        not config["conditions"]
        or len(set(config["conditions"])) != len(config["conditions"])
        or not set(config["conditions"]) <= set(CONDITIONS)
    ):
        raise ValueError("Invalid stress conditions")
    if (
        type(config["stop_at_target"]) is not bool
        or not 0 < config["learning_rate"] <= 2
        or not 0 < config["target_loss"] < 10
        or not 0 < config["deadline"] <= 60
        or not 0 < config["rpc_timeout"] <= 30
        or config["network_profile"] not in PROFILES
    ):
        raise ValueError("Invalid training/runtime configuration")
    if "fixed_action" in config and "action_schedule" in config:
        raise ValueError("Fixed action and scheduled control are mutually exclusive")
    if "fixed_action" in config and (
        config.get("selection_rule", "ucb") != "ucb" or config.get("fund_horizon")
    ):
        raise ValueError("Fixed actions cannot use the reserve selection controls")
    if "fixed_action" in config and (
        config["fixed_action"] not in ACTIONS or config["pd_eta"] != 0
    ):
        raise ValueError("Invalid fixed policy")
    if "trace_genesis" in config and (
        type(config["trace_genesis"]) is not int
        or config["trace_genesis"] <= 0
        or max(config["workers"]) > 16
    ):
        raise ValueError("Paired traces require virtual genesis and at most 16 local test accounts")
    if "remedy_world" in config["conditions"]:
        traces = config.get("remedy_traces")
        if not isinstance(traces, dict) or set(traces) != {
            str(seed) for seed in config["model_seeds"]
        }:
            raise ValueError("The remedy world requires one frozen trace per model seed")
        for trace in traces.values():
            if any(
                len(trace[k]) != config["rounds"]
                for k in ("repairable", "common", "proof", "backup_noise", "schedule")
            ):
                raise ValueError("Remedy traces must cover the horizon")
            if "upload_noise" in trace and len(trace["upload_noise"]) != config["rounds"]:
                raise ValueError("Signed-update failure trace must cover the horizon")
    elif "remedy_traces" in config:
        raise ValueError("Remedy traces require the remedy_world condition")
    if ("traffic_budget" in config) != ("traffic_costs" in config):
        raise ValueError("Traffic budget and costs go together")
    if "app_byte_budget" in config and (
        type(config["app_byte_budget"]) is not int or not 0 < config["app_byte_budget"] < 2**63
    ):
        raise ValueError("Application byte budget must be a positive int64")
    if "protocol_feedback_schema" in config and config["protocol_feedback_schema"] not in (2, 3):
        raise ValueError("Unsupported protocol feedback schema")
    irregular = "irregular_targeted_edge_failure" in config["conditions"]
    mixed = "mixed_availability" in config["conditions"]
    if type(config.get("roster_context", False)) is not bool:
        raise ValueError("Roster context must be a boolean")
    for key in ("affected_clients", "shared_affected_clients"):
        if key not in config:
            continue
        groups = config[key]
        if (
            not mixed
            or not isinstance(groups, dict)
            or set(groups) != {str(seed) for seed in config["model_seeds"]}
            or any(
                not isinstance(group, list)
                or not group
                or any(type(i) is not int or not 0 <= i < min(config["workers"]) for i in group)
                or len(set(group)) != len(group)
                for group in groups.values()
            )
        ):
            raise ValueError("Affected clients require one valid mixed-fault group per seed")
    if irregular and mixed:
        raise ValueError("Boolean and typed fault traces need separate configurations")
    if irregular or mixed:
        traces = config.get("fault_traces")
        if (
            not isinstance(traces, dict)
            or set(traces) != {str(seed) for seed in config["model_seeds"]}
            or any(
                not isinstance(trace, list)
                or len(trace) != config["rounds"]
                or any(
                    (type(value) is not bool)
                    if irregular
                    else (
                        type(value) is not str
                        or value not in ("healthy", "edge", "shared_edge", "participant")
                    )
                    for value in trace
                )
                for trace in traces.values()
            )
        ):
            raise ValueError("Frozen faults require one correctly typed horizon per model seed")
    elif "fault_traces" in config:
        raise ValueError("Fault traces require the irregular fault condition")
    planner_for(config)


def observation_for(config, selected, dimension, issues):
    profile = PROFILES[config["network_profile"]]
    return Observation(
        len(selected),
        dimension,
        profile.model_mbps,
        profile.edge_mbps,
        profile.verifier_mbps,
        profile.latency_ms,
        *issues,
    )


@contextmanager
def preserve_chain(chain, path):
    try:
        yield
    finally:
        path.write_text(json.dumps(chain.export(), indent=2))


def run_joint(config, population, seed, condition, folder):
    # Independent timing runs start cold, while duplicate checks inside a run share the cache.
    _verify_bls.cache_clear()
    groups, validation, test = make_groups(population, condition)
    genesis = config.get("trace_genesis")
    chain = OcpChain.local(population, config["sample_size"], genesis_timestamp=genesis)
    planner = planner_for(config)
    model = make_model(seed, config["hidden"], config["learning_rate"])
    initial_loss, _ = model.score(*validation)
    previous = None
    issues = (False, False)
    accepted = 0
    rows = []
    first = None
    label = f"w{population}_s{seed}_{condition}"
    startup = perf_counter()
    with (
        preserve_chain(chain, folder / f"chain_{label}.json"),
        cluster_for(config, groups, chain) as cluster,
    ):
        startup = perf_counter() - startup
        start = perf_counter()
        stop = "horizon_exhausted"
        if initial_loss <= config["target_loss"]:
            first = 0.0
        with (folder / f"journal_{label}.jsonl").open("x") as journal:
            for attempt in range(1, config["rounds"] + 1):
                if first is not None and config["stop_at_target"]:
                    stop = "target_reached"
                    break
                if planner.ledger.stop_reason():
                    stop = planner.ledger.stop_reason()
                    break
                if rows and rows[-1]["metrics"].get("app_byte_budget", {}).get("denied_frames"):
                    stop = "app_byte_budget_exhausted"
                    break
                cycle_start = perf_counter()
                protocol_start = perf_counter()
                receipt_start = len(chain.receipts)
                seeds = None
                if genesis is not None:
                    seeds = [
                        sha256(
                            f"ocp-pair/{genesis}/{population}/{seed}/{attempt}/{i}".encode()
                        ).digest()
                        for i in range(population)
                    ]
                ticket = chain.run_local_ceremony(
                    attempt,
                    missing=(0,) if condition == "non_reveal" else (),
                    seeds=seeds,
                )
                protocol_seconds = perf_counter() - protocol_start
                obs = observation_for(config, ticket["selected"], cluster.dimension, issues)
                service_start = perf_counter()
                planner.begin_round(obs, attempt, attempt)
                action = planner.choose(obs, previous, selected=ticket["selected"])
                decision = planner.last_decision.copy()
                blocked = 0
                if condition == "forged_schedule":
                    bad = json.loads(json.dumps(ticket))
                    bad["selected"] = list(reversed(ticket["selected"]))
                    if len(bad["selected"]) == 1:
                        bad["selected"] = [(bad["selected"][0] + 1) % population]
                    before = cluster.round_id
                    try:
                        cluster.run_round(
                            model, action, selection=bad, deadline_s=config["deadline"]
                        )
                    except ValueError:
                        blocked = 1
                    if not blocked or cluster.round_id != before:
                        raise AssertionError("Forged roster reached TCP execution")
                fault = configured_fault(
                    config, condition, attempt, ticket["selected"], population, seed
                )
                result = cluster.run_round(
                    model, action, selection=ticket, deadline_s=config["deadline"], fault=fault
                )
                accepted += int(result.accepted)
                loss, accuracy = model.score(*validation)
                metrics = cluster.collect_metrics(cluster.round_id)
                service_seconds = perf_counter() - service_start
                indicators = {}
                if "af_planner" in config:
                    indicators = feedback_from_round(
                        action, result.accepted, result.errors, result.metrics
                    ).planner_arguments()
                else:
                    indicators = dict(primary_edge_error=primary_edge_error(result.errors))
                planner.observe_round(
                    action,
                    bool(result.accepted),
                    service_seconds,
                    result.metrics["switch_wall_s"],
                    **indicators,
                )
                anchor_start = perf_counter()
                model_hash = chain.record_model(attempt, flatten_model(model))
                anchor_seconds = perf_counter() - anchor_start
                if first is None and loss <= config["target_loss"]:
                    first = perf_counter() - start
                row = dict(
                    run=label,
                    population=population,
                    model_seed=seed,
                    condition=condition,
                    attempt=attempt,
                    certificate=ticket,
                    observation=asdict(obs),
                    action=action,
                    decision=decision,
                    planner_statistics=planner.stats,
                    bandit_reward=planner.last_reward,
                    accepted=int(result.accepted),
                    accepted_updates=accepted,
                    validation_loss=loss,
                    validation_accuracy=accuracy,
                    model_hash=model_hash,
                    forged_schedule_blocked=blocked,
                    protocol_wall_s=protocol_seconds,
                    service_wall_s=service_seconds,
                    anchor_wall_s=anchor_seconds,
                    cycle_wall_s=perf_counter() - cycle_start,
                    elapsed_wall_s=perf_counter() - start,
                    first_quality_wall_s=first,
                    gas_used=sum(r["gas_used"] for r in chain.receipts[receipt_start:]),
                    errors=result.errors,
                    metrics={**result.metrics, **metrics},
                )
                if "af_planner" in config:
                    row["protocol_feedback"] = indicators
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                rows.append(row)
                previous = action
                issues = observed_issues(result)
                if result.metrics["update_bound_exceeded"]:
                    stop = "update_out_of_range"
                    break
            elapsed = perf_counter() - start
        if rows and rows[-1]["metrics"].get("app_byte_budget", {}).get("denied_frames"):
            stop = "app_byte_budget_exhausted"
        elif first is not None and config["stop_at_target"]:
            stop = "target_reached"
        elif stop == "horizon_exhausted" and planner.ledger.stop_reason():
            stop = planner.ledger.stop_reason()
    test_loss, test_accuracy = model.score(*test)
    ledger = json.loads((folder / f"chain_{label}.json").read_text())
    bls_cache = _verify_bls.cache_info()
    summary = dict(
        run=label,
        population=population,
        sample_size=config["sample_size"],
        model_seed=seed,
        condition=condition,
        initial_loss=initial_loss,
        final_loss=model.score(*validation)[0],
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        attempts=len(rows),
        accepted_updates=accepted,
        reached=int(first is not None),
        t_epsilon_wall_s=first,
        observed_wall_s=elapsed,
        startup_wall_s=startup,
        bls_cache_misses=bls_cache.misses,
        bls_cache_hits=bls_cache.hits,
        stop_reason=stop,
        spent_credits=float(planner.ledger.spent),
        remaining_credits=float(planner.ledger.remaining),
        reward_sum=sum(r["bandit_reward"] for r in rows),
        observed_rewards=planner.feedback_count(),
        pending_rewards=len(planner.engine.pending),
        gas_used=ledger["gas_used"],
        forged_schedule_blocked=sum(r["forged_schedule_blocked"] for r in rows),
        client_zero_selected=sum(0 in r["certificate"]["selected"] for r in rows),
        app_tx_bytes=sum(r["metrics"]["app_tx_bytes"] for r in rows),
    )
    summary["stand"] = "distributed" if "distributed" in config else "local"
    summary["physical_outage_rounds"] = sum(
        bool(r["metrics"].get("physical_outages")) for r in rows
    )
    if "app_byte_budget" in config:
        summary["app_byte_budget"] = config["app_byte_budget"]
        summary["app_byte_spent"] = rows[-1]["metrics"]["app_byte_budget"]["spent"] if rows else 0
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tcp_ocp_train.json")
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.workers:
        config["workers"] = args.workers
        config["sample_size"] = max(1, min(args.workers) // 2)
    validate_config(config)
    folder = args.output or ROOT / "results" / (
        "ocp_pd_"
        + config["stage"]
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    meta = dict(
        protocol_feedback_schema=3,
        status="running",
        family="ocp_pd_joint",
        bls_cache_policy="reset_per_run",
        config=config,
        source_sha256=source_hashes(),
        basis=["OCP-FL client lottery", "PD-LinUCB execution-mode bandit"],
        backend="local EthereumTester/PyEVM; known local BLS test beacon; no distributed consensus",
        stand=(
            "distributed roles over TCP (RemoteCluster, wall clock, physical edge outages)"
            if "distributed" in config
            else "local multi-process roles over loopback TCP"
        ),
        security="commit/reveal+future beacon slot+Keccak+unbiased shuffle+independent BLS verification; no weak fallback",
        timing="Full training T includes local EVM ceremony, audit and model anchors; no latency-overlap claim. PD throughput reward uses service cycle only.",
        budget="Known allocation credits for mode selection; Ethereum gas is separately measured, no token price or spending real funds.",
        planned_runs=len(config["workers"])
        * len(config["model_seeds"])
        * len(config["conditions"]),
    )
    metadata = folder / "metadata.json"
    metadata.write_text(json.dumps(meta, indent=2))
    capture_sources(folder)
    all_rows = []
    summaries = []
    try:
        for population in config["workers"]:
            for seed in config["model_seeds"]:
                for condition in config["conditions"]:
                    rows, summary = run_joint(config, population, seed, condition, folder)
                    all_rows.extend(rows)
                    summaries.append(summary)
                    append_csv(folder / f"summary_{summary['run']}.csv", [summary])
                    print(
                        f"workers={population} seed={seed} condition={condition}: {summary['accepted_updates']}/{summary['attempts']} updates, credits={summary['spent_credits']}, stop={summary['stop_reason']}",
                        flush=True,
                    )
        meta["status"] = "completed"
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        append_csv(folder / "summary.csv", summaries)
        (folder / "rounds.json").write_text(json.dumps(all_rows, indent=2, allow_nan=False))
        metadata.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {folder}", flush=True)
    return folder


if __name__ == "__main__":
    main()
