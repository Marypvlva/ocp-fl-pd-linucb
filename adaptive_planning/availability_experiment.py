"""Paired real-TCP study of persistent edge and projection availability regimes."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import uuid

from . import tcp_dynamic_experiment as dynamic
from .cli import source_hashes
from .network_telemetry import TELEMETRY_MODES
from .paths import ROOT
from .tcp_closed_loop_pilot import append_csv, capture_sources
from .tcp_network import PROFILES
from .tcp_runtime import ACTIONS, Cluster, Fault, make_model
from .training import make_workers

import numpy as np

POLICIES = (*ACTIONS, "network_rule", "network_freshness", "availability")
REGIMES = ("healthy", "projection_outage", "edge_outage", "combined_outage")


@dataclass(frozen=True)
class AvailabilityFault(Fault):
    projection_failure_delay_s: float = 0.0


@dataclass(frozen=True)
class AvailabilityStep:
    profile: str
    fault: Fault
    regime: str


def make_trace(seed, rounds, workers, condition, phase_min, phase_max):
    """A fresh permutation of all four regimes per cycle; never exposed to policy."""
    rng = np.random.default_rng(seed + 4000)
    result = []
    while len(result) < rounds:
        regimes = rng.permutation(REGIMES) if condition != "healthy" else ["healthy"]
        for regime in regimes:
            duration = int(rng.integers(phase_min, phase_max + 1))
            failed = (
                (int(rng.integers(0, 2)),) if regime in ("edge_outage", "combined_outage") else ()
            )
            missing = (
                (int(rng.integers(0, workers)),)
                if regime in ("projection_outage", "combined_outage")
                else ()
            )
            step = AvailabilityStep(
                "verifier_limited",
                AvailabilityFault(
                    failed_edges=failed,
                    missing_projection=missing,
                    projection_failure_delay_s=0.10 if condition == "persistent_delayed" else 0.0,
                ),
                str(regime),
            )
            result.extend([step] * min(duration, rounds - len(result)))
            if len(result) == rounds:
                break
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tcp_availability_eval.json")
    own, forwarded = parser.parse_known_args(argv)
    args = dynamic.parse_args(["--config", str(own.config), *forwarded])
    if set(args.policies) != set(POLICIES):
        parser.error(
            "Availability study requires all four fixed policies, network_rule, freshness and availability"
        )
    fast_policies = (*ACTIONS, "availability")
    cases = (
        [("persistent_delayed", p) for p in POLICIES]
        + [("persistent_fast", p) for p in fast_policies]
        + [("healthy", p) for p in ("F+P", "availability")]
    )
    if len(cases) * args.repeats * (args.rounds + 1) >= 65536:
        parser.error("Too many rounds for one cluster")
    folder = args.output or ROOT / "results" / (
        "tcp_availability_"
        + args.stage
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    meta = dict(
        status="running",
        family="persistent_availability",
        protocol_revision=2,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        conditions={
            "persistent_delayed": dict(telemetry_mode="stale_noise", policies=list(POLICIES)),
            "persistent_fast": dict(telemetry_mode="stale_noise", policies=list(fast_policies)),
            "healthy": dict(telemetry_mode="exact", policies=["F+P", "availability"]),
        },
        modes={k: asdict(TELEMETRY_MODES[k]) for k in ("exact", "stale_noise")},
        planned_runs=len(cases) * len(args.workers) * args.repeats,
        primary_contrast="availability - best fixed action per trace, persistent_delayed",
        primary_definition="minimum observed reached fixed T, only when all other fixed T are known or censored lower bounds exceed that minimum",
        source_sha256=source_hashes(),
        network_profiles={k: p.to_dict() for k, p in PROFILES.items()},
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "matplotlib", "scikit-learn", "cryptography")
        },
        observation_definition="causal sensor rates and previous protocol errors; regime and injected Fault stay outside planner",
        time_definition="physical localhost T_epsilon; startup/initial validation/warmup excluded; earlier telemetry, selection, validation and journal included",
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
                    traces = {
                        condition: make_trace(
                            args.seed + repeat,
                            args.rounds,
                            count,
                            condition,
                            args.phase_min,
                            args.phase_max,
                        )
                        for condition in meta["conditions"]
                    }
                    for condition, trace in traces.items():
                        (folder / f"trace_w{count}_r{repeat}_{condition}.json").write_text(
                            json.dumps([asdict(t) for t in trace], indent=2)
                        )
                    order = np.random.default_rng(args.seed + 20000 + repeat).permutation(
                        len(cases)
                    )
                    for position, index in enumerate(order):
                        condition, policy = cases[int(index)]
                        config = argparse.Namespace(
                            **{
                                **vars(args),
                                "telemetry_mode": meta["conditions"][condition]["telemetry_mode"],
                                "journal_namespace": condition,
                            }
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
                            cluster,
                            policy,
                            traces[condition],
                            validation,
                            config,
                            repeat,
                            position,
                            folder,
                        )
                        for row in rr:
                            row["condition"] = condition
                        ss["condition"] = condition
                        rows.extend(rr)
                        summaries.append(ss)
                        append_csv(
                            folder
                            / f"rounds_w{count}_r{repeat}_{condition}_{policy.replace('+', '')}.csv",
                            rr,
                        )
                        print(
                            f"workers={count} repeat={repeat} condition={condition} policy={policy}: "
                            f"accepted={ss['accepted_updates']}/{ss['attempts']} T_epsilon={ss['t_epsilon_wall_s']}",
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
