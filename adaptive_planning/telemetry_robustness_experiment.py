"""Blocked paired experiment for noisy/delayed telemetry and multiple MLP seeds."""

from .paths import ROOT
from .cli import source_hashes
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import uuid

import numpy as np

from . import tcp_dynamic_experiment as dynamic
from .network_telemetry import TELEMETRY_MODES
from .tcp_closed_loop_pilot import append_csv, capture_sources
from .tcp_runtime import ACTIONS, Cluster, make_model
from .tcp_network import PROFILES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/tcp_telemetry_robustness.json"
    )
    parser.add_argument(
        "--modes", nargs="+", choices=tuple(TELEMETRY_MODES), default=list(TELEMETRY_MODES)
    )
    parser.add_argument("--control-policy", choices=ACTIONS, default="F+P")
    parser.add_argument(
        "--study", choices=("robustness", "alignment", "freshness"), default="robustness"
    )
    parser.add_argument("--output", type=Path)
    own, forwarded = parser.parse_known_args(argv)
    args = dynamic.parse_args(["--config", str(own.config), *forwarded])
    if len(set(own.modes)) != len(own.modes) or own.control_policy in args.policies:
        parser.error("Duplicate mode or duplicated shared control")
    required = {"network_online", "network_prior"}
    if own.study in ("alignment", "freshness"):
        required = (
            {"network_online", "network_prior", "network_aligned"}
            if own.study == "alignment"
            else {"network_freshness", "network_prior", "network_aligned"}
        )
        if set(own.modes) != {"exact", "stale_noise"} or own.control_policy != "F+P":
            parser.error("This protocol fixes exact/stale_noise and shared F+P")
    if "stale_noise" not in own.modes or set(args.policies) != required:
        parser.error(f"This protocol requires stale_noise and policies {sorted(required)}")
    cases = [(mode, policy) for mode in own.modes for policy in args.policies]
    cases.append(("control", own.control_policy))
    if len(cases) * (args.rounds + 1) * args.repeats >= 65536:
        parser.error("Too many rounds for one dynamic cluster")
    from .training import make_workers

    root = ROOT
    folder = own.output or root / "results" / (
        f"tcp_telemetry_{own.study}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    meta = dict(
        status="running",
        family=f"telemetry_{own.study}",
        protocol_revision=2,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        modes={mode: asdict(TELEMETRY_MODES[mode]) for mode in own.modes},
        network_profiles={name: profile.to_dict() for name, profile in PROFILES.items()},
        shared_control_policy=own.control_policy,
        shared_control_mode="exact",
        planned_runs=len(cases) * args.repeats * len(args.workers),
        primary_mode="stale_noise",
        primary_contrast={
            "alignment": "network_aligned - network_online",
            "freshness": "network_freshness - network_aligned",
            "robustness": "network_online - network_prior",
        }[own.study],
        primary_definition="within stale_noise, paired by initial model/physical trace; other contrasts exploratory",
        python=platform.python_version(),
        platform=platform.platform(),
        cpu_count=__import__("os").cpu_count(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "matplotlib", "scikit-learn", "cryptography")
        },
        source_sha256=source_hashes(),
        observation_definition="synthetic causal sensor report; lag in attempts, lognormal noise on positive magnitudes; zero unlimited flag retained",
        time_definition="physical localhost T_epsilon; includes earlier validation, telemetry, sensor CPU, journal; excludes startup and warmup",
        trace_index="attempt, profile changes only between rounds",
        control_definition="one actual fixed-policy run per model/trace, shared analytically across sensor comparisons; never duplicated as raw runs",
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
                    trace = dynamic.make_trace(
                        args.seed + repeat,
                        args.rounds,
                        count,
                        args.scenario,
                        args.phase_min,
                        args.phase_max,
                    )
                    (folder / f"trace_w{count}_r{repeat}.json").write_text(
                        json.dumps([asdict(s) for s in trace], indent=2)
                    )
                    order = np.random.default_rng(args.seed + 20000 + repeat).permutation(
                        len(cases)
                    )
                    for position, index in enumerate(order):
                        condition, policy = cases[int(index)]
                        config = argparse.Namespace(
                            **{
                                **vars(args),
                                "telemetry_mode": "exact" if condition == "control" else condition,
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
                        rows, summary = dynamic.run_policy(
                            cluster, policy, trace, validation, config, repeat, position, folder
                        )
                        summary["condition"] = condition
                        for row in rows:
                            row["condition"] = condition
                        all_rows.extend(rows)
                        summaries.append(summary)
                        append_csv(
                            folder
                            / f"rounds_w{count}_r{repeat}_{condition}_{policy.replace('+', '')}.csv",
                            rows,
                        )
                        print(
                            f"workers={count} repeat={repeat} model_seed={summary['model_seed']} "
                            f"condition={condition} policy={policy}: accepted="
                            f"{summary['accepted_updates']}/{summary['attempts']} "
                            f"T_epsilon={summary['t_epsilon_wall_s']} loss={summary['final_loss']:.6f}",
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
