"""Reproducible localhost TCP training experiment; see README.md for semantics."""

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
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
import uuid
import zipfile

import numpy as np

from .tcp_runtime import ACTIONS, Cluster, Fault, make_model
from .tcp_network import PROFILES

POLICIES = (*ACTIONS, "rule", "online")


class OnlinePlanner:
    """A deliberately modest bandit baseline, trained only on chosen actions.

    Context contains past outcomes, never the injected fault for the next round.
    No claim of superiority is built into this algorithm.
    """

    def __init__(self, seed, switch_s):
        self.rng = np.random.default_rng(seed)
        self.switch_s = switch_s
        self.stats = defaultdict(lambda: [0, 0, 0.0])
        self.total = {action: 0 for action in ACTIONS}

    def choose(self, observed, previous):
        untried = [a for a in ACTIONS if self.total[a] == 0]
        if untried:
            return untried[0]
        if self.rng.random() < 0.08:
            return str(self.rng.choice(ACTIONS))

        def score(action):
            n, successes, elapsed = self.stats[(observed, action)]
            p = (successes + 1) / (n + 2)
            duration = (elapsed + 0.2) / (n + 2)
            penalty = self.switch_s if previous is not None and previous != action else 0.0
            return p / (duration + penalty)

        return max(ACTIONS, key=score)

    def observe(self, observed, action, success, cycle_s, switch_wall_s):
        entry = self.stats[(observed, action)]
        entry[0] += 1
        entry[1] += int(success)
        entry[2] += max(1e-9, cycle_s - switch_wall_s)
        self.total[action] += 1


def simple_rule(observed, workers):
    failed, slow, projection_failed = observed
    placement = "R" if failed or slow else "F"
    method = "P" if workers >= 12 and not projection_failed else "B"
    return placement + "+" + method


def make_scenario(seed, rounds, workers, kind):
    rng = np.random.default_rng(seed)
    trace = []
    for _ in range(rounds):
        if kind == "clean":
            trace.append(Fault())
        else:
            failed = (0, 1) if rng.random() < 0.12 else ()
            corrupted = (int(rng.integers(0, 4)),) if rng.random() < 0.12 else ()
            missing = (int(rng.integers(0, workers)),) if rng.random() < 0.06 else ()
            delay = (0.03, 0.03, 0.0, 0.0) if rng.random() < 0.2 else (0.0, 0.0, 0.0, 0.0)
            trace.append(
                Fault(
                    failed_edges=failed,
                    corrupt_edges=corrupted,
                    missing_projection=missing,
                    edge_delay_s=delay,
                )
            )
    return trace


def append_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture_sources(folder, root=None):
    root = root or ROOT
    from .cli import source_files

    with zipfile.ZipFile(folder / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(root):
            archive.write(path, path.relative_to(root))


def run_policy(cluster, policy, trace, validation, config, repeat, order):
    model = make_model(config.model_seed, config.hidden)
    loss, _ = model.score(*validation)
    initial_loss = loss
    planner = OnlinePlanner(config.seed + 10_000 + repeat, config.switch_seconds)
    observed, previous = (False, False, False), None
    rows = []
    accepted = switches = traffic = 0
    start = perf_counter()
    reached_s = 0.0 if loss <= config.target_loss else None
    stopped_budget = False
    stop_reason = "max_rounds"
    for attempt, fault in enumerate(trace, 1):
        if reached_s is not None:
            stop_reason = "target_reached"
            break
        cycle_start = perf_counter()
        select_start = perf_counter()
        if policy in ACTIONS:
            action = policy
        elif policy == "rule":
            action = simple_rule(observed, len(cluster.groups))
        else:
            action = planner.choose(observed, previous)
        selection_s = perf_counter() - select_start
        switched = previous is not None and previous != action
        result = cluster.run_round(
            model,
            action,
            deadline_s=config.deadline,
            fault=fault,
            switched=switched,
            switch_s=config.switch_seconds,
        )
        accepted += int(result.accepted)
        switches += int(switched)
        score_start = perf_counter()
        # A rejected round cannot change the loss; avoid unnecessary validation work.
        if result.accepted:
            loss, _ = model.score(*validation)
        score_s = perf_counter() - score_start
        # Detection time includes the complete validation computation.
        quality_observed_s = perf_counter() - start
        if loss <= config.target_loss:
            reached_s = quality_observed_s
        telemetry_start = perf_counter()
        telemetry = cluster.collect_metrics(cluster.round_id)
        telemetry_s = perf_counter() - telemetry_start
        traffic += telemetry["app_tx_bytes"]
        cycle_s = perf_counter() - cycle_start
        if policy == "online":
            planner.observe(
                observed, action, result.accepted, cycle_s, result.metrics["switch_wall_s"]
            )
        row = dict(
            workers=len(cluster.groups),
            policy=policy,
            repeat=repeat,
            order=order,
            network_profile=cluster.network.name,
            seed=config.seed,
            model_seed=config.model_seed,
            scenario_seed=config.seed + 1000 + repeat,
            attempt=attempt,
            accepted_updates=accepted,
            validation_loss=loss,
            target_loss=config.target_loss,
            selection_wall_s=selection_s,
            validation_wall_s=score_s,
            telemetry_wall_s=telemetry_s,
            cycle_wall_s=cycle_s,
            quality_observed_wall_s=quality_observed_s,
            elapsed_wall_s=perf_counter() - start,
            dimension=cluster.dimension,
            scenario=config.scenario,
            observed_context=json.dumps(observed),
            injected_fault=json.dumps(asdict(fault), sort_keys=True),
            errors=json.dumps(result.errors, ensure_ascii=False),
            traffic_is_lower_bound=int(
                bool(result.metrics["timeouts"] or telemetry["telemetry_missing_nodes"])
            ),
            **result.metrics,
            **telemetry,
        )
        rows.append(row)
        observed = (
            not result.accepted,
            result.metrics["round_wall_s"] > 0.5 * config.deadline,
            action.endswith("P") and not result.accepted,
        )
        previous = action
        if result.metrics["update_bound_exceeded"]:
            stop_reason = "update_out_of_range"
            break  # same model/data would deterministically fail again
        if config.traffic_budget_mb and traffic >= config.traffic_budget_mb * 1e6:
            stopped_budget = True
            stop_reason = "traffic_budget"
            break
    if reached_s is not None:
        stop_reason = "target_reached"
    elapsed = perf_counter() - start
    summary = dict(
        workers=len(cluster.groups),
        policy=policy,
        repeat=repeat,
        order=order,
        network_profile=cluster.network.name,
        scenario=config.scenario,
        seed=config.seed,
        model_seed=config.model_seed,
        initial_loss=initial_loss,
        final_loss=loss,
        target_loss=config.target_loss,
        reached=int(reached_s is not None),
        t_epsilon_wall_s=reached_s,
        observed_wall_s=elapsed,
        attempts=len(rows),
        accepted_updates=accepted,
        success_fraction=accepted / len(rows) if rows else 1.0,
        switches=switches,
        app_tx_bytes=traffic,
        startup_wall_s=cluster.startup_s,
        candidate_rejections=sum(r["candidate_rejections"] for r in rows),
        verification_attempts=sum(r["verification_attempts"] for r in rows),
        timeout_count=sum(r["timeouts"] for r in rows),
        budget_stopped=int(stopped_budget),
        stop_reason=stop_reason,
        traffic_is_lower_bound=int(any(r["traffic_is_lower_bound"] for r in rows)),
    )
    return rows, summary


def paired_comparisons(summaries, seed):
    rng = np.random.default_rng(seed + 99_000)
    rows = []
    for workers in sorted({s["workers"] for s in summaries}):
        indexed = {(s["repeat"], s["policy"]): s for s in summaries if s["workers"] == workers}
        repeats = sorted({k[0] for k in indexed})
        for baseline in (*ACTIONS, "rule"):
            pairs = [
                (indexed[(i, "online")], indexed[(i, baseline)])
                for i in repeats
                if (i, "online") in indexed and (i, baseline) in indexed
            ]
            complete = [(a, b) for a, b in pairs if a["reached"] and b["reached"]]
            delta = np.array([a["t_epsilon_wall_s"] - b["t_epsilon_wall_s"] for a, b in complete])
            lo = hi = mean = None
            if len(delta):
                mean = float(delta.mean())
            if len(delta) >= 2:
                draws = rng.choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
                lo, hi = np.quantile(draws, [0.025, 0.975]).tolist()
            rows.append(
                dict(
                    workers=workers,
                    contrast=f"online - {baseline}",
                    pairs=len(pairs),
                    complete_pairs=len(delta),
                    online_reached=sum(a["reached"] for a, _ in pairs),
                    baseline_reached=sum(b["reached"] for _, b in pairs),
                    mean_delta_wall_s=mean,
                    bootstrap95_low_s=lo,
                    bootstrap95_high_s=hi,
                    interpretation="exploratory; complete pairs only; no multiple-testing correction",
                )
            )
    return rows


def plot_results(rows, summaries, folder):
    import matplotlib.pyplot as plt

    counts = sorted({r["workers"] for r in rows})
    if not counts:
        return
    fig, axes = plt.subplots(1, len(counts), figsize=(5 * len(counts), 4), squeeze=False)
    for ax, count in zip(axes[0], counts):
        for policy in POLICIES:
            group = [
                r
                for r in rows
                if r["workers"] == count and r["policy"] == policy and r["repeat"] == 0
            ]
            if group:
                ax.plot(
                    [r["quality_observed_wall_s"] for r in group],
                    [r["validation_loss"] for r in group],
                    marker=".",
                    label=policy,
                )
        ax.axhline(rows[0]["target_loss"], color="black", linestyle="--", linewidth=0.7)
        ax.set(
            title=f"{count} workers · repeat 0",
            xlabel="Elapsed localhost wall-clock (s)",
            ylabel="Validation loss",
        )
        ax.grid(alpha=0.2)
    axes[0][-1].legend()
    fig.tight_layout()
    fig.savefig(folder / "learning_wall_clock.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(counts), figsize=(5 * len(counts), 4), squeeze=False)
    for ax, count in zip(axes[0], counts):
        for i, policy in enumerate(POLICIES):
            completed = [
                s
                for s in summaries
                if s["workers"] == count and s["policy"] == policy and s["reached"]
            ]
            if completed:
                ax.scatter(
                    [i] * len(completed),
                    [s["t_epsilon_wall_s"] for s in completed],
                    s=24,
                    alpha=0.7,
                )
        ax.set_xticks(range(len(POLICIES)), POLICIES, rotation=35)
        ax.set(
            title=f"{count} workers · reached runs only",
            ylabel="T_epsilon localhost wall-clock (s)",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "time_to_quality.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[4])
    parser.add_argument("--policies", choices=POLICIES, nargs="+", default=list(POLICIES))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=670)
    parser.add_argument("--target-loss", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=2026,
        help="Initial MLP weights, independent of scenario/planner seed",
    )
    parser.add_argument("--deadline", type=float, default=5.0)
    parser.add_argument("--rpc-timeout", type=float, default=2.0)
    parser.add_argument("--switch-seconds", type=float, default=0.025)
    parser.add_argument("--scenario", choices=("clean", "faults"), default="clean")
    parser.add_argument("--network-profile", choices=tuple(PROFILES), default="unlimited")
    parser.add_argument("--traffic-budget-mb", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        settings = json.loads(pre.config.read_text())
        allowed = {action.dest for action in parser._actions} - {"config", "help"}
        if set(settings) - allowed:
            parser.error("Unknown config keys")
        parser.set_defaults(**settings)
    args = parser.parse_args(argv)
    if not all(2 <= w <= 16 for w in args.workers):
        parser.error("workers must be between 2 and 16")
    if (
        args.rounds < 1
        or args.repeats < 1
        or not 1 <= args.hidden <= 2000
        or not 0 < args.deadline <= 60
        or not 0 < args.rpc_timeout <= 30
        or not 0 <= args.switch_seconds <= 5
        or args.traffic_budget_mb < 0
        or not 0 < args.target_loss < 10
    ):
        parser.error("Invalid experiment limits")
    return args


def main(argv=None):
    args = parse_args(argv)
    from .training import make_workers

    folder = (
        Path(args.output)
        if args.output
        else ROOT
        / "results"
        / (
            "tcp_closed_loop_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
            + uuid.uuid4().hex[:8]
        )
    )
    folder.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        transport_version=2,
        network=PROFILES[args.network_profile].to_dict(),
        python=platform.python_version(),
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "matplotlib", "scikit-learn", "cryptography")
        },
        time_definition="localhost physical wall-clock; no WAN model; startup and initial validation excluded",
        traffic_definition="application frames sent across all roles; excludes TCP/IP and round-0 telemetry",
        source_sha256=source_hashes(),
        cryptographic_randomness="OS CSPRNG, never seeded; model/scenario/order/planner are seeded",
        status="running",
    )
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    capture_sources(folder)
    all_rows, summaries = [], []
    try:
        for count in args.workers:
            groups, validation, _ = make_workers(count)
            with Cluster(
                groups,
                hidden=args.hidden,
                rpc_timeout=args.rpc_timeout,
                network=PROFILES[args.network_profile],
            ) as cluster:
                for repeat in range(args.repeats):
                    trace = make_scenario(
                        args.seed + 1000 + repeat, args.rounds, count, args.scenario
                    )
                    (folder / f"scenario_w{count}_r{repeat}.json").write_text(
                        json.dumps([asdict(f) for f in trace], indent=2)
                    )
                    policies = np.random.default_rng(args.seed + 20_000 + repeat).permutation(
                        args.policies
                    )
                    for order, policy in enumerate(policies):
                        # Equal warmup, separate model; never contributes to T_epsilon or learning.
                        warm = cluster.run_round(
                            make_model(args.model_seed, args.hidden),
                            "F+B",
                            deadline_s=args.deadline,
                        )
                        if not warm.accepted:
                            raise RuntimeError(f"Warmup failed: {warm.errors}")
                        rows, summary = run_policy(
                            cluster, str(policy), trace, validation, args, repeat, order
                        )
                        all_rows.extend(rows)
                        summaries.append(summary)
                        append_csv(
                            folder / f"rounds_w{count}_r{repeat}_{policy.replace('+', '')}.csv",
                            rows,
                        )
                        print(
                            f"workers={count} repeat={repeat} policy={policy}: "
                            f"accepted={summary['accepted_updates']}/{summary['attempts']} "
                            f"loss={summary['initial_loss']:.6f}->{summary['final_loss']:.6f} "
                            f"T_epsilon={summary['t_epsilon_wall_s']} "
                            f"traffic={summary['app_tx_bytes'] / 1e6:.2f} MB",
                            flush=True,
                        )
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata["status"], metadata["error"] = "failed", repr(exc)
        raise
    finally:
        append_csv(folder / "rounds.csv", all_rows)
        append_csv(folder / "summary.csv", summaries)
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    append_csv(folder / "paired.csv", paired_comparisons(summaries, args.seed))
    if not args.no_plots:
        plot_results(all_rows, summaries, folder)
    print(f"Saved: {folder}", flush=True)
    return folder


if __name__ == "__main__":
    main()
