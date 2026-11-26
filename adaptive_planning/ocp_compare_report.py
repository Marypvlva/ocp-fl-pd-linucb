"""Audit matched training runs and report quality, execution credits and EVM gas."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from .ocp_analyze import validate
from .ocp_compare import (
    EXTENDED_ROSTER_POLICIES,
    ROSTER_POLICIES,
    control_schedules,
    policy_config,
    summarize,
)
from .adaptive_scheduler import ACTIONS
from .tcp_closed_loop_pilot import append_csv


def pareto_policies(cases):
    """Return reached policies not dominated in execution credits and gas."""
    reached = [case for case in cases if case["reached"]]
    frontier = []
    for case in reached:
        dominated = any(
            other["spent_credits"] <= case["spent_credits"]
            and other["gas_used"] <= case["gas_used"]
            and (
                other["spent_credits"] < case["spent_credits"]
                or other["gas_used"] < case["gas_used"]
            )
            for other in reached
        )
        if not dominated:
            frontier.append(case["policy"])
    return frontier


def audit_comparison(folder, *, require_target=True):
    base = json.loads((folder / "base_config.json").read_text())
    if require_target and base.get("stop_at_target") is not True:
        raise ValueError("Resource-to-quality reports require stop_at_target=true")
    schedule_path = folder / "control_schedules.json"
    schedules = json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
    restricted_path = folder / "restricted_bandits.json"
    restricted = json.loads(restricted_path.read_text()) if restricted_path.exists() else None
    if schedules and schedules != control_schedules(base):
        raise ValueError("Control schedules differ from the frozen comparison recipe")
    if restricted is not None and restricted != ["F+P", "R+P"]:
        raise ValueError("Unexpected restricted action set")
    bandits = ("pd_projection", "linucb_projection") if restricted is not None else ()
    reactive_path = folder / "reactive_controls.json"
    reactive = json.loads(reactive_path.read_text()) if reactive_path.exists() else []
    if reactive and reactive != ["reactive_edge", "reactive_any"]:
        raise ValueError("Unexpected reactive control set")
    roster_path = folder / "roster_controls.json"
    roster = json.loads(roster_path.read_text()) if roster_path.exists() else []
    if roster and tuple(roster) not in (ROSTER_POLICIES, EXTENDED_ROSTER_POLICIES):
        raise ValueError("Unexpected roster comparison policy set")
    candidates = ("pd", "linucb", *bandits, *ACTIONS, *schedules, *reactive, *roster)
    selection_path = folder / "selected_policies.json"
    policies = (
        json.loads(selection_path.read_text()) if selection_path.exists() else list(candidates)
    )
    if (
        not isinstance(policies, list)
        or not policies
        or any(type(policy) is not str for policy in policies)
        or len(set(policies)) != len(policies)
        or not set(policies) <= set(candidates)
    ):
        raise ValueError("Expected distinct enabled policies in the comparison selection")
    traces = {}
    cases = []
    for policy in policies:
        metadata, summaries, rows = validate(folder / policy)
        if metadata["config"] != policy_config(base, policy, schedules, restricted):
            raise ValueError(f"Unexpected comparison configuration for {policy}")
        by_run = defaultdict(list)
        for row in rows:
            key = row["run"], row["attempt"]
            selected = row["certificate"]["selected"]
            if traces.setdefault(key, selected) != selected:
                raise ValueError(f"Unpaired roster at {key}")
            by_run[row["run"]].append(row)
        participation = {
            (p["run"], p["group"]): p for p in summarize(rows, base.get("affected_clients"))
        }
        for summary in summaries:
            run_rows = by_run[summary["run"]]
            missing = sum(r["metrics"]["telemetry_missing_nodes"] for r in run_rows)
            cases.append(
                dict(
                    policy=policy,
                    run=summary["run"],
                    model_seed=int(summary["model_seed"]),
                    target_loss=float(base["target_loss"]),
                    reached=bool(int(summary["reached"])),
                    attempts=int(summary["attempts"]),
                    accepted_updates=int(summary["accepted_updates"]),
                    stop_reason=summary["stop_reason"],
                    spent_credits=float(summary["spent_credits"]),
                    gas_used=int(summary["gas_used"]),
                    final_loss=float(summary["final_loss"]),
                    t_epsilon_wall_s=(
                        float(summary["t_epsilon_wall_s"]) if int(summary["reached"]) else None
                    ),
                    app_tx_bytes=(
                        sum(r["metrics"]["app_tx_bytes"] for r in run_rows) if not missing else None
                    ),
                    bls_cache_policy=metadata.get("bls_cache_policy", "shared_process"),
                )
            )
            for group in ("targeted", "other"):
                counts = participation.get((summary["run"], group), {})
                cases[-1][f"{group}_invitations"] = counts.get("invitations", 0)
                cases[-1][f"{group}_contributions"] = counts.get("contributions", 0)
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Report output must be a new directory")
    cases = audit_comparison(args.folder)
    by_policy = defaultdict(list)
    by_run = defaultdict(list)
    for case in cases:
        by_policy[case["policy"]].append(case)
        by_run[case["run"]].append(case)
    policy_summary = []
    for policy, records in by_policy.items():
        complete = all(r["reached"] for r in records)
        policy_summary.append(
            dict(
                policy=policy,
                runs=len(records),
                reached=sum(r["reached"] for r in records),
                credits_at_target_mean=(
                    mean(r["spent_credits"] for r in records) if complete else None
                ),
                gas_at_target_mean=(mean(r["gas_used"] for r in records) if complete else None),
            )
        )
    result = dict(
        input_folder=str(args.folder.resolve()),
        audit_status="passed",
        target_loss=cases[0]["target_loss"],
        quality_reached=sum(case["reached"] for case in cases),
        runs=len(cases),
        rounds=sum(r["attempts"] for r in cases),
        summaries=policy_summary,
        pareto_by_run={run: pareto_policies(records) for run, records in by_run.items()},
        timing="Recorded per run; cache policy explicit; sequential runs do not establish speed ranking",
    )
    lines = [
        "# Сравнение ресурсов до общего качества",
        "",
        f"Порог validation loss {result['target_loss']}. Число прогонов — {result['runs']}, число раундов — {result['rounds']}.",
        "",
        "Недостижение порога указано отдельно. Средний расход до качества выводится, когда все прогоны политики достигли порога.",
        "",
        "| Политика | Достигли | Средние кредиты до качества | Средний gas до качества, млн |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in policy_summary:
        credits = (
            f"{row['credits_at_target_mean']:.2f}"
            if row["credits_at_target_mean"] is not None
            else "—"
        )
        gas = (
            f"{row['gas_at_target_mean'] / 1e6:.2f}"
            if row["gas_at_target_mean"] is not None
            else "—"
        )
        lines.append(f"| {row['policy']} | {row['reached']}/{row['runs']} | {credits} | {gas} |")
    lines += [
        "",
        "| Политика | Инициализация | Достигли | Потрачено кредитов | Gas, млн |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for case in cases:
        reached = "да" if case["reached"] else "нет"
        lines.append(
            f"| {case['policy']} | {case['model_seed']} | {reached} | "
            f"{case['spent_credits']:.2f} | {case['gas_used'] / 1e6:.2f} |"
        )
    failures = [case for case in cases if not case["reached"]]
    lines.append("")
    if failures:
        lines.append("Порог не достигнут в следующих прогонах.")
        lines.append("")
        for case in failures:
            lines.append(
                f"- {case['policy']}, {case['run']} — {case['stop_reason']}, "
                f"loss {case['final_loss']:.6f}"
            )
    else:
        lines.append("Все прогоны достигли порога качества.")
    lines += [
        "",
        "Фронт Парето включает достигшие порога политики, для которых нет варианта с расходом кредитов и gas одновременно не выше и хотя бы по одному ресурсу строго ниже.",
        "",
    ]
    for run, frontier in result["pareto_by_run"].items():
        lines.append(f"- {run} — {', '.join(frontier) or 'порог не достигнут'}")
    args.output.mkdir(parents=True)
    append_csv(args.output / "cases.csv", cases)
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    # Write the completed audit marker after its report artifacts.
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited {len(cases)} runs; report saved to {args.output}")


if __name__ == "__main__":
    main()
