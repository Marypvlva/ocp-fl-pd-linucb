"""Run and audit paired local OCP-FL policy comparisons."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
from fractions import Fraction

from .adaptive_scheduler import ACTIONS
from .ocp_analyze import validate
from .ocp_experiment import configured_fault, main as run_experiment, validate_config
from .paths import ROOT
from .tcp_closed_loop_pilot import append_csv

ROSTER_POLICIES = ("pd_guard", "pd_roster", "risk_roster")
EXTENDED_ROSTER_POLICIES = (*ROSTER_POLICIES, "benefit_roster")


def summarize(rows, affected_clients=None):
    cases = defaultdict(list)
    for row in rows:
        cases[row["run"]].append(row)
    result = []
    for case, records in cases.items():
        population = records[0]["population"]
        counts = [[0, 0] for _ in range(population)]
        for row in records:
            for client in row["certificate"]["selected"]:
                counts[client][0] += 1
                counts[client][1] += row["accepted"]
        group_ids = (affected_clients or {}).get(
            str(records[0]["model_seed"]), range(population // 3)
        )
        targeted = set(group_ids)
        for label, group in (
            ("targeted", [x for i, x in enumerate(counts) if i in targeted]),
            ("other", [x for i, x in enumerate(counts) if i not in targeted]),
        ):
            invited = sum(x[0] for x in group)
            contributed = sum(x[1] for x in group)
            result.append(
                dict(
                    run=case,
                    group=label,
                    invitations=invited,
                    contributions=contributed,
                    rate=contributed / invited if invited else None,
                )
            )
    return result


def control_schedules(config):
    """Freeze cost-matched mixtures without consulting rosters or outcomes."""
    horizon = config["rounds"]
    cheap = Fraction(str(config["action_costs"]["F+P"]))
    reserve = Fraction(str(config["action_costs"]["R+P"]))
    budget = Fraction(str(config["budget_credits"]))
    if reserve <= cheap or budget < horizon * cheap:
        raise ValueError("Scheduled controls require affordable F+P and more costly R+P")
    count = min(horizon, int((budget - horizon * cheap) // (reserve - cheap)))
    spread = [
        "R+P" if (i + 1) * count // horizon > i * count // horizon else "F+P"
        for i in range(horizon)
    ]
    schedules = {"spread": spread}
    for seed in (1729, 1730):
        actions = ["R+P"] * count + ["F+P"] * (horizon - count)
        Random(seed).shuffle(actions)
        schedules[f"shuffle_{seed}"] = actions
    return schedules


def policy_config(base, policy, schedules, restricted_actions):
    """Build the same frozen policy configuration for execution and reporting."""
    config = base.copy()
    if policy not in ("pd", "pd_projection", "pd_guard", "pd_roster"):
        config["pd_eta"] = 0
    if policy in ("pd_projection", "linucb_projection"):
        if restricted_actions is None:
            raise ValueError("Restricted policy requires an explicit action set")
        config["available_actions"] = list(restricted_actions)
    if policy in ("reactive_edge", "reactive_any"):
        config["available_actions"] = ["F+P", "R+P"]
        config["selection_rule"] = policy
    if policy in EXTENDED_ROSTER_POLICIES:
        config["available_actions"] = ["F+P", "R+P"]
        config["roster_context"] = policy != "pd_guard"
        config["fund_horizon"] = policy in ("pd_guard", "pd_roster")
        if policy in ("risk_roster", "benefit_roster"):
            config["selection_rule"] = policy
    if policy in ACTIONS:
        config["fixed_action"] = policy
    if policy in schedules:
        config["action_schedule"] = schedules[policy]
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tcp_ocp_paired_pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed policy runs")
    parser.add_argument("--scheduled-controls", action="store_true")
    parser.add_argument("--restricted-bandits", action="store_true")
    parser.add_argument("--reactive-controls", action="store_true")
    parser.add_argument("--roster-controls", action="store_true")
    parser.add_argument("--policies", nargs="+", help="Run only these named comparison policies")
    args = parser.parse_args(argv)
    base = json.loads(args.config.read_text())
    validate_config(base)
    if base["budget_credits"] < max(base["action_costs"].values()):
        parser.error("The budget must cover at least one round in each fixed mode")
    if (
        base["pd_eta"] <= 0
        or "fixed_action" in base
        or "action_schedule" in base
        or base.get("roster_context")
        or base.get("fund_horizon")
        or base.get("selection_rule", "ucb") != "ucb"
    ):
        parser.error("Comparison requires a positive PD learning rate and no fixed action")
    # 2050-01-01 UTC keeps PyEVM on virtual time and allows exact roster replay.
    base.setdefault("trace_genesis", 2524608000)
    base_path = args.output / "base_config.json"
    if args.resume:
        if json.loads(base_path.read_text()) != base:
            parser.error("Resume requires the original comparison configuration")
    else:
        args.output.mkdir(parents=True)
        base_path.write_text(json.dumps(base, indent=2) + "\n")
    has_controls = args.scheduled_controls or (args.output / "control_schedules.json").exists()
    schedules = control_schedules(base) if has_controls else {}
    if schedules:
        manifest = args.output / "control_schedules.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != schedules:
                raise ValueError("Precommitted schedules changed")
        else:
            manifest.write_text(json.dumps(schedules, indent=2) + "\n")
    restricted_path = args.output / "restricted_bandits.json"
    restricted = args.restricted_bandits or restricted_path.exists()
    if restricted:
        available = ["F+P", "R+P"]
        if restricted_path.exists():
            if json.loads(restricted_path.read_text()) != available:
                raise ValueError("Restricted action set changed")
        else:
            restricted_path.write_text(json.dumps(available) + "\n")
    bandits = ("pd_projection", "linucb_projection") if restricted else ()
    reactive_path = args.output / "reactive_controls.json"
    reactive = (
        ("reactive_edge", "reactive_any")
        if args.reactive_controls or reactive_path.exists()
        else ()
    )
    if reactive:
        if reactive_path.exists() and json.loads(reactive_path.read_text()) != list(reactive):
            raise ValueError("Reactive control set changed")
        reactive_path.write_text(json.dumps(reactive) + "\n")
    roster_path = args.output / "roster_controls.json"
    roster_controls = ()
    if roster_path.exists():
        roster_controls = tuple(json.loads(roster_path.read_text()))
        if roster_controls not in (ROSTER_POLICIES, EXTENDED_ROSTER_POLICIES):
            raise ValueError("Roster comparison policy set changed")
    elif args.roster_controls:
        roster_controls = EXTENDED_ROSTER_POLICIES
        roster_path.write_text(json.dumps(roster_controls) + "\n")
    candidates = ("pd", "linucb", *bandits, *ACTIONS, *schedules, *reactive, *roster_controls)
    selection_path = args.output / "selected_policies.json"
    saved = json.loads(selection_path.read_text()) if selection_path.exists() else None
    policies = args.policies if args.policies is not None else saved or list(candidates)
    if (
        not policies
        or len(set(policies)) != len(policies)
        or any(policy not in candidates for policy in policies)
    ):
        parser.error("Policies must be distinct enabled modes; enable the required control flags")
    if saved is not None and policies != saved:
        parser.error("Resume requires the original policy selection and order")
    if args.policies is not None and saved is None:
        selection_path.write_text(json.dumps(policies, indent=2) + "\n")
    traces = {}
    policy_rows = {}
    limits = {}
    summary_rows = []
    for policy in policies:
        config = policy_config(base, policy, schedules, available if restricted else None)
        config_path = args.output / f"{policy}.json"
        folder = args.output / policy
        if not folder.exists():
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            run_experiment(["--config", str(config_path), "--output", str(folder)])
        meta, summary, rows = validate(folder)
        if meta["config"] != config:
            raise ValueError(f"Unexpected configuration for {policy}")
        policy_rows[policy] = rows
        for case in summary:
            count = int(case["attempts"])
            run = case["run"]
            limits[run] = min(limits.get(run, count), count)
            summary_rows.append(dict(policy=policy, **case))
        for row in rows:
            selected = row["certificate"]["selected"]
            fault = configured_fault(
                config,
                row["condition"],
                row["attempt"],
                selected,
                row["population"],
                row["model_seed"],
            )
            key = row["run"], row["attempt"]
            value = selected, fault
            if traces.setdefault(key, value) != value:
                raise ValueError(f"Unpaired client roster or fault at {key}")

    participation = []
    for policy, rows in policy_rows.items():
        prefix = [row for row in rows if row["attempt"] <= limits[row["run"]]]
        for scope, records in (("budget", rows), ("common_prefix", prefix)):
            participation.extend(
                dict(policy=policy, scope=scope, **row)
                for row in summarize(records, base.get("affected_clients"))
            )
    tables = [("summary.csv", summary_rows)]
    if participation:
        tables.append(("participation.csv", participation))
    for name, records in tables:
        temporary = args.output / f"{name}.tmp"
        temporary.unlink(missing_ok=True)
        append_csv(temporary, records)
        temporary.replace(args.output / name)
    print(f"Audited all policies and matched their overlapping roster traces: {args.output}")


if __name__ == "__main__":
    main()
