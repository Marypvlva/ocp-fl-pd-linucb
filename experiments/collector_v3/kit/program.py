"""Freeze scientific extensions without deploying or starting any cloud experiment."""

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from model import Settings

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def read_plan(path):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != Path(path).with_name("plan.sha256").read_text().strip():
        raise ValueError("Original plan digest mismatch")
    return json.loads(raw), sha256(raw).hexdigest()


def policies(calibration):
    base = Settings(**calibration["records"]["joint"]["chosen"]["settings"])
    result = dict(
        joint=base,
        no_exploration=replace(base, alpha=0.0),
        no_dual=replace(base, price_rule="zero"),
        no_site_price=replace(base, price_rule="global"),
        no_roster=replace(base, use_current_roster=False),
        matched_recent=replace(base, mode="recent"),
        uniform=replace(base, mode="uniform", price_rule="zero"),
        static_rank=replace(base, mode="static_rank", price_rule="zero", allowed_arms=(0, 1, 2, 3)),
    )
    for mode, record in calibration["records"].items():
        result["legacy_" + mode] = Settings(**record["chosen"]["settings"])
    result["article_pd"] = Settings(
        mode="article_pd", alpha=1.0, ridge=1.0, eta=0.05, initial_price=0.0, delay=base.delay
    )
    result["article_no_dual"] = replace(result["article_pd"], eta=0.0)
    # Memory-only variants deliberately retain v0.7.0 censoring for causal ablation.
    result["joint_discounted"] = replace(base, memory="discount")
    result["joint_window"] = replace(base, memory="window")
    result["joint_discounted_legacy_bonus"] = replace(
        result["joint_discounted"], memory_bonus="legacy")
    # Separate feedback correction; never attribute its effect to forgetting.
    for name in ("joint", "joint_discounted", "joint_window", "matched_recent"):
        result[name + "_eligible"] = replace(result[name], feedback_rule="eligible")
    return {name: settings.__dict__ for name, settings in result.items()}


def payload_bound(k=2, hidden=64):
    # Every accepted F+P update needs k model payloads, k full gradients split
    # over two lanes, and two aggregate halves. Headers/proofs only add bytes.
    parameters = 64 * hidden + hidden + hidden * 10 + 10
    return (2 * k + 1) * parameters * 8


def feasibility(config):
    h = config["horizon"]
    minimum = payload_bound(config["k"])
    return dict(
        necessary_payload_bytes_per_accepted=minimum,
        all_rounds_payload_lower_bound=h * minimum,
        accepted_payload_upper_bound=min(h, config["bytes_per_episode"] // minimum),
        all_success_necessarily_infeasible=config["bytes_per_episode"] < h * minimum,
        interpretation="Necessary bound for current F+P payloads, not sufficient feasibility or a measured cost. Rejections consume bytes too; HTTPS/EVM traffic is outside this application-byte boundary.",
    )


def validate_scenario(scenario):
    allowed = {
        "id",
        "horizon",
        "budget_bytes",
        "mirror_fraction",
        "site_fraction",
        "windows_ms",
        "deadline_s",
        "end_to_end_deadline_s",
        "events",
        "random_outage_probability",
        "balanced_probe",
        "network_mode",
        "primary_metric",
        "feedback_delay",
    }
    if set(scenario) - allowed:
        raise ValueError("Unknown scenario fields: " + str(set(scenario) - allowed))
    h = scenario.get("horizon", 120)
    if (
        type(scenario.get("feedback_delay", 2)) is not int
        or not 0 <= scenario.get("feedback_delay", 2) <= 20
    ):
        raise ValueError("Feedback delay must be common and within 0..20 rounds")
    if type(h) is not int or not 1 <= h <= 1000:
        raise ValueError("Horizon must be 1..1000")
    for key in ("mirror_fraction", "site_fraction", "random_outage_probability"):
        x = scenario.get(key, 0)
        if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError("Invalid fraction: " + key)
    windows = scenario.get("windows_ms")
    if windows is not None and (
        len(windows) != 3
        or windows != sorted(set(windows))
        or not all(type(x) is int and 0 < x <= 60000 for x in windows)
    ):
        raise ValueError("Three increasing integer proof windows required")
    for key, maximum in [("deadline_s", 60), ("end_to_end_deadline_s", 3600)]:
        value = scenario.get(key, 3 if key == "deadline_s" else 600)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("Invalid deadline: " + key)
    if "budget_bytes" in scenario and (
        type(scenario["budget_bytes"]) is not int or scenario["budget_bytes"] <= 0
    ):
        raise ValueError("Positive integer byte budget required")
    controlled = bool(scenario.get("events") or scenario.get("random_outage_probability"))
    if controlled and scenario.get("network_mode") != "controlled":
        raise ValueError("Workloads/outages require explicit controlled mode")
    for event in scenario.get("events", []):
        if set(event) - {
            "start",
            "end",
            "failed_edges",
            "unavailable_collectors",
            "cpu_iterations",
            "service_ms",
        }:
            raise ValueError("Unknown event field")
        if not (
            type(event["start"]) is int
            and type(event["end"]) is int
            and 1 <= event["start"] <= event["end"] <= h
        ):
            raise ValueError("Invalid event range")
        if any(type(g) is not int or not 0 <= g < 4 for g in event.get("failed_edges", [])):
            raise ValueError("Invalid failed collector")
        if any(
            type(g) is not int or not 0 <= g < 4 for g in event.get("unavailable_collectors", [])
        ):
            raise ValueError("Invalid unavailable proof service")
        for key, cap in [("cpu_iterations", 100000), ("service_ms", 60000)]:
            values = event.get(key, [0] * 4)
            if len(values) != 4 or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or not 0 <= v <= cap
                or (key == "cpu_iterations" and type(v) is not int)
                for v in values
            ):
                raise ValueError("Invalid " + key)


def effects(scenario, seed):
    """Independent frozen exogenous schedule. Never handed to Policy.decide."""
    h = scenario.get("horizon", 120)
    rng = np.random.default_rng(seed + 917301)
    rows = []
    for rd in range(1, h + 1):
        row = dict(
            failed_edges=[], unavailable_collectors=[], cpu_iterations=[0] * 4, service_ms=[0.0] * 4
        )
        if rng.random() < scenario.get("random_outage_probability", 0):
            row["unavailable_collectors"] = [int(rng.integers(4))]
        for event in scenario.get("events", []):
            if event["start"] <= rd <= event["end"]:
                row["failed_edges"] = sorted(
                    set(row["failed_edges"] + event.get("failed_edges", []))
                )
                row["unavailable_collectors"] = sorted(
                    set(row["unavailable_collectors"] + event.get("unavailable_collectors", []))
                )
                for key in ("cpu_iterations", "service_ms"):
                    row[key] = [
                        a + b for a, b in zip(row[key], event.get(key, [0] * 4), strict=True)
                    ]
        if max(row["cpu_iterations"]) > 100000 or max(row["service_ms"]) > 60000:
            raise ValueError("Overlapping events exceed workload cap")
        rows.append(row)
    return rows


def freeze(
    base_path,
    scenario,
    output,
    phase,
    seeds,
    modes,
    forbidden_seeds=(),
    hyperparameters=None,
    policy_overrides=None,
):
    from distributed_live import dump, sources

    validate_scenario(scenario)
    base, parent = read_plan(base_path)
    if hyperparameters:
        if set(hyperparameters) - {"alpha", "eta", "ridge", "initial_price"}:
            raise ValueError("Unknown hyperparameter")
        for key, value in hyperparameters.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (key == "ridge" and value <= 0)
                or (key == "delay" and type(value) is not int)
            ):
                raise ValueError("Invalid hyperparameter: " + key)
        base["calibration"]["records"]["joint"]["chosen"]["settings"].update(hyperparameters)
    if phase not in ("diagnostic", "pilot", "exploratory", "confirmatory"):
        raise ValueError("Unknown research phase")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or not all(type(s) is int and s > 0 for s in seeds)
    ):
        raise ValueError("Unique positive seed blocks required")
    domain = {"diagnostic": 901000, "pilot": 902000, "exploratory": 903000, "confirmatory": 904000}[
        phase
    ]
    if not all(domain <= s < domain + 1000 for s in seeds):
        raise ValueError("Use disjoint phase seed namespaces 901xxx/902xxx/903xxx/904xxx")
    if phase == "confirmatory" and not forbidden_seeds:
        raise ValueError(
            "Confirmatory freeze requires --exclude-plans for prior development/pilot plans"
        )
    if set(seeds) & (set(base["seeds"]) | set(forbidden_seeds)):
        raise ValueError("Seed overlap with prior/development data")
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("Unique policy names required")
    spec = policies(base["calibration"])
    for name, values in (policy_overrides or {}).items():
        if name not in spec or set(values) - {
            "alpha",
            "eta",
            "ridge",
            "initial_price",
            "lp_window",
            "discount",
            "window",
        }:
            raise ValueError("Unknown policy/hyperparameter override")
        for key, value in values.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (key == "ridge" and value <= 0)
                or (key == "lp_window" and (type(value) is not int or value < 2))
                or (key == "window" and (type(value) is not int or value < 1))
                or (key == "discount" and not 0 < value <= 1)
            ):
                raise ValueError("Invalid policy hyperparameter")
        spec[name] = Settings(**dict(spec[name], **values)).__dict__
    for name in spec:
        spec[name]["delay"] = scenario.get("feedback_delay", base.get("reward_delay", 2))
    if set(modes) - set(spec):
        raise ValueError("Unknown policy")
    h = scenario.get("horizon", 120)
    if scenario.get("balanced_probe"):
        if phase != "diagnostic" or modes != ["uniform"]:
            raise ValueError("Balanced arm probe is diagnostic only and uses uniform policy")
        if scenario.get("mirror_fraction", 1) != 1 or scenario.get("site_fraction", 1) != 1:
            raise ValueError("Balanced probe requires unrestricted booking quotas")
    new = deepcopy(base)
    new.update(
        version=2,
        profile="three_vm_research",
        horizon=h,
        seeds=seeds,
        modes=modes,
        order={str(s): np.random.default_rng(s + 7).permutation(modes).tolist() for s in seeds},
        bytes_per_episode=scenario.get(
            "budget_bytes", int(base["bytes_per_episode"] * h / base["horizon"])
        ),
        mirror_limit=int(scenario.get("mirror_fraction", 0.25) * h),
        site_fraction=scenario.get("site_fraction", 0.2),
        policy_specs={m: spec[m] for m in modes},
        policy_tie_seed=53,
        window_classes={
            str(s): np.concatenate(
                [
                    np.random.default_rng(s + 2101 + chunk).permutation(3)
                    for chunk in range((h + 2) // 3)
                ]
            )[:h].tolist()
            for s in seeds
        },
        windows_ms=scenario.get("windows_ms", base["windows_ms"]),
        deadline_s=scenario.get("deadline_s", base["deadline_s"]),
        end_to_end_deadline_s=scenario.get("end_to_end_deadline_s", base["end_to_end_deadline_s"]),
        network_mode=scenario.get("network_mode", "natural"),
        chain_mode="local_evm",
        source_hashes=sources(ROOT),
        commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        created=time.time(),
        parent_plan_sha256=parent,
        external_config_digest=None,
        program=dict(
            phase=phase,
            scenario=scenario,
            history="empty_per_episode",
            hyperparameter_override=hyperparameters,
            policy_overrides=policy_overrides,
            seed_exclusions=sorted(set(base["seeds"]) | set(forbidden_seeds)),
            primary_metric=scenario.get("primary_metric", "accepted"),
            inference="paired_seed_time_block_conditional_on_three_VMs",
            retry_rule="preserve_all_attempts; conditional_completed_analysis_and_missing_bounds; no_confirmatory_claim_after_unplanned_retry",
        ),
        effects={str(s): effects(scenario, s) for s in seeds},
        feedback_schema=2,
    )
    if scenario.get("balanced_probe"):
        new["mirror_limit"] = h
        new["site_fraction"] = 1.0
        new["forced_arms"] = {
            str(s): np.random.default_rng(s + 51).permutation(np.resize(np.arange(10), h)).tolist()
            for s in seeds
        }
    new["feasibility"] = feasibility(new)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "plan.json", new)
    (output / "plan.sha256").write_text(
        sha256((output / "plan.json").read_bytes()).hexdigest() + "\n"
    )
    return new


def verify_complete(directory, config):
    """Do not silently trust a stale/tampered complete marker on resume."""
    complete = json.loads((directory / "complete.json").read_text())
    paths = sorted(directory.glob("round-*.json"))
    if complete["rounds"] != config["horizon"] or len(paths) != config["horizon"]:
        raise ValueError("Invalid completed episode length")
    if (
        complete.get("plan_sha256")
        != sha256((directory.parent / "plan.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Completed episode belongs to another plan")
    if complete.get("round_hashes") != {p.name: sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise ValueError("Completed episode changed or lacks v2 manifest")
    for rd, p in enumerate(paths, 1):
        row = json.loads(p.read_text())
        if row["round"] != rd or row.get("plan_sha256") != complete["plan_sha256"]:
            raise ValueError("Completed round/plan binding changed")
    audit = json.loads((directory / "ledger-audit.json").read_text())
    if audit["reserved"] or not 0 <= audit["spent"] <= config["bytes_per_episode"]:
        raise ValueError("Completed episode lacks valid resource audit")
    last = json.loads(paths[-1].read_text())
    if audit["spent"] != last["byte_ledger"]["spent"]:
        raise ValueError("Completed ledger/round mismatch")
    if (directory / "budget.sqlite").exists():
        from deploy.timeweb.audit_budget import audit as independent_audit

        current = independent_audit(directory / "budget.sqlite", immutable=True)
        if (
            not current["valid"]
            or current["spent"] != audit["spent"]
            or current["unsettled_frames"]
        ):
            raise ValueError("Completed SQLite changed or failed independent audit")


def unlock(output):
    lock = Path(output) / "RUNNING"
    record = json.loads(lock.read_text())
    if record.get("host") != socket.gethostname():
        raise ValueError("Lock belongs to a different host; no automatic recovery")
    try:
        os.kill(record["pid"], 0)
    except ProcessLookupError:
        lock.unlink()
        return
    raise ValueError("Recorded process still exists; stop it and preserve diagnostics first")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--base", type=Path, required=True)
    q.add_argument("--scenario", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument(
        "--phase", choices=["diagnostic", "pilot", "exploratory", "confirmatory"], required=True
    )
    q.add_argument("--seeds", nargs="+", type=int, required=True)
    q.add_argument("--policies", nargs="+", required=True)
    q.add_argument("--exclude-plans", nargs="*", type=Path, default=[])
    q.add_argument("--hyperparameters", type=Path)
    q.add_argument("--policy-overrides", type=Path)
    q = sub.add_parser("unlock")
    q.add_argument("directory", type=Path)
    q = sub.add_parser("bounds")
    q.add_argument("--rounds", type=int, default=120)
    a = p.parse_args()
    if a.command == "plan":
        forbidden = []
        for path in a.exclude_plans:
            old, _ = read_plan(path)
            forbidden.extend(old["seeds"])
        result = freeze(
            a.base,
            json.loads(a.scenario.read_text()),
            a.out,
            a.phase,
            a.seeds,
            a.policies,
            forbidden,
            json.loads(a.hyperparameters.read_text()) if a.hyperparameters else None,
            json.loads(a.policy_overrides.read_text()) if a.policy_overrides else None,
        )
        print(
            json.dumps(
                dict(path=str(a.out / "plan.json"), feasibility=result["feasibility"]), indent=2
            )
        )
    elif a.command == "unlock":
        unlock(a.directory)
    else:
        print(
            json.dumps(
                [
                    dict(
                        budget_bytes=b,
                        **feasibility(dict(horizon=a.rounds, k=2, bytes_per_episode=b)),
                    )
                    for b in [30000000, 27000000, 24000000, 21000000, 18000000, 15000000]
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
