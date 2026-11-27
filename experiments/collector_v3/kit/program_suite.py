"""Plan finite scenario families; execute only when the operator explicitly calls run."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from program import ROOT, freeze, read_plan
from distributed_live import dump
from design_tools import grid

FAMILIES = ("C0", "E1", "E2", "E2-protocol", "E3", "E4", "E5", "E6", "E7", "E7-article", "E8")


def template(name):
    return json.loads((ROOT / "configs/collector_program" / f"{name}.json").read_text())


def cases(family):
    natural = template("natural")
    switch = template("switch")
    cpu = template("heterogeneous")

    def case(name, scenario, modes, overrides=None):
        return dict(id=name, scenario=scenario, modes=modes, policy_overrides=overrides)

    if family == "C0":
        return [case("actions", template("probe"), ["uniform"])]
    if family == "E1":
        return [
            case("B" + name, s, ["joint", "no_dual", "legacy_lp_mean"])
            for name, s in grid(natural, "budget").items()
        ]
    if family == "E2":
        return [
            case(
                "W" + "-".join(map(str, windows)),
                dict(natural, id="proof_windows", windows_ms=windows),
                ["joint", "no_roster", "legacy_lp_mean"],
            )
            for windows in [[27, 35, 46], [41, 53, 69], [80, 130, 200]]
        ]
    if family == "E2-protocol":
        return [
            case(
                "T" + str(deadline),
                dict(natural, id="protocol_deadline", deadline_s=deadline),
                ["joint", "article_pd", "legacy_lp_mean"],
            )
            for deadline in [0.10, 0.13, 0.16]
        ]
    if family == "E3":
        return [
            case(name, s, ["joint", "matched_recent", "legacy_lp_mean"])
            for name, s in [
                ("proof_switch", switch),
                ("cpu_switch", cpu),
                ("random_proof", template("random_outages")),
            ]
        ]
    if family == "E4":
        stationary = deepcopy(cpu)
        stationary["id"] = "stationary_CPU_heterogeneity"
        stationary["events"] = [dict(start=1, end=120, cpu_iterations=[1000, 0, 1000, 0])]
        return [
            case(name, s, ["joint", "no_roster", "matched_recent", "static_rank"])
            for name, s in [("stationary", stationary), ("switch", cpu)]
        ]
    if family == "E5":
        return [
            case(
                "bytes_windows_outage",
                dict(switch, budget_bytes=24000000, windows_ms=[27, 35, 46]),
                [
                    "joint",
                    "no_dual",
                    "no_exploration",
                    "matched_recent",
                    "legacy_lp_mean",
                    "article_pd",
                ],
            ),
            case(
                "booking_windows_cpu",
                dict(cpu, windows_ms=[27, 35, 46], mirror_fraction=0.1, site_fraction=0.1),
                [
                    "joint",
                    "no_dual",
                    "no_exploration",
                    "matched_recent",
                    "legacy_lp_mean",
                    "article_pd",
                ],
            ),
        ]
    if family == "E6":
        return [
            case("H" + name, s, ["joint", "legacy_lp_mean"])
            for name, s in grid(natural, "horizon").items()
        ]
    if family == "E7":
        return [
            case(
                "matched",
                dict(cpu, windows_ms=[27, 35, 46]),
                ["joint", "no_roster", "no_exploration", "no_site_price", "no_dual"],
            )
        ]
    if family == "E7-article":
        return [
            case(
                "literal_core",
                dict(switch, windows_ms=[27, 35, 46]),
                ["article_pd", "article_no_dual"],
            )
        ]
    if family == "E8":
        result = []
        configurations = {
            "joint": [
                dict(alpha=0.0, eta=0.08),
                dict(alpha=0.15, eta=0.2),
                dict(alpha=0.55, eta=0.08),
            ],
            "article_pd": [
                dict(alpha=0.15, eta=0.05),
                dict(alpha=1.0, eta=0.05),
                dict(alpha=1.0, eta=0.2),
            ],
            "legacy_lp_mean": [dict(lp_window=24), dict(lp_window=96), dict(lp_window=192)],
        }
        for mode, values in configurations.items():
            for i, value in enumerate(values):
                result.append(
                    case(
                        mode + "_" + str(i),
                        dict(cpu, windows_ms=[27, 35, 46]),
                        [mode],
                        {mode: value},
                    )
                )
        return result
    raise ValueError("Unknown experiment family")


def steps(suite):
    for seed in suite["seeds"]:
        order = np.random.default_rng(seed + 99173).permutation(len(suite["cells"]))
        for index in order:
            yield seed, suite["cells"][int(index)]


def plan_suite(base, out, family, phase, seeds, chosen=None, exclude=(), hyperparameters=None):
    if family == "C0" and phase != "diagnostic":
        raise ValueError("C0 is diagnostic only")
    if phase == "confirmatory":
        raise ValueError(
            "Freeze one primary confirmation via program.py after a reviewed power/design decision, not a whole grid"
        )
    forbidden = []
    excluded = []
    for path in exclude:
        previous, digest = read_plan(path)
        forbidden += previous["seeds"]
        excluded.append(digest)
    selected = cases(family)
    if chosen:
        names = {c["id"] for c in selected}
        if not set(chosen) <= names:
            raise ValueError("Unknown selected scenario")
        selected = [c for c in selected if c["id"] in chosen]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for cell in selected:
        freeze(
            base,
            cell["scenario"],
            out / cell["id"],
            phase,
            seeds,
            cell["modes"],
            forbidden,
            hyperparameters,
            cell["policy_overrides"],
        )
        cell["plan_sha256"] = (out / cell["id"] / "plan.sha256").read_text().strip()
    suite = dict(
        version=1,
        family=family,
        phase=phase,
        seeds=seeds,
        cells=selected,
        excluded_plan_digests=excluded,
        new_rounds=sum(
            c["scenario"].get("horizon", 120) * len(c["modes"]) * len(seeds) for c in selected
        ),
        order_rule="seed99173_randomized_scenarios_within_seed; policies randomized within each scenario",
        created=time.time(),
    )
    episodes = sum(len(c["modes"]) * len(seeds) for c in selected)
    suite["resource_estimate"] = dict(
        episodes=episodes,
        planning_wall_hours_low=(2 * suite["new_rounds"] + 10 * episodes) / 3600,
        planning_wall_hours_high=(6 * suite["new_rounds"] + 10 * episodes) / 3600,
        result_export_bytes_scaled=100000 * suite["new_rounds"],
        cloud_cost_formula="wall_hours * (coordinator_hourly + peerA_hourly + peerB_hourly) + idle/storage/egress",
        caution="Planning assumptions 2..6 seconds/round and 10 seconds/episode, not confidence bounds. Previous measured mean was 1.72..1.75 s/round. Export estimate excludes SQLite, WAL, remote duplicates and backups.",
    )
    dump(out / "suite.json", suite)
    (out / "suite.sha256").write_text(sha256((out / "suite.json").read_bytes()).hexdigest() + "\n")
    return suite


def load_suite(out):
    out = Path(out)
    raw = (out / "suite.json").read_bytes()
    if sha256(raw).hexdigest() != (out / "suite.sha256").read_text().strip():
        raise ValueError("Suite digest mismatch")
    suite = json.loads(raw)
    for cell in suite["cells"]:
        _, digest = read_plan(out / cell["id"] / "plan.json")
        if digest != cell["plan_sha256"]:
            raise ValueError("Scenario plan changed")
    return suite


def run_suite(out, retry=False):
    out = Path(out).resolve()
    suite = load_suite(out)
    lock = out / "RUNNING"
    with lock.open("x") as stream:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), stream)
    try:
        for seed, cell in steps(suite):
            diag = ROOT / "deploy/timeweb/state/diagnostics.json"
            if not diag.exists() or time.time() - json.loads(diag.read_text())["at"] > 3300:
                subprocess.run(
                    [
                        str(ROOT / ".venv/bin/python"),
                        str(ROOT / "deploy/timeweb/manage.py"),
                        "diagnose",
                    ],
                    check=True,
                )
            args = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "deploy/timeweb/manage.py"),
                "run",
                "--out",
                str(out / cell["id"]),
                "--seed-block",
                str(seed),
            ]
            if retry:
                args.append("--retry-incomplete")
            dump(
                out / "progress.json",
                dict(seed=seed, cell=cell["id"], stage="running", at=time.time()),
            )
            subprocess.run(args, check=True)
        dump(out / "progress.json", dict(stage="completed", at=time.time()))
    except BaseException as exc:
        dump(
            out / "suite-failure.json",
            dict(type=type(exc).__name__, message=str(exc)[:300], at=time.time()),
        )
        raise
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--base", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--family", choices=FAMILIES, required=True)
    q.add_argument("--phase", choices=["diagnostic", "pilot", "exploratory"], required=True)
    q.add_argument("--seeds", type=int, nargs="+", required=True)
    q.add_argument("--cells", nargs="+")
    q.add_argument("--exclude-plans", type=Path, nargs="*", default=[])
    q.add_argument("--hyperparameters", type=Path)
    q = sub.add_parser("run")
    q.add_argument("directory", type=Path)
    q.add_argument("--retry-incomplete", action="store_true")
    q = sub.add_parser("analyze")
    q.add_argument("directory", type=Path)
    q.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "plan":
        s = plan_suite(
            a.base,
            a.out,
            a.family,
            a.phase,
            a.seeds,
            a.cells,
            a.exclude_plans,
            json.loads(a.hyperparameters.read_text()) if a.hyperparameters else None,
        )
        print(
            json.dumps(
                dict(
                    family=s["family"],
                    cells=len(s["cells"]),
                    new_rounds=s["new_rounds"],
                    resource_estimate=s["resource_estimate"],
                ),
                indent=2,
            )
        )
    elif a.command == "run":
        run_suite(a.directory, a.retry_incomplete)
    else:
        from analysis_program import analyze

        suite = load_suite(a.directory)
        result = analyze([(c["id"], a.directory / c["id"]) for c in suite["cells"]], a.out)
        if not result["complete"] or not result["no_retries"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
