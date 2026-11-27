"""Read-only seed/round analysis, archive integrity and publication exports."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256, file_digest
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from program import read_plan
from statistics_program import paired, holm, power_plan


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows, latex=True):
    if not rows:
        path.write_text("")
        path.with_suffix(".tex").write_text("% No observed data; table omitted.\n")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
            for row in rows
        )
    if not latex:
        return

    def escape(v):
        text = "" if v is None else str(v)
        return "".join(
            {
                "_": r"\_",
                "%": r"\%",
                "&": r"\&",
                "#": r"\#",
                "$": r"\$",
                "{": r"\{",
                "}": r"\}",
                "\\": r"\textbackslash{}",
            }.get(c, c)
            for c in text
        )

    tex = [
        "\\begin{tabular}{" + "l" * len(fields) + "}",
        " & ".join(map(escape, fields)) + r" \\",
        r"\hline",
    ]
    tex += [" & ".join(escape(row.get(k)) for k in fields) + r" \\" for row in rows]
    path.with_suffix(".tex").write_text("\n".join(tex + ["\\end{tabular}"]) + "\n")


def unpack(archive, output):
    """Extract only checked research exports into an absent directory."""
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise ValueError("Destination exists")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or "SHA256SUMS" not in names:
            raise ValueError("Duplicate entries or missing checksum manifest")
        expected = {}
        for line in z.read("SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            if name in expected:
                raise ValueError("Duplicate manifest path")
            expected[name] = digest
        if set(expected) != (set(names) - {"SHA256SUMS"}):
            raise ValueError("Manifest does not cover every exported file")
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
                raise ValueError("Unsafe archive path")
            info = z.getinfo(name)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink in archive")
            if name != "SHA256SUMS":
                with z.open(name) as stream:
                    if file_digest(stream, "sha256").hexdigest() != expected[name]:
                        raise ValueError("Archive checksum mismatch: " + name)
        output.mkdir(parents=True)
        for name in names:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, path.open("xb") as dest:
                while block := source.read(1024 * 1024):
                    dest.write(block)
    with archive.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    return dict(archive_sha256=digest, files=len(names))


def load(directory, label):
    from deploy.timeweb.summarize import audit

    directory = Path(directory)
    if (directory / "RUNNING").exists():
        raise ValueError("Wait for or stop the active series before final analysis")
    plan, digest = read_plan(directory / "plan.json")
    crypto_audit = audit(directory)  # read-only; does not call the CLI that overwrites summary.json
    rows = []
    episodes = []
    hashes = {"plan.json": digest}
    for seed in plan["seeds"]:
        for policy in plan["order"][str(seed)]:
            path = directory / f"{seed}_{policy}"
            files = sorted(path.glob("round-*.json"))
            complete = (path / "complete.json").exists()
            if complete and len(files) != plan["horizon"]:
                raise ValueError("Completed episode has missing rounds")
            io = {}
            if (path / "round-io.jsonl").exists():
                io = {
                    r["round"]: r
                    for r in map(json.loads, (path / "round-io.jsonl").read_text().splitlines())
                }
            accepted = 0
            previous = 0
            for expected, file in enumerate(files, 1):
                hashes[str(file.relative_to(directory))] = sha256(file.read_bytes()).hexdigest()
                r = json.loads(file.read_text())
                if r["round"] != expected or r["seed"] != seed or r["mode"] != policy:
                    raise ValueError("Round identity/order mismatch")
                spent = r["byte_ledger"]["spent"]
                increment = spent - previous
                if increment < 0 or increment != r["telemetry"]["app_tx_bytes"]:
                    raise ValueError("Resource increment mismatch")
                previous = spent
                decision = r.get("policy_decision", {})
                selected = tuple(r["gateways"])
                val = r.get("validation", [None, None])
                accepted += bool(r["accepted"])
                row = dict(
                    series=label,
                    scenario=plan.get("program", {}).get("scenario", {}).get("id", label),
                    seed=seed,
                    policy=policy,
                    round=r["round"],
                    accepted=int(r["accepted"]),
                    horizon=plan["horizon"],
                    budget_bytes=plan["bytes_per_episode"],
                    spent_bytes=spent,
                    round_bytes=increment,
                    accepted_cumulative=accepted,
                    gateway_action="+".join(map(str, selected)),
                    selected_workers="+".join(map(str, r["selected"])),
                    collector_count=len(selected),
                    window_ms=r["window_ms"],
                    total_round_s=r["total_round_s"],
                    full_cycle_s=io.get(r["round"], {}).get("full_cycle_s"),
                    planning_s=r.get("planning_s"),
                    scientific_audit_s=r.get("scientific_audit_s"),
                    validation_loss=val[0],
                    validation_accuracy=val[1],
                    mirror_spent=r["mirror"],
                    denied_frames=r["byte_ledger"]["denied_frames"],
                    dual=decision.get("effective_mirror_price", [None])[0],
                    gateway_duals=decision.get("effective_gateway_prices"),
                    faults=len(r["metrics"].get("physical_outages", [])),
                    projection_started=r["metrics"].get("projection_started"),
                    error_count=len(r["errors"]),
                    errors=r["errors"],
                    network_mode=r["network_mode"],
                    chain_mode=r["chain_mode"],
                    roster_finality_s=r.get("roster_finality_s"),
                    booking_and_node_validation_s=r.get("booking_and_node_validation_s"),
                    data_and_update_s=r.get("data_and_update_s"),
                    accounting_and_model_finality_s=r.get("accounting_and_model_finality_s"),
                )
                rows.append(row)
            episodes.append(
                dict(
                    series=label,
                    policy=policy,
                    seed=seed,
                    complete=complete,
                    observed_rounds=len(files),
                    expected_rounds=plan["horizon"],
                    accepted_observed=accepted,
                    accepted_lower=accepted,
                    accepted_upper=accepted + plan["horizon"] - len(files),
                    spent_bytes=previous,
                    bytes_per_accepted=previous / accepted if accepted else None,
                    retries=len(list((directory / "attempts" / path.name).glob("*"))),
                )
            )
            if complete and (path / "ledger-audit.json").exists():
                final = json.loads((path / "ledger-audit.json").read_text())
                if final["reserved"] or final["spent"] != previous:
                    raise ValueError("Exported final audit disagrees with round counters")
    return (
        plan,
        rows,
        episodes,
        dict(
            plan_sha256=digest,
            source_commit=plan["commit"],
            crypto_audit=crypto_audit,
            input_hashes=hashes,
            sqlite_present=bool(list(directory.rglob("budget.sqlite"))),
            retry_records=(directory / "attempts.jsonl").read_text()
            if (directory / "attempts.jsonl").exists()
            else None,
        ),
    )


def contrasts(episodes, practical_pp=1.0):
    effects = []
    power = []
    for series in sorted({r["series"] for r in episodes}):
        group = [r for r in episodes if r["series"] == series]
        names = {r["policy"] for r in group}
        reference_name = (
            "joint" if "joint" in names else "article_pd" if "article_pd" in names else None
        )
        if reference_name is None:
            continue
        policies = sorted(names - {reference_name})
        reference = {r["seed"]: r for r in group if r["policy"] == reference_name}
        family = max(1, len(policies))
        family_p = {policy: None for policy in policies}
        for policy in policies:
            matched = {r["seed"]: r for r in group if r["policy"] == policy}
            seeds = sorted(set(reference) & set(matched))
            complete = [s for s in seeds if reference[s]["complete"] and matched[s]["complete"]]
            if not complete:
                continue
            delta = [
                reference[s]["accepted_observed"] - matched[s]["accepted_observed"]
                for s in complete
            ]
            h = reference[complete[0]]["expected_rounds"]
            result = paired(delta, family)
            family_p[policy] = result["p_exchangeable"]
            effects.append(
                dict(
                    series=series,
                    control=policy,
                    reference=reference_name,
                    complete_seeds=complete,
                    missing_pair_seeds=[s for s in seeds if s not in complete],
                    conditional_on_complete_pairs=len(complete) != len(seeds),
                    conditional_on_final_attempts=any(
                        reference[s].get("retries", 0) or matched[s].get("retries", 0)
                        for s in complete
                    ),
                    difference_pp=result["mean"] * 100 / h,
                    practical_difference_rounds=practical_pp * h / 100,
                    family_size=family,
                    **result,
                )
            )
            power.append(
                dict(
                    series=series,
                    control=policy,
                    **power_plan(delta, practical_pp * h / 100, family),
                )
            )
        corrected = holm(family_p)
        for row in effects:
            if row["series"] == series:
                row["p_holm"] = corrected[row["control"]]
    return effects, power


def action_overlap(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["series"], r["seed"], r["round"])][r["policy"]] = r["gateway_action"]
    counts = defaultdict(lambda: [0, 0])
    for (series, seed, rd), values in by.items():
        if "joint" not in values:
            continue
        for policy, action in values.items():
            if policy == "joint":
                continue
            count = counts[(series, seed, policy)]
            count[0] += action == values["joint"]
            count[1] += 1
    return [
        dict(
            series=s,
            seed=seed,
            control=p,
            compared_rounds=v[1],
            same_action=v[0],
            agreement=v[0] / v[1],
        )
        for (s, seed, p), v in sorted(counts.items())
    ]


def analyze(inputs, output, practical_pp=1.0):
    if not 0 < practical_pp <= 100:
        raise ValueError("Practical difference must be positive percentage points")
    output = Path(output).resolve()
    if len({label for label, path in inputs}) != len(inputs):
        raise ValueError("Duplicate series labels")
    for label, path in inputs:
        if output == Path(path).resolve() or Path(path).resolve() in output.parents:
            raise ValueError("Analysis output must be outside every original series")
    output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    episodes = []
    provenance = {}
    plans = {}
    for label, path in inputs:
        plan, rows, ep, meta = load(path, label)
        plans[label] = plan
        provenance[label] = meta
        all_rows.extend(rows)
        episodes.extend(ep)
    effects, power = contrasts(episodes, practical_pp)
    table(output / "rounds.csv", all_rows, latex=False)
    table(output / "episodes.csv", episodes)
    table(output / "effects.csv", effects)
    table(output / "action_overlap.csv", action_overlap(all_rows))
    prior = []
    for label, path in inputs:
        for attempt in sorted((Path(path) / "attempts").glob("*/*")):
            if not attempt.is_dir():
                continue
            for file in sorted(attempt.glob("round-*.json")):
                r = json.loads(file.read_text())
                prior.append(
                    dict(
                        series=label,
                        attempt=str(attempt.relative_to(path)),
                        record_type="completed_round_in_prior_attempt",
                        seed=r["seed"],
                        policy=r["mode"],
                        round=r["round"],
                        accepted=r["accepted"],
                        errors=r["errors"],
                        byte_ledger=r["byte_ledger"],
                        metrics=r["metrics"],
                    )
                )
            failure = attempt / "failure.json"
            if failure.exists():
                f = json.loads(failure.read_text())
                prior.append(
                    dict(
                        series=label,
                        attempt=str(attempt.relative_to(path)),
                        record_type="unfinished_result_not_imputed",
                        accepted=None,
                        progress=f.get("progress"),
                        partial_result=f.get("partial_result"),
                    )
                )
    table(output / "prior_attempt_records.csv", prior, latex=False)
    # Failure logs and earlier attempts remain explicit; no missing round is imputed as zero.
    status = dict(
        complete=all(r["complete"] for r in episodes),
        no_retries=not any(r["retries"] for r in episodes),
        prior_attempt_records=len(prior),
        observed_rounds=len(all_rows),
        expected_rounds=sum(r["expected_rounds"] for r in episodes),
        prior_metrics_unavailable=["dual", "planning_s"]
        if any("policy_specs" not in p for p in plans.values())
        else [],
        regret=None,
        regret_reason="No admissible counterfactual comparator observed for real interhost series.",
        scope="Conditional on three VMs and sequential time blocks. Round files are dependent observations.",
        effects=effects,
    )
    write_json(output / "analysis.json", status)
    write_json(output / "power.json", power)
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "analysis_code_hashes.json",
        {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )
    from figures_program import render

    render(all_rows, episodes, effects, plans, output)
    return status


def reported(output):
    """Descriptive arithmetic only, explicitly supplied aggregates, not raw measurements."""
    p = ROOT / "configs/collector_program/reported_061.json"
    data = json.loads(p.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for series, values in data["accepted"].items():
        for policy, count in values.items():
            rows.append(
                dict(
                    series=series,
                    policy=policy,
                    accepted_reported=count,
                    total_reported=960,
                    rate_reported=count / 960,
                    joint_minus_reported=data["accepted"][series]["joint"] - count,
                    provenance="user_supplied_aggregate_unverified_against_raw",
                )
            )
    table(output / "reported_aggregates.csv", rows)
    write_json(
        output / "status.json",
        dict(
            raw_analysis="not_performed_by_reported_command_use_analyze",
            supplied_total_rounds=13440,
            independent_blocks_reported=8,
            new_ci_computed=False,
            power_from_seed_differences=None,
            no_raw_data_claim=True,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("unpack")
    q.add_argument("archive", type=Path)
    q.add_argument("output", type=Path)
    q = sub.add_parser("analyze")
    q.add_argument(
        "--series", action="append", required=True, help="LABEL=/absolute/path/to/series"
    )
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--practical-pp", type=float, default=1.0)
    q = sub.add_parser("reported")
    q.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "unpack":
        print(json.dumps(unpack(a.archive, a.output), indent=2))
    elif a.command == "reported":
        reported(a.out)
    else:
        rows = [item.split("=", 1) for item in a.series]
        result = analyze(rows, a.out, a.practical_pp)
        print(json.dumps({k: v for k, v in result.items() if k != "effects"}, indent=2))
        if not result["complete"] or not result["no_retries"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
