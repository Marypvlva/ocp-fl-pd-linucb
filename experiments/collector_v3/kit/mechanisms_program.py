"""Observed stages, costs, exposure and causal reconstructed-state diagnostics."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from analysis_program import table, write_json
from program import read_plan


def analyze(directory, reconstruction, output):
    directory, reconstruction, output = map(Path, (directory, reconstruction, output))
    if directory.resolve() == output.resolve() or directory.resolve() in output.resolve().parents:
        raise ValueError("Keep derived results outside originals")
    output.mkdir(parents=True, exist_ok=False)
    plan, _ = read_plan(directory / "plan.json")
    stages = []
    resources = []
    prices = []
    cells = defaultdict(list)
    calibration = []
    timings = []
    quality = []
    cycle = []
    proof_completion = []
    sizes = []
    for seed in plan["seeds"]:
        for mode in plan["modes"]:
            folder = directory / f"{seed}_{mode}"
            files = sorted(folder.glob("round-*.json"))
            replay = {
                r["round"]: r
                for r in map(
                    json.loads, (reconstruction / f"{seed}_{mode}.jsonl").read_text().splitlines()
                )
            }
            counts = Counter()
            bill = Counter()
            phase = defaultdict(list)
            brier = defaultdict(list)
            last = None
            first = None
            last_round = None
            for file in files:
                r = json.loads(file.read_text())
                m = r["metrics"]
                t = r["telemetry"]
                rd = r["round"]
                d = replay[rd]
                last = r
                last_round = rd
                first = first or r["started_at"]
                counts["rounds"] += 1
                counts["accepted"] += bool(r["accepted"])
                if not r["accepted"]:
                    stage = (
                        "compute"
                        if not m["compute_completed"]
                        else "aggregate"
                        if not m["aggregate_completed"]
                        else "before_projection"
                        if not m["projection_started"]
                        else "projection_or_later"
                    )
                    counts["failed_" + stage] += 1
                if not m["projection_started"]:
                    counts["unexecuted_projection_rounds"] += 1
                for key in [
                    "roster_finality_s",
                    "booking_and_node_validation_s",
                    "data_and_update_s",
                    "accounting_and_model_finality_s",
                    "total_round_s",
                ]:
                    phase[key].append(r[key])
                phase["projection_rpc_wall_s"].append(m["projection_rpc_wall_s"])
                for key, value in t.items():
                    if key.endswith("_tx_bytes") or key == "app_tx_bytes":
                        bill[key] += value
                feasible = np.array(d["feasible_arms"][0], dtype=bool)
                predictions = np.array(d["predictions"][0])
                chosen = d["arm"][0]
                influenced = predictions[chosen] < predictions[feasible].max() - 1e-7
                gp = d["effective_gateway_prices"][0]
                prices.append(
                    dict(
                        seed=seed,
                        policy=mode,
                        round=rd,
                        mirror_price=d["effective_mirror_price"][0],
                        **{f"gateway_price_{g}": gp[g] for g in range(4)},
                        price_changes_reward_rank=bool(influenced),
                        provenance="causally_reconstructed_not_directly_measured",
                    )
                )
                counts["price_changed_reward_rank"] += bool(influenced)
                counts["mirror_price_positive"] += d["effective_mirror_price"][0] > 1e-10
                counts["any_gateway_price_positive"] += any(v > 1e-10 for v in gp)
                observed = m.get("collector_ms")
                if m["projection_started"] and observed is not None:
                    completion = []
                    for member in r["selected"]:
                        values = [observed.get(str(g), {}).get(str(member)) for g in r["gateways"]]
                        good = [v for v in values if v is not None]
                        completion.append(min(good) if good else None)
                    if all(v is not None for v in completion):
                        proof_completion.append(
                            dict(
                                seed=seed,
                                policy=mode,
                                round=rd,
                                window_ms=r["window_ms"],
                                completion_ms=max(completion),
                                conditional_on_complete_selected_coverage=True,
                            )
                        )
                    for i, member in enumerate(r["selected"]):
                        for g in r["gateways"]:
                            value = observed.get(str(g), {}).get(str(member))
                            y = int(value is not None)
                            cells[(mode, member, g, r["window_ms"])].append(y)
                            if d.get("path_mean_prediction") is not None:
                                own = d["path_mean_prediction"][0][i][g]
                                pooled = d["path_pooled_prediction"][0][g]
                                brier["named"].append((own - y) ** 2)
                                brier["pooled_same_history"].append((pooled - y) ** 2)
                sizes.append(file.stat().st_size)
            if last:
                counts.update(
                    dict(
                        mirror_final=last["mirror"],
                        denied_frames_final=last["byte_ledger"]["denied_frames"],
                    )
                )
                stages.append(dict(seed=seed, policy=mode, **counts))
                resources.append(
                    dict(
                        seed=seed,
                        policy=mode,
                        **bill,
                        final_gateway_spent=last["gateway_spent"],
                        mirror_final=last["mirror"],
                    )
                )
                quality.append(
                    dict(
                        seed=seed,
                        policy=mode,
                        validation_loss=last["validation"][0],
                        validation_accuracy=last["validation"][1],
                        provenance="observed_validation_not_test_set",
                    )
                )
                for key, values in phase.items():
                    timings.append(
                        dict(
                            seed=seed,
                            policy=mode,
                            stage=key,
                            mean=float(np.mean(values)),
                            p50=float(np.quantile(values, 0.5)),
                            p95=float(np.quantile(values, 0.95)),
                            sum_s=float(np.sum(values)),
                        )
                    )
                for key, values in brier.items():
                    calibration.append(
                        dict(
                            seed=seed,
                            policy=mode,
                            predictor=key,
                            labels=len(values),
                            brier=float(np.mean(values)),
                            scope="Same logged-policy history and observed attempted paths only; no counterfactual policy evaluation. Reconstructed historical censoring retained.",
                        )
                    )
                complete = json.loads((folder / "complete.json").read_text())
                cycle.append((complete["finished"] - first) / last_round)
    cell_rows = [
        dict(
            policy=m,
            worker=w,
            gateway=g,
            window_ms=c,
            attempts=len(v),
            successes=sum(v),
            success_observed=float(np.mean(v)),
        )
        for (m, w, g, c), v in sorted(cells.items())
    ]
    for name, data in [
        ("stages", stages),
        ("resource_decomposition", resources),
        ("reconstructed_prices", prices),
        ("path_cells", cell_rows),
        ("prequential_brier", calibration),
        ("stage_timings", timings),
        ("final_validation", quality),
        ("observed_proof_completion", proof_completion),
    ]:
        table(
            output / (name + ".csv"),
            data,
            latex=name not in ("reconstructed_prices", "observed_proof_completion"),
        )
    proof_values = [r["completion_ms"] for r in proof_completion]
    byte_size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    total = sum(r["rounds"] for r in stages)
    write_json(
        output / "diagnostics.json",
        dict(
            observed_rounds=total,
            export_bytes_per_round=byte_size / total,
            episode_first_round_to_complete_mean_seconds_per_round=float(np.mean(cycle)),
            episode_first_round_to_complete_p95_seconds_per_round=float(np.quantile(cycle, 0.95)),
            proof_completion_conditional_p50_p80_p95_ms=np.quantile(
                proof_values, [0.5, 0.8, 0.95]
            ).tolist(),
            proof_completion_scope="Observed completed selected paths, pooled across policies; censored paths omitted from quantiles explicitly. Use generous-window randomized diagnostic probe before freezing SLOs.",
            no_regret_computed=True,
            missing_direct_measurements=[
                "historical_prices",
                "historical_planning_time",
                "test_accuracy",
                "true_counterfactual_rewards",
                "SQLite_export",
            ],
        ),
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6), sharex=True)
    for key in ["mirror_price"] + [f"gateway_price_{g}" for g in range(4)]:
        by = defaultdict(list)
        for r in prices:
            if r["policy"] == "joint":
                by[r["round"]].append(r[key])
        x = sorted(by)
        ax = axes[0] if key == "mirror_price" else axes[1]
        ax.plot(x, [np.mean(by[t]) for t in x], label=key)
    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_ylabel("Реконструированная цена")
    axes[0].set_title("Joint: причинный replay, не прямое измерение")
    axes[1].set_xlabel("Раунд")
    fig.tight_layout()
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(output / ("reconstructed_duals." + suffix), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    analyze(a.directory, a.reconstruction, a.out)


if __name__ == "__main__":
    main()
