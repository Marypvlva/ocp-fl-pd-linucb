"""Audit persistent-availability trials and compare with every fixed configuration."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from types import SimpleNamespace
import zipfile

import numpy as np

from .adaptive_scheduler import network_rule, overhead_prior, transfer_prior
from .availability_experiment import make_trace
from .availability_scheduler import AvailabilityPlanner
from .freshness_scheduler import FreshnessMixturePlanner
from .network_telemetry import NetworkTelemetry, TelemetryMode
from .tcp_dynamic_experiment import observation_for, observed_issues
from .tcp_network import NetworkProfile
from .tcp_runtime import ACTIONS
from .telemetry_alignment_analyze import close_tree
from .telemetry_robustness_analyze import read_csv, write_csv


def best_fixed(records):
    """Exact hindsight minimum only if censoring cannot hide a faster fixed run."""
    if {r["policy"] for r in records} != set(ACTIONS) or len(records) != 4:
        raise ValueError("Four distinct measured fixed baselines are required")
    reached = [r for r in records if int(r["reached"])]
    if not reached:
        return None
    winner = min(reached, key=lambda r: float(r["t_epsilon_wall_s"]))
    best = float(winner["t_epsilon_wall_s"])
    if not math.isfinite(best) or best < 0:
        raise ValueError("Invalid measured time")
    if any(float(r["observed_wall_s"]) < best for r in records if not int(r["reached"])):
        return None
    return winner


def validate(folder):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("family") != "persistent_availability" or meta["status"] != "completed":
        raise ValueError("Expected a completed persistent-availability study")
    config = meta["config"]
    expected = {
        (w, r, c, p)
        for w in config["workers"]
        for r in range(config["repeats"])
        for c, case in meta["conditions"].items()
        for p in case["policies"]
    }
    for condition, case in meta["conditions"].items():
        if condition.startswith("persistent") and not (set(ACTIONS) | {"availability"}) <= set(
            case["policies"]
        ):
            raise ValueError("Missing strong baseline or changed study design")
    summary, rows = read_csv(folder / "summary.csv"), read_csv(folder / "rounds.csv")

    def key(row):
        return int(row["workers"]), int(row["repeat"]), row["condition"], row["policy"]

    if (
        len(summary) != meta["planned_runs"]
        or len(summary) != len(expected)
        or {key(s) for s in summary} != expected
    ):
        raise ValueError("Unplanned, duplicate or missing cases")
    with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt source snapshot")
        for name, digest in meta["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Source mismatch: {name}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    if set(grouped) - expected:
        raise ValueError("Unknown round case")
    traces = {}
    for s in summary:
        workers, repeat, condition, policy = key(s)
        trace_key = workers, repeat, condition
        if trace_key not in traces:
            trace = make_trace(
                config["seed"] + repeat,
                config["rounds"],
                workers,
                condition,
                config["phase_min"],
                config["phase_max"],
            )
            stored = json.loads(
                (folder / f"trace_w{workers}_r{repeat}_{condition}.json").read_text()
            )
            if json.loads(json.dumps([asdict(t) for t in trace])) != stored:
                raise ValueError("Trace differs from assigned seed")
            traces[trace_key] = stored
        sensor_mode = meta["conditions"][condition]["telemetry_mode"]
        sensor = NetworkTelemetry(
            TelemetryMode(**meta["modes"][sensor_mode]), config["seed"] + 3000 + repeat
        )
        planner = (
            AvailabilityPlanner(config["seed"] + 11000 + repeat, config["switch_seconds"])
            if policy == "availability"
            else FreshnessMixturePlanner(config["seed"] + 11000 + repeat, config["switch_seconds"])
            if policy == "network_freshness"
            else None
        )
        records = grouped[key(s)]
        previous = None
        issues = (False, False)
        ledger = {}
        seed = config["model_seeds"][repeat] if config.get("model_seeds") else config["model_seed"]
        for attempt, row in enumerate(records, 1):
            if (
                int(row["attempt"]) != attempt
                or int(row["model_seed"]) != seed
                or float(row["learning_rate"]) != config["learning_rate"]
            ):
                raise ValueError("Broken attempt prefix or training pair")
            step = traces[trace_key][attempt - 1]
            if (
                row["profile"] != step["profile"]
                or json.loads(row["injected_fault"]) != step["fault"]
            ):
                raise ValueError("Different physical scenario")
            reading = sensor.observe(
                NetworkProfile(**meta["network_profiles"][step["profile"]]), attempt
            )
            obs = observation_for(reading, workers, 75 * config["hidden"] + 10, *issues)
            if not close_tree(json.loads(row["network_observation"]), asdict(obs)):
                raise ValueError("Observation is not available causal information")
            if (
                int(row["telemetry_source_attempt"]) != reading.capture_attempt
                or int(row["telemetry_age_rounds"]) != attempt - reading.capture_attempt
                or row["telemetry_source_profile"] != reading.source_profile
                or row["telemetry_mode"] != sensor_mode
            ):
                raise ValueError("Inconsistent sensor capture")
            if policy == "network_freshness":
                planner.begin_round(obs, reading.capture_attempt, attempt)
            action = (
                policy
                if policy in ACTIONS
                else network_rule(obs, previous, config["switch_seconds"])
                if policy == "network_rule"
                else planner.choose(obs, previous)
            )
            if action != row["action"]:
                raise ValueError("Action cannot be causally replayed")
            if planner:
                if not close_tree(json.loads(row["decision"]), planner.last_decision):
                    raise ValueError("Decision metadata mismatch")
                outcome = (
                    action,
                    bool(int(row["accepted"])),
                    float(row["cycle_wall_s"]),
                    float(row["switch_wall_s"]),
                )
                if not all(math.isfinite(v) and v >= 0 for v in outcome[2:]):
                    raise ValueError("Invalid elapsed time")
                if policy == "network_freshness":
                    planner.observe_round(*outcome)
                    if int(row["cost_feedback_updates"]) != planner.cost_feedback_count():
                        raise ValueError("Mismatched cost feedback count")
                    # Raw capture IDs are consumed once, on their first delivery.
                    ledger.setdefault(
                        reading.capture_attempt, (records[reading.capture_attempt - 1], obs)
                    )
                else:
                    planner.observe(obs, *outcome)
                if int(row["feedback_updates"]) != planner.feedback_count() or not close_tree(
                    json.loads(row["planner_statistics"]), planner.stats
                ):
                    raise ValueError("Statistics include missing or extra feedback")
            if (
                not int(row["traffic_is_lower_bound"])
                and row["app_tx_bytes"] != row["app_rx_bytes"]
            ):
                raise ValueError("Application byte counters disagree")
            previous = action
            issues = observed_issues(SimpleNamespace(errors=json.loads(row["errors"])))
        if policy == "network_freshness":
            for action in ACTIONS:
                matched = [(r, o) for r, o in ledger.values() if r["action"] == action]
                expected_cost = 2 * overhead_prior(action)
                for original, captured in matched:
                    expected_cost += max(
                        0.001,
                        float(original["cycle_wall_s"])
                        - float(original["switch_wall_s"])
                        - transfer_prior(action, captured),
                    )
                stat = planner.stats[action]
                if stat["cost_n"] != len(matched) or not math.isclose(
                    stat["residual_sum"], expected_cost, rel_tol=1e-12, abs_tol=1e-10
                ):
                    raise ValueError("Independent matched-feedback ledger differs")
        if len(records) != int(s["attempts"]) or sum(int(r["accepted"]) for r in records) != int(
            s["accepted_updates"]
        ):
            raise ValueError("Summary differs from rounds")
        if records:
            hit = float(records[-1]["validation_loss"]) <= config["target_loss"]
            if int(s["reached"]) != int(hit) or (
                hit
                and float(s["t_epsilon_wall_s"]) != float(records[-1]["quality_observed_wall_s"])
            ):
                raise ValueError("Incorrect time to target or censoring")
        elif not int(s["reached"]) or float(s["initial_loss"]) > config["target_loss"]:
            raise ValueError("Missing nontrivial round records")
        mode = meta["conditions"][condition]["telemetry_mode"]
        journal = (
            folder
            / f"journal_{condition}_w{workers}_r{repeat}_{mode}_{policy.replace('+', '')}.jsonl"
        )
        recorded = [json.loads(line) for line in journal.read_text().splitlines()]
        if len(recorded) != len(records):
            raise ValueError("Journal truncated")
        for a, b in zip(recorded, records):
            if any(
                (b[k] not in ("", "None") if v is None else str(v) != b[k]) for k, v in a.items()
            ):
                raise ValueError("JSONL and CSV disagree")
    return meta, summary, rows


def contrasts(summary):
    index = {(int(r["workers"]), int(r["repeat"]), r["condition"], r["policy"]): r for r in summary}
    if len(index) != len(summary):
        raise ValueError("Duplicate run")
    rng = np.random.default_rng(193026)
    results, pairs = [], []
    for workers in sorted({k[0] for k in index}):
        repeats = sorted({k[1] for k in index if k[0] == workers})
        specs = [
            (condition, p, "best_fixed")
            for condition in sorted(
                {k[2] for k in index if k[0] == workers and k[2].startswith("persistent")}
            )
            for p in ("availability", "network_rule", "network_freshness")
            if (workers, repeats[0], condition, p) in index
        ]
        specs += [
            ("persistent_delayed", "availability", "network_rule"),
            ("healthy", "availability", "F+P"),
        ]
        specs = [
            spec
            for spec in specs
            if (workers, repeats[0], spec[0], spec[1]) in index
            and (spec[2] == "best_fixed" or (workers, repeats[0], spec[0], spec[2]) in index)
        ]
        for condition, first, reference in specs:
            values = []
            for repeat in repeats:
                a = index[(workers, repeat, condition, first)]
                b = (
                    best_fixed([index[(workers, repeat, condition, p)] for p in ACTIONS])
                    if reference == "best_fixed"
                    else index[(workers, repeat, condition, reference)]
                )
                if b is not None and (a["model_seed"], a["learning_rate"]) != (
                    b["model_seed"],
                    b["learning_rate"],
                ):
                    raise ValueError("Training configurations differ within a pair")
                complete = b is not None and int(a["reached"]) and int(b["reached"])
                delta = (
                    float(a["t_epsilon_wall_s"]) - float(b["t_epsilon_wall_s"])
                    if complete
                    else None
                )
                if complete:
                    values.append(delta)
                pairs.append(
                    dict(
                        workers=workers,
                        repeat=repeat,
                        condition=condition,
                        policy=first,
                        reference=reference,
                        winning_fixed=b["policy"] if b else "unknown",
                        delta_wall_s=delta,
                        complete=int(bool(complete)),
                    )
                )
            low = high = None
            if len(values) >= 2:
                low, high = np.quantile(
                    rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1),
                    [0.025, 0.975],
                ).tolist()
            results.append(
                dict(
                    workers=workers,
                    condition=condition,
                    policy=first,
                    reference=reference,
                    primary=int(
                        condition == "persistent_delayed"
                        and first == "availability"
                        and reference == "best_fixed"
                    ),
                    pairs=len(repeats),
                    complete_pairs=len(values),
                    mean_delta_wall_s=float(np.mean(values)) if values else None,
                    bootstrap95_low_s=low,
                    bootstrap95_high_s=high,
                )
            )
    return results, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    meta, summary, rows = validate(args.folder)
    paired, individual = contrasts(summary)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "individual_pairs.csv", individual)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = sorted({int(r["workers"]) for r in summary})
    conditions = [c for c in meta["conditions"] if c.startswith("persistent")]
    fig, axes = plt.subplots(
        len(counts), len(conditions), figsize=(12.8, 4.6 * len(counts)), squeeze=False
    )
    for row_axes, count in zip(axes, counts):
        for ax, condition in zip(row_axes, conditions):
            policies = meta["conditions"][condition]["policies"]
            for i, policy in enumerate(policies):
                values = [
                    float(r["t_epsilon_wall_s"])
                    for r in summary
                    if int(r["workers"]) == count
                    and r["condition"] == condition
                    and r["policy"] == policy
                    and int(r["reached"])
                ]
                color = "#358570" if policy == "availability" else "#6c8faa"
                ax.scatter(
                    i + np.linspace(-0.06, 0.06, len(values)), values, color=color, alpha=0.6, s=22
                )
                if values:
                    ax.scatter([i], [median(values)], color=color, marker="D", s=45)
            labels = [
                p.replace("network_rule", "rule")
                .replace("network_freshness", "freshness")
                .replace("availability", "stateful")
                for p in policies
            ]
            ax.set_xticks(range(len(policies)), labels, fontsize=9)
            ax.set(
                title=f"{count} workers · {condition}", ylabel="T_epsilon · physical localhost (s)"
            )
            ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "availability_time_to_quality.png", dpi=180)
    plt.close(fig)
    fmt = lambda x: "—" if x is None else f"{x:+.3f}"
    lines = [
        "# Условия полезности совместной адаптации",
        "",
        "Исследуется сохранение отказов во времени, а не настройка под предыдущие проверочные трассы. "
        "Новая причинная stateful-эвристика помнит наблюдённую недоступность edge/P четыре выбора, затем пробует вернуться. "
        "Она не получает текущий режим или Fault. Это простая контрольная стратегия, а не универсальный обученный оптимизатор.",
        "",
    ]
    for p in paired:
        if p["primary"]:
            lines += [
                f"Первичный stateful−лучшее фиксированное действие на трассе: {fmt(p['mean_delta_wall_s'])} s, "
                f"описательный bootstrap 95% [{fmt(p['bootstrap95_low_s'])}; {fmt(p['bootstrap95_high_s'])}], "
                f"полных пар {p['complete_pairs']}/{p['pairs']}.",
                "",
            ]
    lines += [
        f"Порог достигнут {sum(int(r['reached']) for r in summary)}/{len(summary)} запусков; "
        f"{len(rows)} TCP-раундов, {sum(int(r['accepted']) for r in rows)} принятых обновлений, "
        f"{sum(int(r['timeouts']) for r in rows)} тайм-аутов.",
        "",
        "## Дизайн и сильное сравнение",
        "",
        "В каждом цикле случайно переставляются четыре режима: здоровая сеть, недоступность одного участника "
        "на интерактивном этапе P, отказ primary-edge, совместное состояние. Фазы 12–18 попыток; быстрый ответ об отказе P и ответ с реальной задержкой 0.10 s измеряются отдельно. "
        "Во всех режимах вход проверяющего ограничен 40 MB/s, 2 ms на сообщение. "
        "B получает подписанные обновления до интерактивного этапа и сохраняет работоспособность при отказе P; "
        "R имеет резервную копию полосы при отказе primary. Это контролируемая модель доступности, "
        "не реальные отключения физического хоста. Отказы процессов отдельно проверяются тестами.",
        "",
        "Первичное сравнение использует минимум из ВСЕХ ЧЕТЫРЁХ действительно измеренных фиксированных стратегий "
        "для каждой пары модели/трассы. Это благоприятный для fixed hindsight-выбор, доступный только анализатору. "
        "При цензурировании минимум считается известным лишь когда все цензурированные нижние границы времени "
        "не меньше лучшего достигнутого времени. Иначе пара исключается с явной отметкой unknown. "
        "Никакой алгоритм не получает результаты невыбранных действий.",
        "",
        "Отдельный healthy-контроль сравнивает stateful с F+P без инъекций. "
        "Внутри блока перемешаны семь delayed-стратегий, пять fast-стратегий и два healthy-контроля. "
        "Старые абсолютные времена не объединяются с новыми после реорганизации кода.",
        "",
        f"![Время до качества]({output}/availability_time_to_quality.png)",
        "",
        "## Парные контрасты",
        "",
        "| Условие | Стратегия | Сравнение | Полных пар | Разность, s | Bootstrap 95%, s |",
        "|---|---|---|---:|---:|---:|",
    ]
    for p in paired:
        lines.append(
            f"| {p['condition']} | {p['policy']} | {p['reference']} | {p['complete_pairs']}/{p['pairs']} | "
            f"{fmt(p['mean_delta_wall_s'])} | [{fmt(p['bootstrap95_low_s'])}; {fmt(p['bootstrap95_high_s'])}] |"
        )
    lines += [
        "",
        "Отрицательная разность означает ускорение. Только stateful−best_fixed в persistent_delayed является первичным. "
        "Остальные интервалы описательные, без коррекции множественности. 20000 bootstrap-выборок, seed 193026. "
        "Мощность не рассчитана. Недостигнутые времена не подменяются временем остановки.",
        "",
        "## Ограничения",
        "",
        "Один Mac, MLP/Digits, честные участники, проверяется сумма. Сценарии специально задают длительные "
        "состояния доступности, где выбор конфигурации может быть полезен. Это проверка области применимости, "
        "а не универсальная победа над fixed. Режимы привязаны к номеру попытки, не общему wall-clock, поэтому "
        "алгоритмы встречают разные состояния к моменту достижения порога. Все реплики R собираются до "
        "проверки, включая неудачные. Уменьшение ожидания лишних реплик и WAN здесь не исследуются.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    result = dict(
        status="passed",
        runs=len(summary),
        reached=sum(int(s["reached"]) for s in summary),
        rounds=len(rows),
        accepted=sum(int(r["accepted"]) for r in rows),
        timeouts=sum(int(r["timeouts"]) for r in rows),
        primary=[p for p in paired if p["primary"]],
        source_files=len(meta["source_sha256"]),
    )
    (output / "validation.json").write_text(json.dumps(result, indent=2))
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(result), flush=True)
    return output


if __name__ == "__main__":
    main()
