"""Analyze causal telemetry perturbations with a genuinely shared fixed control."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from statistics import median
import zipfile

import numpy as np

from .network_telemetry import NetworkTelemetry, TelemetryMode
from .tcp_network import NetworkProfile, PROFILES

POLICIES = ("network_online", "network_prior")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparisons(summary, modes, primary_mode="stale_noise", control="F+P"):
    index = {}
    for row in summary:
        key = (int(row["workers"]), int(row["repeat"]), row["condition"], row["policy"])
        if key in index:
            raise ValueError(f"Duplicate run: {key}")
        index[key] = row
    rng = np.random.default_rng(163026)
    results = []
    for workers in sorted({k[0] for k in index}):
        repeats = sorted({k[1] for k in index if k[0] == workers})
        specs = []
        for mode in modes:
            specs += [
                (mode, "network_online", mode, "network_prior"),
                (mode, "network_online", "control", control),
            ]
            if mode != "exact" and "exact" in modes:
                specs += [(mode, policy, "exact", policy) for policy in POLICIES]
        for ca, pa, cb, pb in specs:
            pairs = [
                (index[(workers, i, ca, pa)], index[(workers, i, cb, pb)])
                for i in repeats
                if (workers, i, ca, pa) in index and (workers, i, cb, pb) in index
            ]
            if any(
                (a.get("model_seed"), a.get("learning_rate"))
                != (b.get("model_seed"), b.get("learning_rate"))
                for a, b in pairs
            ):
                raise ValueError("Pair has different initial model or learning rate")
            good = [(a, b) for a, b in pairs if int(a["reached"]) and int(b["reached"])]
            delta = np.array(
                [float(a["t_epsilon_wall_s"]) - float(b["t_epsilon_wall_s"]) for a, b in good]
            )
            low = high = None
            if len(delta) >= 2:
                draws = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975]).tolist()
            results.append(
                dict(
                    workers=workers,
                    first_condition=ca,
                    first_policy=pa,
                    second_condition=cb,
                    second_policy=pb,
                    primary=int(
                        ca == cb == primary_mode
                        and pa == "network_online"
                        and pb == "network_prior"
                    ),
                    pairs=len(pairs),
                    complete_pairs=len(good),
                    first_reached=sum(int(a["reached"]) for a, _ in pairs),
                    second_reached=sum(int(b["reached"]) for _, b in pairs),
                    mean_delta_wall_s=float(delta.mean()) if len(delta) else None,
                    bootstrap95_low_s=low,
                    bootstrap95_high_s=high,
                    note="one predeclared primary; other contrasts descriptive; shared control is not replicated",
                )
            )
    return results


def validate(folder, family="telemetry_robustness"):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("family") != family or meta["status"] != "completed":
        raise ValueError(f"Expected a completed {family} study")
    summary, rows = read_csv(folder / "summary.csv"), read_csv(folder / "rounds.csv")
    if len(summary) != meta["planned_runs"]:
        raise ValueError("Missing planned runs")
    with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
        for name, digest in meta["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Source snapshot mismatch: {name}")
    traces, sensors, attempts = {}, {}, {}
    config = meta["config"]
    for row in rows:
        workers, repeat, attempt = int(row["workers"]), int(row["repeat"]), int(row["attempt"])
        key = (workers, repeat)
        if key not in traces:
            traces[key] = json.loads((folder / f"trace_w{workers}_r{repeat}.json").read_text())
        step = traces[key][attempt - 1]
        if row["profile"] != step["profile"] or json.loads(row["injected_fault"]) != step["fault"]:
            raise ValueError("Physical trace pairing mismatch")
        mode = "exact" if row["condition"] == "control" else row["condition"]
        if row["telemetry_mode"] != mode:
            raise ValueError("Wrong sensor mode")
        case = (workers, repeat, row["condition"], row["policy"])
        if case not in sensors:
            settings = meta["modes"].get(mode, dict(name="exact", delay_rounds=0, log_sigma=0.0))
            sensors[case] = NetworkTelemetry(
                TelemetryMode(**settings), config["seed"] + 3000 + repeat
            )
            attempts[case] = 0
        attempts[case] += 1
        actual = (
            NetworkProfile(**meta["network_profiles"][step["profile"]])
            if "network_profiles" in meta
            else PROFILES[step["profile"]]
        )
        report = sensors[case].observe(actual, attempt)
        observation = json.loads(row["network_observation"])
        if any(
            not math.isclose(observation[k], getattr(report, k), rel_tol=1e-12, abs_tol=1e-10)
            for k in ("model_mbps", "edge_mbps", "verifier_mbps", "latency_ms")
        ):
            raise ValueError("Telemetry is not the expected causal report")
        if (
            int(row["telemetry_source_attempt"]) != report.capture_attempt
            or int(row["telemetry_age_rounds"]) != attempt - report.capture_attempt
            or row["telemetry_source_profile"] != report.source_profile
            or report.capture_attempt > attempt
        ):
            raise ValueError("Future or inconsistent telemetry capture")
        seed = config["model_seeds"][repeat] if config.get("model_seeds") else config["model_seed"]
        if int(row["model_seed"]) != seed or float(row["learning_rate"]) != config["learning_rate"]:
            raise ValueError("Incorrect training configuration")
        if row["policy"] == "network_online" and int(row["feedback_updates"]) != attempt:
            raise ValueError("Chosen-action feedback mismatch")
        if row["policy"] == "network_prior" and int(row["feedback_updates"]) != 0:
            raise ValueError("Frozen policy learned from feedback")
        if not int(row["traffic_is_lower_bound"]) and row["app_tx_bytes"] != row["app_rx_bytes"]:
            raise ValueError("Application byte counters mismatch")
    control_runs = [r for r in summary if r["condition"] == "control"]
    if len(control_runs) != len(config["workers"]) * config["repeats"]:
        raise ValueError("Shared fixed control was duplicated or omitted")
    return meta, summary, rows


def figures(summary, modes, control, output, diagnostic=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, mode in enumerate(modes):
        for offset, policy, colour in (
            (-0.15, "network_online", "#d26832"),
            (0.15, "network_prior", "#3676a8"),
        ):
            values = [
                float(r["t_epsilon_wall_s"])
                for r in summary
                if r["condition"] == mode and r["policy"] == policy and int(r["reached"])
            ]
            if values:
                ax.scatter(
                    i + offset + np.linspace(-0.04, 0.04, len(values)),
                    values,
                    color=colour,
                    alpha=0.55,
                    s=20,
                    label=policy if i == 0 else None,
                )
                ax.scatter([i + offset], [median(values)], color=colour, marker="D", s=48)
    values = [
        float(r["t_epsilon_wall_s"])
        for r in summary
        if r["condition"] == "control" and int(r["reached"])
    ]
    if values:
        ax.scatter(
            len(modes) + np.linspace(-0.08, 0.08, len(values)),
            values,
            color="#7a818b",
            alpha=0.6,
            label="shared " + control,
        )
        ax.axhline(median(values), color="#7a818b", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(modes) + 1), [*modes, "shared " + control])
    counts = sorted({int(r["workers"]) for r in summary})
    if len(counts) != 1:
        raise ValueError("Use one worker count per analysis")
    seeds = {int(r["model_seed"]) for r in summary}
    rates = {float(r["learning_rate"]) for r in summary}
    ax.set(
        title=f"{counts[0]} workers · {len(seeds)} initial models · learning rates {sorted(rates)}",
        ylabel="T_epsilon, localhost wall-clock (s)",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.text(
        0.5,
        0.01,
        "Dots: reached runs; diamonds: medians. Fixed control is measured once per model/trace.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "telemetry_time_to_quality.png", dpi=180)
    plt.close(fig)
    if diagnostic:
        records = read_csv(diagnostic / "summary.csv")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for j, rate in enumerate((1.0, 0.5)):
            for i, group in enumerate(("development", "validation")):
                rr = [
                    r
                    for r in records
                    if r["seed_set"] == group and float(r["learning_rate"]) == rate
                ]
                reached = sum(int(r["reached"]) for r in rr)
                x = i + (j - 0.5) * 0.3
                ax.bar(
                    x,
                    reached / len(rr),
                    width=0.27,
                    color=("#3676a8", "#c87538")[j],
                    label=f"learning rate {rate}" if i == 0 else None,
                )
                ax.text(x, reached / len(rr) + 0.02, f"{reached}/{len(rr)}", ha="center")
        ax.set_xticks([0, 1], ["initialization development", "initialization validation"])
        ax.set(
            ylabel="Fraction reaching loss 0.22 within 120 updates",
            ylim=(0, 1.18),
            title="Training-only diagnostic · no TCP timings",
        )
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output / "initialization_convergence.png", dpi=180)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-diagnostic", type=Path)
    parser.add_argument("--tests-log", type=Path)
    args = parser.parse_args(argv)
    folder, output = args.folder.resolve(), args.output.resolve()
    meta, summary, rows = validate(folder)
    modes, control = list(meta["modes"]), meta["shared_control_policy"]
    paired = paired_comparisons(summary, modes, meta["primary_mode"], control)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired)
    figures(summary, modes, control, output, args.training_diagnostic)
    config = meta["config"]
    tests = ""
    if args.tests_log:
        log = args.tests_log.read_text()
        match = re.search(r"Ran (\d+) tests", log)
        if not match or not log.rstrip().endswith("OK"):
            raise ValueError("Tests not confirmed")
        tests = f"Пройдено автоматических тестов: {match.group(1)}."
        shutil.copy2(args.tests_log, output / "tests.log")
    primary = next(r for r in paired if r["primary"])
    mean, low, high = (
        primary[k] for k in ("mean_delta_wall_s", "bootstrap95_low_s", "bootstrap95_high_s")
    )
    result = (
        f"Первичный контраст stale_noise: network_online−network_prior = {mean:+.3f} s, "
        f"описательный bootstrap 95% [{low:+.3f}; {high:+.3f}]."
        if low is not None
        else "Недостаточно полных пар для первичного интервала."
    )
    lines = [
        "# adaptive_planning: устойчивость к телеметрии и начальным весам",
        "",
        result,
        "",
        f"Фактических TCP-запусков {len(summary)}, достигли loss≤{config['target_loss']}: "
        f"{sum(int(r['reached']) for r in summary)}/{len(summary)}. {tests}",
        "",
        "## Протокол",
        "",
        f"Участников 12; начальные модели {config['model_seeds']}, learning rate={config['learning_rate']}, "
        f"seed трасс={config['seed']}, повторов={config['repeats']}. В каждом повторе порядок всех условий и стратегий перемешивается. Профили сети и отказы одинаковы для сравниваемых стратегий и режимов датчика.",
        "",
        "Режимы: exact; noise с мультипликативным логнормальным шумом σ=0.35; stale с задержкой на 2 попытки; stale_noise объединяет оба и является заранее выбранным первичным условием. Отчёт сначала захватывается, потом доставляется; будущие состояния не используются.",
        "",
        "Физическая передача по TCP остаётся прежней: 40 MB/s на соответствующий общий поток, 2 ms на сообщение, смена ограниченного потока между раундами. Сенсор синтетический: шум не применяется к самой сети.",
        "",
        "Фиксированный F+P не использует датчик. Он реально измеряется один раз на инициализацию/трассу, а не четыре раза. Общий контроль участвует в нескольких описательных парных контрастах; эти контрасты не независимы. Исходные записи контроля не дублируются.",
        "",
        "## Время до качества",
        "",
        "Медианы T_epsilon, секунды физического localhost wall-clock:",
        "",
        "| Датчик | network_online | network_prior | Достигли порога (online/prior) |",
        "|---|---:|---:|---:|",
    ]
    for mode in modes:
        groups = [
            [r for r in summary if r["condition"] == mode and r["policy"] == p] for p in POLICIES
        ]
        good = [[r for r in group if int(r["reached"])] for group in groups]
        values = [
            f"{median(float(r['t_epsilon_wall_s']) for r in rr):.3f}" if rr else "—" for rr in good
        ]
        lines.append(
            f"| {mode} | {values[0]} | {values[1]} | {len(good[0])}/{len(groups[0])}; {len(good[1])}/{len(groups[1])} |"
        )
    cc = [r for r in summary if r["condition"] == "control"]
    completed = [r for r in cc if int(r["reached"])]
    if completed:
        lines += [
            "",
            f"Общий фиксированный контроль {control}: медиана {median(float(r['t_epsilon_wall_s']) for r in completed):.3f} s, достигли {len(completed)}/{len(cc)}.",
        ]
    lines += [
        "",
        f"![Время до качества]({output}/telemetry_time_to_quality.png)",
        "",
        "## Парные различия online−prior",
        "",
        "| Условие | Полных пар | Средняя разность, s | Описательный bootstrap 95%, s |",
        "|---|---:|---:|---:|",
    ]
    for row in paired:
        if (
            row["first_condition"] != row["second_condition"]
            or row["second_policy"] != "network_prior"
        ):
            continue
        value = f"{row['mean_delta_wall_s']:+.3f}" if row["mean_delta_wall_s"] is not None else "—"
        ci = (
            f"[{row['bootstrap95_low_s']:+.3f}; {row['bootstrap95_high_s']:+.3f}]"
            if row["bootstrap95_low_s"] is not None
            else "—"
        )
        lines.append(f"| {row['first_condition']} | {row['complete_pairs']} | {value} | {ci} |")
    lines += [
        "",
        "Отрицательное значение означает выигрыш обучаемого планировщика. Единственный первичный контраст — online−prior в stale_noise. Остальные интервалы описательные, без поправки на множественные сравнения. Недостигнутый порог не заменяется временем остановки; анализ полных пар может быть смещён при цензурировании.",
        "",
        "## Устойчивость обучения модели",
        "",
    ]
    if args.training_diagnostic:
        diagnostic = read_csv(args.training_diagnostic / "summary.csv")
        write_csv(output / "initialization_summary.csv", diagnostic)
        lines += [
            "Отдельная центральная диагностика обучения, без TCP и без утверждений о стоимости криптографии:",
            "",
            "| Группа начальных весов | Learning rate | Достигли порога | Медиана успешных обновлений среди достигших |",
            "|---|---:|---:|---:|",
        ]
        for group in ("development", "validation"):
            for rate in (1.0, 0.5):
                rr = [
                    r
                    for r in diagnostic
                    if r["seed_set"] == group and float(r["learning_rate"]) == rate
                ]
                hit = [r for r in rr if int(r["reached"])]
                lines.append(
                    f"| {group} | {rate} | {len(hit)}/{len(rr)} | {median(int(r['successful_updates']) for r in hit) if hit else '—'} |"
                )
        lines += [
            "",
            f"![Сходимость инициализаций]({output}/initialization_convergence.png)",
            "",
            "Более консервативный шаг 0.5 был задан до просмотра этих результатов. Все восемь заранее выбранных инициализаций 16…23 включены в TCP-серию; неуспешные seed не отбирались и градиенты не обрезались. Времена новой серии нельзя напрямую сравнивать с прежними измерениями на шаге 1.0 и одной модели 2026.",
        ]
    lines += [
        "",
        "## Проверки и ограничения",
        "",
        f"Реальных раундов {len(rows)}, принятых обновлений {sum(int(r['accepted']) for r in rows)}, "
        f"записей о тайм-аутах {sum(int(r['timeouts']) for r in rows)}. Проверены точные префиксы трасс, модели, learning rate, причинно воспроизведённые отчёты датчика, источник каждого отчёта, feedback только выбранного действия и снимки исходников.",
        "",
        "Начальная калибровка первого профиля доступна сразу: пока задержанный канал заполняется, повторяется первый отчёт. Шум возмущает только положительные величины; флаг 0=неограниченный поток сохраняется, поэтому не моделируется ошибочное распознавание этого флага. Задержка измеряется попытками, а не секундами. Измерительные TCP-пробы не выполняются: это контролируемая инъекция ошибок наблюдения.",
        "",
        "Профили меняются только между раундами на одном Mac. WAN, несколько физических машин, шум градиентов от разных minibatch и обновления посреди кадра не проверены. Восемь инициализаций не доказывают универсальную устойчивость MLP, а bootstrap-интервалы не заменяют расчёт мощности.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "cd /Users/mariiapavlova/PycharmProjects/adaptive_planning",
        ".venv/bin/python training_initialization_check.py",
        ".venv/bin/python telemetry_robustness_experiment.py --config configs/tcp_telemetry_robustness.json",
        "```",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    shutil.copy2(folder / "metadata.json", output / "metadata.json")
    with zipfile.ZipFile(output / "measurements.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in folder.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(folder))
        if args.training_diagnostic:
            for path in args.training_diagnostic.rglob("*"):
                if path.is_file():
                    archive.write(
                        path,
                        Path("training_only_diagnostic")
                        / path.relative_to(args.training_diagnostic),
                    )
        if args.tests_log:
            archive.write(args.tests_log, "tests.log")
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
