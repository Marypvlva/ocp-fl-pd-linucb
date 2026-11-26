"""Analyze a completed paced TCP sweep without mixing censored observations."""

from .paths import ROOT
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from statistics import median
import zipfile

import numpy as np

POLICIES = ("F+B", "F+P", "R+B", "R+P", "rule", "online")
CONTRASTS = (
    ("F+P", "F+B"),
    ("R+P", "R+B"),
    ("R+B", "F+B"),
    ("R+P", "F+P"),
    ("online", "rule"),
    ("online", "F+B"),
    ("online", "F+P"),
    ("online", "R+B"),
    ("online", "R+P"),
)


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


def paired_rows(summary):
    rng = np.random.default_rng(121026)
    indexed = {}
    for row in summary:
        key = (row["network_profile"], int(row["workers"]), int(row["repeat"]), row["policy"])
        if key in indexed:
            raise ValueError(f"Duplicate run: {key}")
        indexed[key] = row
    results = []
    for profile, workers in sorted({key[:2] for key in indexed}):
        repeats = sorted({key[2] for key in indexed if key[:2] == (profile, workers)})
        for first, second in CONTRASTS:
            pairs = [
                (indexed[(profile, workers, i, first)], indexed[(profile, workers, i, second)])
                for i in repeats
                if (profile, workers, i, first) in indexed
                and (profile, workers, i, second) in indexed
            ]
            deltas = [
                float(a["t_epsilon_wall_s"]) - float(b["t_epsilon_wall_s"])
                for a, b in pairs
                if int(a["reached"]) and int(b["reached"])
            ]
            lo = hi = None
            if len(deltas) >= 2:
                means = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
                lo, hi = np.quantile(means, [0.025, 0.975]).tolist()
            results.append(
                dict(
                    network_profile=profile,
                    workers=workers,
                    contrast=f"{first} - {second}",
                    pairs=len(pairs),
                    complete_pairs=len(deltas),
                    first_reached=sum(int(a["reached"]) for a, _ in pairs),
                    second_reached=sum(int(b["reached"]) for _, b in pairs),
                    mean_delta_wall_s=float(np.mean(deltas)) if deltas else None,
                    bootstrap95_low_s=lo,
                    bootstrap95_high_s=hi,
                    interpretation="exploratory, complete pairs, no multiplicity correction",
                )
            )
    return results


def figures(summary, paired, profiles, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4.8), squeeze=False)
    for ax, profile in zip(axes[0], profiles):
        for i, policy in enumerate(POLICIES):
            values = [
                float(s["t_epsilon_wall_s"])
                for s in summary
                if s["network_profile"] == profile and s["policy"] == policy and int(s["reached"])
            ]
            if not values:
                continue
            colour = (
                "#ce622f" if policy == "online" else "#777f8c" if policy == "rule" else "#3676a8"
            )
            ax.scatter(
                i + np.linspace(-0.1, 0.1, len(values)), values, color=colour, s=22, alpha=0.6
            )
            ax.scatter([i], [median(values)], color=colour, marker="D", s=50)
        ax.set_xticks(range(6), POLICIES, rotation=35)
        ax.set(title=profile, ylabel="T_epsilon, localhost wall-clock (s)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("12 workers · target loss 0.22 · paired runs within each profile")
    fig.text(
        0.5,
        0.01,
        "Dots: reached runs; diamonds: medians. Profile panels use separate y-axis scales.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / "network_time_to_quality.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions, labels = [], []
    for profile in profiles:
        for contrast in ("F+P - F+B", "R+P - R+B"):
            row = next(
                r for r in paired if r["network_profile"] == profile and r["contrast"] == contrast
            )
            if row["bootstrap95_low_s"] is None:
                continue
            i = len(labels)
            mean, low, high = (
                row[k] for k in ("mean_delta_wall_s", "bootstrap95_low_s", "bootstrap95_high_s")
            )
            ax.errorbar(
                mean, i, xerr=[[mean - low], [high - mean]], fmt="o", capsize=4, color="#3676a8"
            )
            labels.append(profile + " · " + contrast)
            positions.append(i)
    ax.axvline(0, color="#777", linestyle="--", linewidth=1)
    ax.set_yticks(positions, labels)
    ax.set(
        xlabel="Mean paired T_epsilon difference P - B (s)",
        title="Descriptive bootstrap 95% intervals · negative values favor P",
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "network_projection_contrasts.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tests-log", type=Path)
    args = parser.parse_args(argv)
    sweep, output = args.sweep.resolve(), args.output.resolve()
    status = json.loads((sweep / "sweep.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Sweep did not complete")
    profiles = status["profiles"]
    summary = read_csv(sweep / "summary.csv")
    rounds = [row for profile in profiles for row in read_csv(sweep / profile / "rounds.csv")]
    if {int(s["workers"]) for s in summary} != {12}:
        raise ValueError(
            "This report is designed for the 12-worker pilot; raw CSVs support other counts"
        )
    checked = {}
    for profile in profiles:
        meta = json.loads((sweep / profile / "metadata.json").read_text())
        if meta["status"] != "completed":
            raise ValueError("Incomplete profile")
        # Verify against captured source, independent of subsequent edits in the workspace.
        with zipfile.ZipFile(sweep / profile / "source_snapshot.zip") as snapshot:
            for name, digest in meta["source_sha256"].items():
                if hashlib.sha256(snapshot.read(name)).hexdigest() != digest:
                    raise ValueError(f"Source snapshot mismatch: {profile}/{name}")
        checked[profile] = meta
    if any(
        r["app_tx_bytes"] != r["app_rx_bytes"]
        for r in rounds
        if not int(r["traffic_is_lower_bound"])
    ):
        raise ValueError("Unexpected frame counter mismatch")
    paired = paired_rows(summary)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired)
    figures(summary, paired, profiles, output)
    reached = sum(int(s["reached"]) for s in summary)
    tests = ""
    if args.tests_log:
        log = args.tests_log.read_text()
        match = re.search(r"Ran (\d+) tests", log)
        if not match or not log.rstrip().endswith("OK"):
            raise ValueError("No successful test run in log")
        tests = f"Пройдено автоматических тестов: {match.group(1)}."
        shutil.copy2(args.tests_log, output / "tests.log")
    lines = [
        "# Следующий этап adaptive_planning: ограниченная передача TCP",
        "",
        "Изменения выполнены в `/Users/mariiapavlova/PycharmProjects/adaptive_planning`.",
        "",
        "Добавлены общий между процессами ограничитель передачи, аутентифицированный дедлайн RPC, калибровка и сравнение трёх сетевых профилей. Передача и обучение выполняются реально; это прикладное управление потоком на одном Mac.",
        "",
        f"Завершено {len(summary)} запусков стратегий: 12 участников, {len({s['policy'] for s in summary})} стратегий, {len({s['repeat'] for s in summary})} парных повторов и {len(profiles)} профиля. Порог loss≤0.22 достигнут в {reached}/{len(summary)} запусках. {tests}",
        "",
        "Начальная модель seed=2026; новые сценарии seed=23026+r, планировщик=32026+r, порядок стратегий=42026+r. Дедлайн 1.5 s, timeout RPC 1.2 s, стоимость переключения 0.025 s. Профили исполнялись блоками в заранее перемешанном порядке: "
        + ", ".join(profiles)
        + ".",
        "",
        "## Физическое время до качества",
        "",
        "Медианы T_epsilon в секундах; недостигнутые пороги не подменяются последним временем.",
        "",
        "| Стратегия | unlimited | edge_limited | verifier_limited |",
        "|---|---:|---:|---:|",
    ]
    for policy in POLICIES:
        cells = []
        for profile in ("unlimited", "edge_limited", "verifier_limited"):
            values = [
                float(s["t_epsilon_wall_s"])
                for s in summary
                if s["policy"] == policy and s["network_profile"] == profile and int(s["reached"])
            ]
            cells.append(f"{median(values):.3f}" if values else "не достигнут")
        lines.append("| " + policy + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"![Время до качества]({output}/network_time_to_quality.png)",
        "",
        "## Проверка зависимости B/P от ограниченного потока",
        "",
        "edge_limited: 40 MB/s на общий worker→edge; verifier_limited: 40 MB/s на общий вход проверяющего. В обоих ограниченных профилях задержка каждого запроса и ответа составляет 2 ms. В unlimited заданного ограничения и задержки нет. MB означает миллион байт.",
        "",
        "Сравнения P−B выполняются внутри каждого профиля на одинаковых сценариях.",
        "",
        "| Профиль | Контраст | Полных пар | Средняя разность, s | Описательный bootstrap 95%, s |",
        "|---|---|---:|---:|---:|",
    ]
    for row in paired:
        if row["contrast"] not in ("F+P - F+B", "R+P - R+B"):
            continue
        interval = (
            f"[{row['bootstrap95_low_s']:+.3f}; {row['bootstrap95_high_s']:+.3f}]"
            if row["bootstrap95_low_s"] is not None
            else "—"
        )
        value = f"{row['mean_delta_wall_s']:+.3f}" if row["mean_delta_wall_s"] is not None else "—"
        lines.append(
            f"| {row['network_profile']} | {row['contrast']} | {row['complete_pairs']} | {value} | {interval} |"
        )
    lines += [
        "",
        f"![Парные контрасты]({output}/network_projection_contrasts.png)",
        "",
        "Отрицательная разность означает меньшее время у P. Интервалы предварительные, по доступным полным парам, без поправки на множественные сравнения. Это не подтверждение универсального преимущества онлайн-алгоритма.",
        "",
        "## Калибровка и корректность",
        "",
    ]
    if args.calibration:
        calibration = read_csv(args.calibration)
        write_csv(output / "calibration.csv", calibration)
        lines += [
            "Общий предел 40 MB/s, фактическая передача по TCP, отдельный процесс на поток:",
            "",
            "| Процессов | Медиана суммарной скорости, MB/s |",
            "|---|---:|",
        ]
        for n in sorted({int(r["streams"]) for r in calibration}):
            value = median(float(r["measured_mb_s"]) for r in calibration if int(r["streams"]) == n)
            lines.append(f"| {n} | {value:.2f} |")
        lines.append("")
    lines += [
        f"В обучении выполнено {len(rounds)} раундов и {sum(int(r['accepted']) for r in rounds)} принятых обновлений; тайм-аутов: {sum(int(r['timeouts']) for r in rounds)}. Счётчики отправленных/полученных кадров совпали во всех раундах, для которых трафик не помечен нижней оценкой. SHA256 всех снимков исходников проверены.",
        "",
        "Проверяются подмена, повторы, отсутствие участника, неравные блоки, восстановление резервной копии, завершение процесса, переполнение, ограничение общей полосы и остановка отправителей по дедлайну.",
        "",
        "## Изменённые файлы и запуск",
        "",
        "Добавлены `tcp_network.py`, `tcp_network_calibrate.py`, `tcp_network_sweep.py`, `tcp_network_analyze.py`, `configs/tcp_network_pilot.json`, `tests/test_tcp_network.py`, `NETWORK_EXPERIMENTS.md`. Обновлены `tcp_transport.py`, `tcp_runtime.py`, `tcp_closed_loop_pilot.py` и тесты транспорта. Исходные серии измерений сохранены.",
        "",
        "```bash",
        "cd /Users/mariiapavlova/PycharmProjects/adaptive_planning",
        ".venv/bin/python -m unittest discover -s tests -v",
        ".venv/bin/python tcp_network_sweep.py --config configs/tcp_network_pilot.json",
        "```",
        "",
        "## Что осталось проверить",
        "",
        "Профили имеют постоянные параметры внутри запуска. Это этап проверки зависимости стоимости от ограниченного потока; адаптация к меняющейся полосе внутри обучения ещё не проверена. Профили выполнялись блоками, поэтому сравнения между профилями могут зависеть от времени и температуры компьютера. Разница unlimited/limited также включает заданные 2 ms, а не только полосу.",
        "",
        "Pacing ограничивает прикладные кадры, не пакеты TCP/IP. Кванты 16 KiB, буферизация и планировщик ОС допускают локальные колебания скорости. Абсолютный deadline_ns применим только к общим монотонным часам одного хоста. Для реального распределённого запуска нужны машины, аутентифицированный транспорт и переносимый бюджет времени. Эти измерения выполнены на текущем Mac; другой хост не проверялся.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    shutil.copy2(
        ROOT / "docs/experiments/NETWORK_EXPERIMENTS.md", output / "NETWORK_EXPERIMENTS.md"
    )
    (output / "metadata.json").write_text(json.dumps(checked, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(output / "measurements.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sweep.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(sweep))
        if args.calibration:
            archive.write(args.calibration, "calibration.csv")
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
