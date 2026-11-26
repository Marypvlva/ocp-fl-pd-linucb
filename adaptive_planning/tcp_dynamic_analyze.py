"""Paired analysis and figures for dynamic TCP training experiments."""

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

from .tcp_dynamic_experiment import POLICIES
from .tcp_network import PROFILES

CONTRASTS = (
    ("network_online", "network_rule"),
    ("network_online", "network_prior"),
    ("network_online", "online"),
    ("network_online", "rule"),
    ("network_rule", "rule"),
    *(("network_online", a) for a in POLICIES[:4]),
    *(("network_rule", a) for a in POLICIES[:4]),
)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparisons(summary, primary="network_online - network_rule"):
    indexed = {}
    for row in summary:
        key = (int(row["workers"]), int(row["repeat"]), row["policy"])
        if key in indexed:
            raise ValueError(f"Duplicate run: {key}")
        indexed[key] = row
    rng = np.random.default_rng(143026)
    results = []
    for workers in sorted({key[0] for key in indexed}):
        repeats = sorted({key[1] for key in indexed if key[0] == workers})
        for a, b in CONTRASTS:
            pairs = [
                (indexed[(workers, i, a)], indexed[(workers, i, b)])
                for i in repeats
                if (workers, i, a) in indexed and (workers, i, b) in indexed
            ]
            complete = [(x, y) for x, y in pairs if int(x["reached"]) and int(y["reached"])]
            delta = np.array(
                [float(x["t_epsilon_wall_s"]) - float(y["t_epsilon_wall_s"]) for x, y in complete]
            )
            low = high = None
            if len(delta) >= 2:
                samples = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975]).tolist()
            results.append(
                dict(
                    workers=workers,
                    contrast=f"{a} - {b}",
                    primary=int(f"{a} - {b}" == primary),
                    pairs=len(pairs),
                    complete_pairs=len(complete),
                    first_reached=sum(int(x["reached"]) for x, _ in pairs),
                    second_reached=sum(int(y["reached"]) for _, y in pairs),
                    mean_delta_wall_s=float(delta.mean()) if len(delta) else None,
                    bootstrap95_low_s=low,
                    bootstrap95_high_s=high,
                    note="predeclared primary; secondary exploratory; complete-pair percentile bootstrap",
                )
            )
    return results


def validate(folder):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["status"] != "completed":
        raise ValueError("Experiment did not complete")
    summary, rounds = read_csv(folder / "summary.csv"), read_csv(folder / "rounds.csv")
    if any(row.get("telemetry_mode", "exact") != "exact" for row in rounds):
        raise ValueError("Use telemetry_robustness_analyze.py for perturbed or mixed telemetry")
    with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
        for name, digest in meta["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Source snapshot mismatch: {name}")
    traces = {}
    for row in rounds:
        workers, repeat = int(row["workers"]), int(row["repeat"])
        key = (workers, repeat)
        if key not in traces:
            traces[key] = json.loads((folder / f"trace_w{workers}_r{repeat}.json").read_text())
        step = traces[key][int(row["attempt"]) - 1]
        if row["profile"] != step["profile"] or json.loads(row["injected_fault"]) != step["fault"]:
            raise ValueError("Pairing/trace mismatch")
        observation = json.loads(row["network_observation"])
        actual = PROFILES[step["profile"]]
        if any(
            observation[key] != getattr(actual, key)
            for key in ("model_mbps", "edge_mbps", "verifier_mbps", "latency_ms")
        ):
            raise ValueError("Advertised profile mismatch")
        if not int(row["traffic_is_lower_bound"]) and row["app_tx_bytes"] != row["app_rx_bytes"]:
            raise ValueError("Frame byte counters differ")
    for row in summary:
        if row["policy"] == "network_online":
            chosen = [
                r
                for r in rounds
                if (r["workers"], r["repeat"], r["policy"])
                == (row["workers"], row["repeat"], row["policy"])
            ]
            if [int(r["feedback_updates"]) for r in chosen] != list(range(1, len(chosen) + 1)):
                raise ValueError("Online feedback count mismatch")
        elif row["policy"] == "network_prior":
            chosen = [
                r
                for r in rounds
                if (r["workers"], r["repeat"], r["policy"])
                == (row["workers"], row["repeat"], row["policy"])
            ]
            if any(int(r["feedback_updates"]) != 0 for r in chosen):
                raise ValueError("Frozen prior received feedback")
    return meta, summary, rounds


def plot(summary, rounds, contrasts, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = sorted({int(r["workers"]) for r in summary})
    policies = [p for p in POLICIES if any(r["policy"] == p for r in summary)]
    fig, axes = plt.subplots(1, len(counts), figsize=(max(8, 7 * len(counts)), 4.8), squeeze=False)
    for ax, count in zip(axes[0], counts):
        for i, policy in enumerate(policies):
            values = [
                float(r["t_epsilon_wall_s"])
                for r in summary
                if int(r["workers"]) == count and r["policy"] == policy and int(r["reached"])
            ]
            if values:
                colour = (
                    "#d26832"
                    if policy == "network_online"
                    else "#247b65"
                    if policy == "network_rule"
                    else "#617b9a"
                )
                ax.scatter(
                    i + np.linspace(-0.12, 0.12, len(values)),
                    values,
                    color=colour,
                    alpha=0.55,
                    s=18,
                )
                ax.scatter([i], [median(values)], color=colour, marker="D", s=48)
        ax.set_xticks(range(len(policies)), policies, rotation=40, ha="right")
        ax.set(
            title=f"{count} workers · changing bottleneck",
            ylabel="T_epsilon, localhost wall-clock (s)",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        0.01,
        "Dots: reached runs; diamonds: medians. Fixed initial model; paired attempt-indexed traces.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / "dynamic_time_to_quality.png", dpi=180)
    plt.close(fig)
    # Show decisions on a single preselected trace, not the best-looking run.
    count = counts[-1]
    comparison = "network_prior" if "network_prior" in policies else "network_rule"
    selected = [
        r
        for r in rounds
        if int(r["workers"]) == count
        and int(r["repeat"]) == 0
        and r["policy"] in (comparison, "network_online")
    ]
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]}
    )
    trace_rows = sorted(
        {int(r["attempt"]): r for r in selected}.values(), key=lambda r: int(r["attempt"])
    )
    axes[0].step(
        [int(r["attempt"]) for r in trace_rows],
        [0 if r["profile"] == "edge_limited" else 1 for r in trace_rows],
        where="post",
    )
    axes[0].set_yticks([0, 1], ["edge cap", "verifier cap"])
    axes[0].set_title(f"{count} workers · repeat 0 · advertised bottleneck and selected actions")
    for ax, policy in zip(axes[1:], (comparison, "network_online")):
        rr = [r for r in selected if r["policy"] == policy]
        ax.step(
            [int(r["attempt"]) for r in rr], [POLICIES.index(r["action"]) for r in rr], where="post"
        )
        rejected = [r for r in rr if not int(r["accepted"])]
        ax.scatter(
            [int(r["attempt"]) for r in rejected],
            [POLICIES.index(r["action"]) for r in rejected],
            color="#be4539",
            marker="x",
            s=25,
        )
        ax.set_yticks(range(4), POLICIES[:4])
        ax.set_ylabel(policy)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Attempt · red x: rejected round")
    fig.tight_layout()
    fig.savefig(output / "dynamic_decisions.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tests-log", type=Path)
    args = parser.parse_args(argv)
    folder, output = args.folder.resolve(), args.output.resolve()
    meta, summary, rounds = validate(folder)
    paired = paired_comparisons(summary, meta["primary_contrast"])
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / "summary.csv", summary)
    save_csv(output / "paired.csv", paired)
    plot(summary, rounds, paired, output)
    config = meta["config"]
    tests = ""
    if args.tests_log:
        log = args.tests_log.read_text()
        match = re.search(r"Ran (\d+) tests", log)
        if not match or not log.rstrip().endswith("OK"):
            raise ValueError("Test log does not confirm success")
        tests = f"Пройдено автоматических тестов: {match.group(1)}."
        shutil.copy2(args.tests_log, output / "tests.log")
    lines = [
        "# adaptive_planning: изменяющиеся ограничения сети",
        "",
        "Реализована смена ограниченного потока между раундами настоящего TCP-обучения. Добавлены простое правило и онлайн-планировщик, использующие текущие объявленные параметры сети и прошлую обратную связь.",
        "",
        f"Серия: {config['stage']}, seed={config['seed']}, повторов={config['repeats']}, "
        f"порог loss={config['target_loss']}, исходная модель seed={config['model_seed']}. "
        f"Порог достигнут в {sum(int(r['reached']) for r in summary)}/{len(summary)} запусках. {tests}",
        "",
        f"Первичный контраст, заданный до серии: **{meta['primary_contrast']}**. Отрицательное значение означает выигрыш онлайн-алгоритма по времени до качества. Остальные сравнения описательные.",
        "",
        "## Фактическое время до качества",
        "",
        "Медианы T_epsilon (s), доля достигших порога и медиана числа переключений:",
        "",
        "| Участников | Стратегия | T_epsilon, s | Достигли порога | Переключений | Трафик до остановки, MB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for count in sorted({int(r["workers"]) for r in summary}):
        for policy in POLICIES:
            runs = [r for r in summary if int(r["workers"]) == count and r["policy"] == policy]
            if not runs:
                continue
            completed = [r for r in runs if int(r["reached"])]
            value = (
                f"{median(float(r['t_epsilon_wall_s']) for r in completed):.3f}"
                if completed
                else "—"
            )
            lines.append(
                f"| {count} | {policy} | {value} | {len(completed)}/{len(runs)} | "
                f"{median(int(r['switches']) for r in runs):.1f} | "
                f"{median(float(r['app_tx_bytes']) / 1e6 for r in runs):.1f} |"
            )
    lines += [
        "",
        f"![Время до качества]({output}/dynamic_time_to_quality.png)",
        "",
        "## Парные сравнения",
        "",
        "| Участников | Контраст | Полных пар | Средняя разность, s | Описательный bootstrap 95%, s |",
        "|---|---|---:|---:|---:|",
    ]
    for row in paired:
        mean = f"{row['mean_delta_wall_s']:+.3f}" if row["mean_delta_wall_s"] is not None else "—"
        ci = (
            f"[{row['bootstrap95_low_s']:+.3f}; {row['bootstrap95_high_s']:+.3f}]"
            if row["bootstrap95_low_s"] is not None
            else "—"
        )
        prefix = "**" if row["primary"] else ""
        lines.append(
            f"| {row['workers']} | {prefix}{row['contrast']}{prefix} | {row['complete_pairs']} | {mean} | {ci} |"
        )
    lines += [
        "",
        "Недостигнутый порог не заменяется последним временем. Интервалы рассчитаны по полным парам; при цензурировании они могут быть смещены. Для вторичных сравнений нет поправки на множественные проверки. Число повторов не является гарантией мощности.",
        "",
        "## Как устроен эксперимент",
        "",
        f"Профили edge_limited/verifier_limited чередуются после {config['phase_min']}–{config['phase_max']} попыток. Общий предел соответствующего потока — 40 MB/s, задержка сообщения — 2 ms. Профиль неизменен в пределах раунда. Одинаковые префиксы трассы и отказы используются для всех стратегий данного повтора.",
        "",
        "Планировщику доступны текущие объявленные ограничения, а не будущая трасса. Это идеализированная мгновенная телеметрия без шума. Принимаемый вектор поступает только от edge-процессов и проходит B/P. Обе новые стратегии учитывают стоимость переключения; network_online обновляет статистику только выбранного действия.",
        "",
        "network_rule минимизирует грубую оценку передачи и постоянных издержек с текущей ценой переключения. После наблюдённой edge-ошибки пробует резерв, после отсутствующего P-участника — B. network_online использует ту же оценку передачи как prior, учит остаточное время и вероятность успеха и исследует действия с вероятностью 5%. Это эвристические оценки, не замена физического измерения времени.",
        "",
        "network_prior — отдельная проверка без обучения: те же priors, правило выбора, exploration и RNG, что у network_online, но observe() ничего не меняет. Значение feedback_updates остаётся нулём. Эта проверка использует независимые от основной серии трассы.",
        "",
        f"Дедлайн раунда={config['deadline']} s, RPC timeout={config['rpc_timeout']} s, физическое искусственное ожидание переключения={config['switch_seconds']} s. T_epsilon измеряется непрерывно до validation, подтвердившей порог; включены предыдущая validation, телеметрия, выбор и журналирование, исключены startup и одинаковый прогрев.",
        "",
        f"![Решения на заранее выбранном повторе 0]({output}/dynamic_decisions.png)",
        "",
        "## Проверки и ограничения",
        "",
        f"Выполнено раундов: {len(rounds)}; принятых обновлений: {sum(int(r['accepted']) for r in rounds)}; "
        f"тайм-аутов: {sum(int(r['timeouts']) for r in rounds)}. Проверены снимки исходников, соответствие профиля и инъекции каждой строки трассе и число обновлений обратной связи. Счётчики tx/rx совпадают там, где трафик не помечен нижней оценкой.",
        "",
        "Привязка профиля к номеру раунда общая для процессов и неизменяема: поздний ответ не получает параметры следующего раунда. Никакая дополнительная модельная задержка не прибавляется к фактическому времени. Состояние раунда сохраняется в JSONL после каждой попытки; завершающие CSV формируются отдельно.",
        "",
        "Ограничения меняются между раундами и индексируются попытками, а не общим wall-clock. Разные стратегии могут пройти разное число попыток и фаз до порога. Измерения выполнены на одном Mac с прикладным pacing, без WAN и изменений посреди TCP-кадра. Исходная модель фиксирована; обобщение на другие инициализации не проверено. Статистические интервалы условны на этих сценариях и окружении ОС.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "cd /Users/mariiapavlova/PycharmProjects/adaptive_planning",
        ".venv/bin/python -m unittest discover -s tests -v",
        ".venv/bin/python tcp_dynamic_experiment.py --config configs/tcp_dynamic_eval.json",
        "```",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    shutil.copy2(folder / "metadata.json", output / "metadata.json")
    with zipfile.ZipFile(output / "measurements.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in folder.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(folder))
        if args.tests_log:
            archive.write(args.tests_log, "tests.log")
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
