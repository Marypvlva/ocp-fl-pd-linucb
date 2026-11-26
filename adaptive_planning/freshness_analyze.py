"""Audit the causal age-mixture policy and its predeclared paired contrast."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, median

import numpy as np

from .adaptive_scheduler import Observation, transfer_prior
from .telemetry_alignment_analyze import comparisons, validate as validate_matched
from .telemetry_robustness_analyze import write_csv

POLICIES = ("network_freshness", "network_aligned", "network_prior")


def validate(folder):
    meta, summary, rows = validate_matched(folder, family="telemetry_freshness", policies=POLICIES)
    if meta["primary_contrast"] != "network_freshness - network_aligned" or set(meta["modes"]) != {
        "exact",
        "stale_noise",
    }:
        raise ValueError("Unexpected primary contrast or sensor conditions")
    histories = defaultdict(dict)
    for row in rows:
        if row["policy"] != "network_freshness":
            continue
        key = (row["workers"], row["repeat"], row["condition"])
        capture = int(row["telemetry_source_attempt"])
        obs = Observation(**json.loads(row["network_observation"]))
        history = histories[key]
        if capture not in history:
            history[capture] = obs
            while len(history) > 16:
                del history[next(iter(history))]
        age = int(row["attempt"]) - capture
        action = row["action"]
        latest = transfer_prior(action, obs)
        # Independent formula, using raw reports rather than the planner's state.
        average = fmean(transfer_prior(action, old) for old in history.values())
        expected = latest if age == 0 else latest / (1 + age) + age / (1 + age) * average
        decision = json.loads(row["decision"])
        if (
            decision["history_capture_ids"] != list(history)
            or decision["telemetry_age_rounds"] != age
            or not math.isclose(decision["latest_report_weight"], 1 / (1 + age), rel_tol=1e-12)
            or not math.isclose(
                decision["transmission_budget_s"], expected, rel_tol=1e-12, abs_tol=1e-10
            )
        ):
            raise ValueError("Age mixture uses unavailable or duplicate reports")
    return meta, summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    folder, output = args.folder.resolve(), args.output.resolve()
    meta, summary, rows = validate(folder)
    paired = comparisons(
        summary,
        meta["modes"],
        candidate="network_freshness",
        references=("network_aligned", "network_prior"),
        seed=184026,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired)
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(meta["modes"])
    counts = sorted({int(r["workers"]) for r in summary})
    fig, axes = plt.subplots(
        len(counts), len(modes), figsize=(11.8, 4.5 * len(counts)), squeeze=False
    )
    for i, count in enumerate(counts):
        for ax, mode in zip(axes[i], modes):
            for j, (policy, color) in enumerate(
                zip((*POLICIES, "F+P"), ("#398575", "#d17b38", "#4278ad", "#848b93"))
            ):
                case = "control" if policy == "F+P" else mode
                values = [
                    float(r["t_epsilon_wall_s"])
                    for r in summary
                    if int(r["workers"]) == count
                    and r["condition"] == case
                    and r["policy"] == policy
                    and int(r["reached"])
                ]
                ax.scatter(
                    j + np.linspace(-0.07, 0.07, len(values)), values, color=color, alpha=0.65, s=24
                )
                if values:
                    ax.scatter([j], [median(values)], color=color, marker="D", s=45)
            ax.set_xticks(range(4), ["age mixture", "aligned", "frozen", "fixed F+P"])
            ax.set(title=f"{count} workers · {mode}", ylabel="T_epsilon · physical localhost (s)")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Freshness-aware action selection · independent network traces")
    fig.text(
        0.5,
        0.012,
        "Dots: reached runs. Diamonds: medians. Fixed control is shared across sensor conditions, not measured twice.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output / "freshness_time_to_quality.png", dpi=180)
    plt.close(fig)
    primaries = [r for r in paired if r["primary"]]
    fmt = lambda v: "—" if v is None else f"{v:+.3f}"
    lines = ["# adaptive_planning: возраст телеметрии при выборе действия", ""]
    for p in primaries:
        mean, low, high = (
            p[k] for k in ("mean_delta_wall_s", "bootstrap95_low_s", "bootstrap95_high_s")
        )
        if low is None:
            conclusion = "Недостаточно полных пар для интервала."
        elif high < 0:
            conclusion = "В этой серии новый вариант быстрее aligned. Преимущество над фиксированным F+P оценивается отдельно."
        elif low > 0:
            conclusion = (
                "В этой серии новый вариант медленнее aligned; гипотеза ускорения не подтвердилась."
            )
        else:
            conclusion = "Интервал пересекает ноль: убедительного ускорения относительно aligned не обнаружено."
        lines += [
            f"Первичный контраст freshness−aligned, stale_noise, {p['workers']} участников: "
            f"{fmt(mean)} s; описательный bootstrap 95% [{fmt(low)}; {fmt(high)}]. {conclusion}",
            "",
        ]
    c = meta["config"]
    lines += [
        f"Целевой loss≤{c['target_loss']} достигнут в {sum(int(s['reached']) for s in summary)}/{len(summary)} "
        f"запусках; {len(rows)} реальных раундов, {sum(int(r['accepted']) for r in rows)} принятых обновлений, "
        f"{sum(int(r['timeouts']) for r in rows)} тайм-аутов.",
        "",
        "## Что изменено",
        "",
        "Стоимость передачи при выборе действия теперь смешивает последний отчёт и среднюю оценку по последним "
        "16 различным меткам захвата. Для возраста d вес последнего отчёта равен 1/(1+d). "
        "При d=0 выбор и обучение совпадают с aligned при общей последовательности исходов. "
        "Это эвристика неопределённости, а не калиброванная вероятность текущего профиля. В историях нет "
        "истинных текущих/будущих профилей, невыбранных действий или известных длительностей фаз. "
        "Повторяемый стартовый отчёт учитывается один раз. Остаточная стоимость обучается по совпадающим "
        "меткам раунда, успех — сразу; exploration 5% и остальные priors сохранены.",
        "",
        "## Заранее заданная проверка",
        "",
        f"Seed новых трасс {c['seed']}+r; модели {c['model_seeds']}, learning rate {c['learning_rate']}, "
        f"до {c['rounds']} попыток. Модели и шаг сохранены из предыдущей диагностики; новыми являются "
        "трассы, отказы, шум, exploration и порядок запусков. Все модели включены без отбора по исходам. "
        "exact/stale_noise × freshness/aligned/frozen и один фактический фиксированный F+P на блок. "
        "Семь случаев перемешаны; общий контроль не дублируется в исходных измерениях.",
        "",
        "stale_noise: лаг 2 попытки, lognormal σ=0.35; первые отчёты повторяются при заполнении канала. "
        "Профили меняются каждые 4–8 попыток; общий предел соответствующего потока 40 MB/s, 2 ms на сообщение. "
        "Сеть передаёт данные по TCP, датчик синтетический. Нулевой флаг unlimited не искажается.",
        "",
        f"![Время до качества]({output}/freshness_time_to_quality.png)",
        "",
        "## Парные разности freshness−сравнение",
        "",
        "| Участников | Датчик | Сравнение | Полные пары | Разность, s | Bootstrap 95%, s |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for p in paired:
        lines.append(
            f"| {p['workers']} | {p['condition']} | {p['second_policy']} | {p['complete_pairs']}/{p['pairs']} | "
            f"{fmt(p['mean_delta_wall_s'])} | [{fmt(p['bootstrap95_low_s'])}; {fmt(p['bootstrap95_high_s'])}] |"
        )
    lines += [
        "",
        "Отрицательная разность означает выигрыш freshness. Единственный первичный контраст основной "
        "серии — freshness−aligned в stale_noise на 12 участниках. Остальные интервалы описательные, без поправки "
        "на множественные сравнения. Парный bootstrap среднего: 20000 выборок, seed 184026. Мощность не оценивалась. "
        "Цензурированное T не заменяется временем остановки; полные пары могут давать смещение.",
        "",
        "## Аудит и границы",
        "",
        "Воспроизведены все решения и состояния планировщиков по записанным исходам; независимо пересчитаны "
        "суммы matched residual, история доступных отчётов, её ограничение 16 метками и веса смеси. Проверены "
        "трассы по seed, модели, выбранная обратная связь, источник/возраст отчётов, SHA256 исходников, "
        "состав блоков и счётчики трафика. Старые серии не объединены с новой.",
        "",
        "Один Mac и одна архитектура MLP. Не проверены WAN, несколько хостов, измерительные сетевые пробы, "
        "потери отчётов и неизвестный/поддельный момент захвата. История предполагает относительную устойчивость "
        "распределения сетевых условий; при новой длительной фазе она может запаздывать. Улучшение на этих "
        "трассах не является общей гарантией. Цель по validation loss используется для остановки, не является "
        "независимой оценкой обобщения модели.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        ".venv/bin/python project.py doctor",
        ".venv/bin/python project.py test",
        ".venv/bin/python project.py smoke",
        ".venv/bin/python project.py study",
        ".venv/bin/python project.py verify RESULTS_FOLDER",
        ".venv/bin/python project.py report RESULTS_FOLDER --output NEW_REPORT_FOLDER",
        "```",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    validation = dict(
        status="passed",
        runs=len(summary),
        reached=sum(int(r["reached"]) for r in summary),
        rounds=len(rows),
        accepted=sum(int(r["accepted"]) for r in rows),
        timeouts=sum(int(r["timeouts"]) for r in rows),
        primaries=primaries,
        causal_decisions_replayed=True,
        independent_mixture_audit=True,
    )
    (output / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation), flush=True)
    return output


if __name__ == "__main__":
    main()
