"""Audit matched cost feedback and analyze the predeclared alignment contrast."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from .adaptive_scheduler import (
    NetworkOnlinePlanner,
    FrozenNetworkPlanner,
    ACTIONS,
    overhead_prior,
    transfer_prior,
    Observation,
)
from .aligned_scheduler import AlignedNetworkPlanner
from .freshness_scheduler import FreshnessMixturePlanner
from .tcp_dynamic_experiment import make_trace
from .telemetry_robustness_analyze import validate as validate_common, write_csv

POLICIES = ("network_aligned", "network_online", "network_prior")


def close_tree(actual, expected):
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(close_tree(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(close_tree(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, (float, int)):
        return isinstance(actual, (float, int)) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-10
        )
    return actual == expected


def validate(folder, family="telemetry_alignment", policies=POLICIES):
    meta, summary, rows = validate_common(folder, family=family)
    config = meta["config"]
    expected = {
        (w, r, mode, policy)
        for w in config["workers"]
        for r in range(config["repeats"])
        for mode in meta["modes"]
        for policy in policies
    }
    expected |= {
        (w, r, "control", meta["shared_control_policy"])
        for w in config["workers"]
        for r in range(config["repeats"])
    }

    def key(row):
        return (int(row["workers"]), int(row["repeat"]), row["condition"], row["policy"])

    if len(summary) != len(expected) or {key(s) for s in summary} != expected:
        raise ValueError("Missing, duplicated or unplanned cases")
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    if set(grouped) != expected:
        raise ValueError("Missing round journals")
    for workers in config["workers"]:
        for repeat in range(config["repeats"]):
            trace = [
                asdict(s)
                for s in make_trace(
                    config["seed"] + repeat,
                    config["rounds"],
                    workers,
                    config["scenario"],
                    config["phase_min"],
                    config["phase_max"],
                )
            ]
            # JSON normalizes tuple-valued faults to lists.
            trace = json.loads(json.dumps(trace))
            stored = json.loads((folder / f"trace_w{workers}_r{repeat}.json").read_text())
            if stored != trace:
                raise ValueError("Trace differs from predeclared RNG seed")
    for s in summary:
        records = grouped[key(s)]
        policy = s["policy"]
        planner = None
        if policy in policies:
            cls = {
                "network_aligned": AlignedNetworkPlanner,
                "network_online": NetworkOnlinePlanner,
                "network_prior": FrozenNetworkPlanner,
                "network_freshness": FreshnessMixturePlanner,
            }[policy]
            planner = cls(config["seed"] + 11000 + int(s["repeat"]), config["switch_seconds"])
        previous, previous_errors = None, []
        for attempt, row in enumerate(records, 1):
            if int(row["attempt"]) != attempt:
                raise ValueError("Noncontiguous attempt prefix")
            obs = Observation(**json.loads(row["network_observation"]))
            trouble = any(
                e.startswith(("e0:", "e1:", "e2:", "e3:"))
                or "Подмена агрегированной суммы" in e
                or "Некорректная агрегированная" in e
                for e in previous_errors
            )
            missing = any("Missing projection participant" in e for e in previous_errors)
            if obs.edge_trouble != trouble or obs.projection_missing != missing:
                raise ValueError("Observation uses unavailable protocol outcomes")
            if planner:
                if policy in ("network_aligned", "network_freshness"):
                    planner.begin_round(obs, int(row["telemetry_source_attempt"]), attempt)
                action = planner.choose(obs, previous)
                if action != row["action"] or not close_tree(
                    json.loads(row["decision"]), planner.last_decision
                ):
                    raise ValueError("Decision cannot be causally replayed")
                outcome = (
                    row["action"],
                    bool(int(row["accepted"])),
                    float(row["cycle_wall_s"]),
                    float(row["switch_wall_s"]),
                )
                if policy in ("network_aligned", "network_freshness"):
                    planner.observe_round(*outcome)
                    if (
                        int(row["cost_feedback_updates"]) != planner.cost_feedback_count()
                        or int(row["pending_cost_outcomes"]) != len(planner.pending)
                        or int(row["cost_feedback_source_attempt"]) != planner.last_cost_capture
                    ):
                        raise ValueError("Misaligned or duplicate cost feedback")
                    # Independent sum: each delivered report is applied to the
                    # stored action/cost in its CAPTURE round, never this round.
                    captures = {}
                    for delivered in records[:attempt]:
                        captures[int(delivered["telemetry_source_attempt"])] = delivered
                    for action in ACTIONS:
                        matched = [
                            (records[c - 1], delivered)
                            for c, delivered in captures.items()
                            if records[c - 1]["action"] == action
                        ]
                        cost_sum = 2 * overhead_prior(action)
                        for original, delivered in matched:
                            cost_sum += max(
                                0.001,
                                float(original["cycle_wall_s"])
                                - float(original["switch_wall_s"])
                                - transfer_prior(
                                    action,
                                    Observation(**json.loads(delivered["network_observation"])),
                                ),
                            )
                        stat = planner.stats[action]
                        if stat["cost_n"] != len(matched) or not math.isclose(
                            stat["residual_sum"], cost_sum, rel_tol=1e-12, abs_tol=1e-10
                        ):
                            raise ValueError("Independent matched-residual ledger mismatch")
                else:
                    planner.observe(obs, *outcome)
                if int(row["feedback_updates"]) != planner.feedback_count() or not close_tree(
                    json.loads(row["planner_statistics"]), planner.stats
                ):
                    raise ValueError("Planner statistics mismatch")
            elif row["action"] != meta["shared_control_policy"]:
                raise ValueError("Fixed control changed action")
            previous, previous_errors = row["action"], json.loads(row["errors"])
        last = records[-1]
        hit = float(last["validation_loss"]) <= config["target_loss"]
        if (
            len(records) != int(s["attempts"])
            or sum(int(r["accepted"]) for r in records) != int(s["accepted_updates"])
            or bool(int(s["reached"])) != hit
            or (
                hit
                and not math.isclose(
                    float(s["t_epsilon_wall_s"]), float(last["quality_observed_wall_s"])
                )
            )
        ):
            raise ValueError("Summary disagrees with raw rounds")
    return meta, summary, rows


def comparisons(
    summary,
    modes,
    candidate="network_aligned",
    references=("network_online", "network_prior"),
    seed=174026,
):
    index = {(int(r["workers"]), int(r["repeat"]), r["condition"], r["policy"]): r for r in summary}
    if len(index) != len(summary):
        raise ValueError("Duplicated raw run")
    rng = np.random.default_rng(seed)
    result = []
    for workers in sorted({k[0] for k in index}):
        repeats = sorted({k[1] for k in index if k[0] == workers})
        for mode in modes:
            for other_mode, other_policy in (*[(mode, p) for p in references], ("control", "F+P")):
                pairs = [
                    (
                        index[(workers, r, mode, candidate)],
                        index[(workers, r, other_mode, other_policy)],
                    )
                    for r in repeats
                ]
                if any(
                    (a["model_seed"], a["learning_rate"]) != (b["model_seed"], b["learning_rate"])
                    for a, b in pairs
                ):
                    raise ValueError("Mismatched initial models")
                delta = np.array(
                    [
                        float(a["t_epsilon_wall_s"]) - float(b["t_epsilon_wall_s"])
                        for a, b in pairs
                        if int(a["reached"]) and int(b["reached"])
                    ]
                )
                low = high = None
                if len(delta) >= 2:
                    low, high = np.quantile(
                        rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1),
                        [0.025, 0.975],
                    ).tolist()
                result.append(
                    dict(
                        workers=workers,
                        condition=mode,
                        first_policy=candidate,
                        second_condition=other_mode,
                        second_policy=other_policy,
                        primary=int(mode == "stale_noise" and other_policy == references[0]),
                        pairs=len(pairs),
                        complete_pairs=len(delta),
                        first_reached=sum(int(a["reached"]) for a, _ in pairs),
                        second_reached=sum(int(b["reached"]) for _, b in pairs),
                        mean_delta_wall_s=float(delta.mean()) if len(delta) else None,
                        bootstrap95_low_s=low,
                        bootstrap95_high_s=high,
                    )
                )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    folder, output = args.folder.resolve(), args.output.resolve()
    meta, summary, rows = validate(folder)
    paired = comparisons(summary, meta["modes"])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired)
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(meta["modes"])
    fig, axes = plt.subplots(
        1, len(modes), figsize=(5.8 * len(modes), 4.8), squeeze=False, sharey=True
    )
    for ax, mode in zip(axes[0], modes):
        for i, (policy, colour) in enumerate(
            zip((*POLICIES, "F+P"), ("#33876f", "#d26832", "#3676a8", "#7a818b"))
        ):
            case = "control" if policy == "F+P" else mode
            group = [
                r
                for r in summary
                if r["condition"] == case and r["policy"] == policy and int(r["reached"])
            ]
            times = [float(r["t_epsilon_wall_s"]) for r in group]
            ax.scatter(
                i + np.linspace(-0.08, 0.08, len(times)), times, color=colour, alpha=0.6, s=24
            )
            if times:
                ax.scatter([i], [median(times)], color=colour, marker="D", s=50)
        ax.set_xticks(range(4), ["aligned", "original", "frozen", "fixed F+P"])
        ax.set(title=mode, ylabel="T_epsilon · physical localhost (s)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Causal matching of sensor capture and cost feedback · 12 workers")
    fig.text(
        0.5,
        0.012,
        "Dots: reached runs. Diamonds: medians. The same shared fixed control appears in both panels.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output / "alignment_time_to_quality.png", dpi=180)
    plt.close(fig)
    primary = next(p for p in paired if p["primary"])
    fmt = lambda value: "—" if value is None else f"{value:+.3f}"
    c = meta["config"]
    lines = [
        "# adaptive_planning: согласование телеметрии и обратной связи",
        "",
        f"Первичный контраст aligned−original в stale_noise: {fmt(primary['mean_delta_wall_s'])} s; "
        f"описательный парный bootstrap 95% [{fmt(primary['bootstrap95_low_s'])}; {fmt(primary['bootstrap95_high_s'])}].",
        "",
        f"Достигли loss≤{c['target_loss']}: {sum(int(s['reached']) for s in summary)}/{len(summary)} запусков. "
        f"Реальных раундов {len(rows)}, принятых обновлений {sum(int(r['accepted']) for r in rows)}, "
        f"тайм-аутов {sum(int(r['timeouts']) for r in rows)}.",
        "",
        "## Поправка",
        "",
        "Исходный learner вычитает из измеренного времени раунда prior передачи по доставленному отчёту сети. "
        "При задержке отчёт относится к другому раунду. Новый network_aligned хранит действие и стоимость до прихода "
        "отчёта с той же меткой попытки, затем обновляет остаточное время один раз. Успех протокола учитывается сразу; "
        "для вероятности успеха и остаточного времени используются отдельные счётчики. Повторяемый стартовый отчёт "
        "не даёт повторного обучения. Недоставленные к остановке исходы остаются в очереди, без обучения задним числом.",
        "",
        "Выбор действия использует тот же prior по доступному, возможно устаревшему отчёту, scoring и exploration=0.05. "
        "Истинный текущий профиль и будущие состояния не передаются планировщику. При exact новый алгоритм "
        "совпадает с исходным на одинаковой последовательности исходов; физический шум времени между отдельными запусками остаётся.",
        "",
        "## Независимая проверка",
        "",
        f"Трассы seed={c['seed']}+r, модели {c['model_seeds']}, learning rate={c['learning_rate']}, 12 участников, "
        "50260 параметров. Модели и шаг взяты из предыдущей проверки; независимыми являются новые трассы, "
        "отказы, шум датчика, RNG exploration и порядок запусков. Это не проверка новых начальных весов. "
        "В каждом блоке перемешаны exact/stale_noise × три планировщика и один фактический F+P: 7 случаев. "
        "Общий фиксированный контроль не дублируется в сырых данных.",
        "",
        "Профили edge_limited/verifier_limited меняются каждые 4–8 попыток; предел общего потока 40 MB/s, "
        "задержка сообщения 2 ms. stale_noise: лаг 2 попытки, lognormal σ=0.35. Датчик синтетический; "
        "нулевой флаг неограниченного потока сохраняется. Возраст известен из метки захвата; реальные сетевые пробы не выполняются.",
        "",
        f"![Время до качества]({output}/alignment_time_to_quality.png)",
        "",
        "## Парные разности aligned−сравнение",
        "",
        "| Условие | Сравнение | Полные пары | Средняя разность, s | Bootstrap 95%, s |",
        "|---|---|---:|---:|---:|",
    ]
    for p in paired:
        lines.append(
            f"| {p['condition']} | {p['second_policy']} | {p['complete_pairs']}/{p['pairs']} | "
            f"{fmt(p['mean_delta_wall_s'])} | [{fmt(p['bootstrap95_low_s'])}; {fmt(p['bootstrap95_high_s'])}] |"
        )
    lines += [
        "",
        "Отрицательная разность означает выигрыш aligned. Единственный заранее заданный первичный "
        "контраст — aligned−original в stale_noise. Остальные интервалы описательные, без поправки на множественные "
        "сравнения. Bootstrap среднего: 20000 выборок, seed 174026; расчёт мощности не выполнялся. "
        "Недостигнутый порог не заменяется временем остановки; полные пары при цензурировании могут быть смещены.",
        "",
        "## Аудит и ограничения",
        "",
        "Проверены сохранённые SHA256 исходников, воспроизведение физических трасс и причинных отчётов, "
        "полный состав блоков, пары моделей, каждое решение и статистика learner. Дополнительно независимо пересчитана "
        "сумма residual по исходному действию и времени именно захваченного раунда. Проверено отсутствие повторного "
        "обучения на одном отчёте и обратной связи от невыбранных действий.",
        "",
        "Поправка согласует стоимость и контекст для обучения, но не восстанавливает текущую полосу из старого "
        "отчёта. Она предполагает корректную метку захвата. Очередь и API ограничены возрастом 20 попыток; "
        "задержка 2 проверена экспериментально. Потери отчётов, WAN и несколько физических машин не проверялись. "
        "Результаты старого отрицательного эксперимента сохранены отдельно и не объединены с новыми измерениями.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        ".venv/bin/python -m unittest discover -s tests -v",
        ".venv/bin/python telemetry_robustness_experiment.py --study alignment --modes exact stale_noise --config configs/tcp_telemetry_alignment.json",
        ".venv/bin/python telemetry_alignment_analyze.py RESULTS_FOLDER --output NEW_REPORT_FOLDER",
        "```",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    validation = dict(
        status="passed",
        runs=len(summary),
        rounds=len(rows),
        reached=sum(int(s["reached"]) for s in summary),
        accepted=sum(int(r["accepted"]) for r in rows),
        timeouts=sum(int(r["timeouts"]) for r in rows),
        primary=primary,
        source_snapshot_sha256=hashlib.sha256(
            (folder / "source_snapshot.zip").read_bytes()
        ).hexdigest(),
    )
    (output / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation), flush=True)
    return output


if __name__ == "__main__":
    main()
