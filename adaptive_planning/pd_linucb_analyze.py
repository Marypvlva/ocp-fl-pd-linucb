"""Replay the integrated PD-LinUCB and independently check the article equations."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
import hashlib
import heapq
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
import zipfile

import numpy as np

from .network_telemetry import NetworkTelemetry, TelemetryMode
from .pd_linucb_experiment import POLICIES, make_trace
from .pd_linucb_planner import make_context, outcome_reward
from .tcp_dynamic_experiment import create_pd_planner, observation_for, observed_issues
from .tcp_network import NetworkProfile
from .tcp_runtime import ACTIONS
from .telemetry_alignment_analyze import close_tree
from .telemetry_robustness_analyze import read_csv, write_csv


def validate(folder):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("family") != "pd_linucb_budget" or meta["status"] != "completed":
        raise ValueError("Expected a completed PD-LinUCB integration study")
    config = meta["config"]
    policies = meta["policies"]
    if not policies or len(set(policies)) != len(policies) or not set(policies) <= set(POLICIES):
        raise ValueError("Unknown or duplicate policy")
    expected = {
        (w, r, f"budget{i}", p)
        for w in config["workers"]
        for r in range(config["repeats"])
        for i in range(len(meta["budgets"]))
        for p in policies
    }
    summary = read_csv(folder / "summary.csv")
    rows = read_csv(folder / "rounds.csv") if (folder / "rounds.csv").exists() else []
    key = lambda r: (int(r["workers"]), int(r["repeat"]), r["condition"], r["policy"])
    if (
        len(summary) != meta["planned_runs"]
        or len(summary) != len(expected)
        or {key(s) for s in summary} != expected
    ):
        raise ValueError("Duplicate, missing or unplanned study cases")
    with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt source snapshot")
        for name, digest in meta["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Source checksum mismatch: {name}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    if set(grouped) - expected:
        raise ValueError("Unknown round case")
    traces = {}
    for s in summary:
        workers, repeat, condition, policy = key(s)
        trace_key = workers, repeat
        if trace_key not in traces:
            trace = [asdict(t) for t in make_trace(SimpleNamespace(**config), repeat, workers)]
            stored = json.loads((folder / f"trace_w{workers}_r{repeat}.json").read_text())
            if json.loads(json.dumps(trace)) != stored:
                raise ValueError("Scenario differs from assigned seed")
            traces[trace_key] = stored
        budget = Fraction(str(meta["budgets"][int(condition.removeprefix("budget"))]))
        if Fraction(s["budget_credits"]) != budget:
            raise ValueError("Budget assignment changed")
        local = SimpleNamespace(**{**config, "budget_credits": float(budget)})
        planner = create_pd_planner(policy, local) if policy in ("pd_linucb", "linucb") else None
        sensor = NetworkTelemetry(TelemetryMode(**meta["sensor"]), config["seed"] + 3000 + repeat)
        costs = {a: Fraction(str(config["action_costs"][a])) for a in ACTIONS}
        records = grouped[key(s)]
        previous = None
        issues = (False, False)
        spent = Fraction(0)
        accepted = 0
        reward_sum = 0.0
        dimension = len(meta["context_names"])
        matrices = {a: config["pd_ridge"] * np.eye(dimension) for a in ACTIONS}
        rhs = {a: np.zeros(dimension) for a in ACTIONS}
        label_queue = []
        observed = 0
        dual = 0.0
        for attempt, row in enumerate(records, 1):
            if int(row["attempt"]) != attempt or attempt > config["rounds"]:
                raise ValueError("Noncontiguous or over-horizon selection")
            seed = (
                config["model_seeds"][repeat] if config.get("model_seeds") else config["model_seed"]
            )
            if (
                int(row["model_seed"]) != seed
                or float(row["learning_rate"]) != config["learning_rate"]
            ):
                raise ValueError("Training pair changed")
            step = traces[trace_key][attempt - 1]
            if (
                row["profile"] != step["profile"]
                or json.loads(row["injected_fault"]) != step["fault"]
            ):
                raise ValueError("Physical scenario mismatch")
            report = sensor.observe(
                NetworkProfile(**meta["network_profiles"][step["profile"]]), attempt
            )
            obs = observation_for(report, workers, 75 * config["hidden"] + 10, *issues)
            if not close_tree(json.loads(row["network_observation"]), asdict(obs)):
                raise ValueError("Noncausal observable context")
            if (
                int(row["telemetry_source_attempt"]) != report.capture_attempt
                or int(row["telemetry_age_rounds"]) != attempt - report.capture_attempt
                or row["telemetry_source_profile"] != report.source_profile
                or row["telemetry_mode"] != config["telemetry_mode"]
            ):
                raise ValueError("Sensor age mismatch")
            feasible = tuple(a for a in ACTIONS if costs[a] <= budget - spent)
            if not feasible:
                raise ValueError("Action executed after budget exhaustion")
            action = row["action"]
            if planner:
                while label_queue and label_queue[0][0] <= attempt:
                    _, _, arm, old_context, reward = heapq.heappop(label_queue)
                    rhs[arm] += old_context * reward
                    observed += 1
                x = make_context(obs, previous, attempt - report.capture_attempt)
                # Independent ridge solve: no use of the implementation's inverse.
                means = {a: float(np.linalg.solve(matrices[a], rhs[a]) @ x) for a in ACTIONS}
                radii = {
                    a: config["pd_alpha"]
                    * math.sqrt(max(0.0, float(x @ np.linalg.solve(matrices[a], x))))
                    for a in ACTIONS
                }
                scores = {a: means[a] + radii[a] - dual * float(costs[a]) for a in ACTIONS}
                planner.begin_round(obs, report.capture_attempt, attempt)
                chosen = planner.choose(obs, previous)
                if (
                    chosen != action
                    or action not in feasible
                    or scores[action] < max(scores[a] for a in feasible) - 1e-10
                ):
                    raise ValueError("Selected action violates PD-LinUCB score or budget")
                decision = json.loads(row["decision"])
                if not close_tree(decision, planner.last_decision):
                    raise ValueError("Decision cannot be replayed")
                target = float((budget - spent) / (config["rounds"] - attempt + 1))
                eta = config["pd_eta"] if policy == "pd_linucb" else 0.0
                new_dual = max(0.0, dual + eta * (float(costs[action]) - target))
                if (
                    not math.isclose(decision["dual_before"], dual, abs_tol=1e-12)
                    or not math.isclose(decision["dual_after"], new_dual, abs_tol=1e-12)
                    or not math.isclose(decision["target_cost_rate"], target, abs_tol=1e-12)
                    or not close_tree(decision["predicted_reward"], [means[a] for a in ACTIONS])
                    or not close_tree(decision["confidence_bonus"], [radii[a] for a in ACTIONS])
                ):
                    raise ValueError("Independent equation audit failed")
                matrices[action] += np.outer(x, x)
                dual = new_dual
            elif action != policy:
                raise ValueError("Fixed baseline changed action")
            if action not in feasible:
                raise ValueError("Unfunded execution")
            spent += costs[action]
            if (
                spent > budget
                or float(row["spent_credits"]) != float(spent)
                or float(row["remaining_credits"]) != float(budget - spent)
                or Fraction(row["action_cost_credits"]) != costs[action]
            ):
                raise ValueError("Budget accounting mismatch")
            accepted += int(row["accepted"])
            if int(row["accepted_updates"]) != accepted:
                raise ValueError("Update counter mismatch")
            reward = outcome_reward(
                bool(int(row["accepted"])),
                float(row["cycle_wall_s"]),
                config["pd_reward"],
                config["pd_reward_reference_seconds"],
            )
            if float(row["bandit_reward"]) != reward:
                raise ValueError("Reward mapping mismatch")
            reward_sum += reward
            if planner:
                planner.observe_round(
                    action,
                    bool(int(row["accepted"])),
                    float(row["cycle_wall_s"]),
                    float(row["switch_wall_s"]),
                )
                if config["pd_feedback_delay"] == 0:
                    rhs[action] += x * reward
                    observed += 1
                else:
                    heapq.heappush(
                        label_queue,
                        (attempt + config["pd_feedback_delay"], attempt, action, x.copy(), reward),
                    )
                if int(row["feedback_updates"]) != observed or not close_tree(
                    json.loads(row["planner_statistics"]), planner.stats
                ):
                    raise ValueError("Delayed reward statistics mismatch")
                for i, arm in enumerate(ACTIONS):
                    if not np.allclose(
                        planner.engine.A_inv[i],
                        np.linalg.inv(matrices[arm]),
                        atol=1e-11,
                        rtol=1e-11,
                    ) or not np.allclose(planner.engine.b[i], rhs[arm], atol=1e-12):
                        raise ValueError("Design-now/reward-later matrix mismatch")
            if (
                not int(row["traffic_is_lower_bound"])
                and row["app_tx_bytes"] != row["app_rx_bytes"]
            ):
                raise ValueError("Byte counters disagree")
            previous = action
            issues = observed_issues(SimpleNamespace(errors=json.loads(row["errors"])))
        if (
            len(records) != int(s["attempts"])
            or accepted != int(s["accepted_updates"])
            or float(s["spent_credits"]) != float(spent)
            or not math.isclose(float(s["bandit_reward_sum"]), reward_sum, abs_tol=1e-10)
        ):
            raise ValueError("Summary differs from charged executions")
        if planner and (
            int(s["observed_rewards"]) != observed or int(s["pending_rewards"]) != len(label_queue)
        ):
            raise ValueError("Future labels were flushed at stop")
        hits = [r for r in records if float(r["validation_loss"]) <= config["target_loss"]]
        first = (
            0.0
            if float(s["initial_loss"]) <= config["target_loss"]
            else float(hits[0]["quality_observed_wall_s"])
            if hits
            else None
        )
        if int(s["reached"]) != int(first is not None) or (
            first is not None and float(s["t_epsilon_wall_s"]) != first
        ):
            raise ValueError("Incorrect first-passage quality time")
        remaining_actions = [a for a in ACTIONS if costs[a] <= budget - spent]
        reason = (
            "target_reached"
            if first is not None and config["stop_at_target"]
            else "update_out_of_range"
            if records and int(records[-1]["update_bound_exceeded"])
            else "horizon_exhausted"
            if len(records) == config["rounds"]
            else "budget_exhausted"
            if not remaining_actions
            else "fixed_action_unaffordable"
            if policy in ACTIONS and policy not in remaining_actions
            else None
        )
        if reason is None or s["stop_reason"] != reason:
            raise ValueError("Premature or incorrectly classified stop")
        journal = (
            folder
            / f"journal_{condition}_w{workers}_r{repeat}_{config['telemetry_mode']}_{policy.replace('+', '')}.jsonl"
        )
        recorded = [json.loads(line) for line in journal.read_text().splitlines()]
        if len(recorded) != len(records) or any(
            any((b[k] not in ("", "None") if v is None else str(v) != b[k]) for k, v in a.items())
            for a, b in zip(recorded, records)
        ):
            raise ValueError("JSONL and CSV differ")
    return meta, summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    meta, summary, rows = validate(args.folder)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    rng = np.random.default_rng(203026)
    comparisons = []
    for workers in (
        meta["config"]["workers"] if {"pd_linucb", "linucb"} <= set(meta["policies"]) else []
    ):
        for index, budget in enumerate(meta["budgets"]):
            groups = {
                p: {
                    int(r["repeat"]): r
                    for r in summary
                    if int(r["workers"]) == workers
                    and r["condition"] == f"budget{index}"
                    and r["policy"] == p
                }
                for p in meta["policies"]
            }
            delta = np.array(
                [
                    float(groups["pd_linucb"][r]["bandit_reward_sum"])
                    - float(groups["linucb"][r]["bandit_reward_sum"])
                    for r in groups["pd_linucb"]
                ]
            )
            low = high = None
            if len(delta) >= 2:
                low, high = np.quantile(
                    rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1),
                    [0.025, 0.975],
                ).tolist()
            comparisons.append(
                dict(
                    workers=workers,
                    budget_credits=budget,
                    primary=int(budget == min(meta["budgets"])),
                    pairs=len(delta),
                    reward_delta_PD_minus_LinUCB=float(delta.mean()),
                    bootstrap95_low=low,
                    bootstrap95_high=high,
                )
            )
    write_csv(output / "paired.csv", comparisons)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = meta["config"]["workers"]
    fig, axes = plt.subplots(
        len(counts),
        len(meta["budgets"]),
        figsize=(6 * len(meta["budgets"]), 4.5 * len(counts)),
        squeeze=False,
    )
    for row_axes, workers in zip(axes, counts):
        for ax, (index, budget) in zip(row_axes, enumerate(meta["budgets"])):
            for i, policy in enumerate(meta["policies"]):
                rr = [
                    r
                    for r in summary
                    if int(r["workers"]) == workers
                    and r["condition"] == f"budget{index}"
                    and r["policy"] == policy
                ]
                values = [float(r["bandit_reward_sum"]) for r in rr]
                ax.scatter(
                    i + np.linspace(-0.06, 0.06, len(values)),
                    values,
                    color="#338574" if policy == "pd_linucb" else "#547d9c",
                    s=24,
                )
                ax.plot([i - 0.12, i + 0.12], [mean(values)] * 2, color="#222222")
            ax.set_xticks(
                range(len(meta["policies"])),
                [
                    p.replace("pd_linucb", "PD-LinUCB").replace("linucb", "LinUCB")
                    for p in meta["policies"]
                ],
                rotation=15,
            )
            ax.set(
                title=f"{workers} workers · B={budget} allocation credits",
                ylabel="Total bounded reward",
            )
            ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "pd_linucb_rewards.png", dpi=180)
    plt.close(fig)
    reward_description = (
        "reward=1 за принятое проверенное обновление, иначе 0. "
        if meta["config"]["pd_reward"] == "success"
        else "Используется явная throughput-адаптация: reward=accepted*min(1, reference/cycle). "
    )
    comparison_description = (
        f"Инженерный пилот на {meta['config']['repeats']} заранее выбранных моделях/новых трассах. "
        "Первичный описательный контраст — награда PD−LinUCB при меньшем бюджете. Знак плюс означает выигрыш PD. "
        "Bootstrap 20000 выборок, seed 203026; мощность не рассчитывалась. "
        if comparisons
        else "Это однопланнерный запуск PD-LinUCB, без сравнительного вывода об эффективности. "
    )
    stop_description = (
        "Запуск останавливается при целевом качестве, исчерпании квоты или горизонта. "
        if meta["config"]["stop_at_target"]
        else "Пилот продолжает обучение до горизонта/бюджета даже после первого достижения loss. "
    )
    lines = [
        "# PD-LinUCB: интеграция алгоритма статьи в TCP-планировщик",
        "",
        "Реализованы выбор UCB−dual×cost, динамический темп расхода оставшегося бюджета, "
        "немедленное обновление design через Sherman–Morrison и отложенное обновление reward-вектора. "
        "LinUCB — тот же код с eta=0 и тем же допустимым множеством. Награды не наблюдаются для невыбранных действий.",
        "",
        f"Запусков {len(summary)}, реальных раундов {len(rows)}, принятых обновлений {sum(int(r['accepted']) for r in rows)}. "
        f"Достигли validation loss≤{meta['config']['target_loss']}: {sum(int(s['reached']) for s in summary)}/{len(summary)}. "
        "Недостигнутый порог не заменяется временем остановки.",
        "",
        "## Что перенесено из статьи и что адаптировано",
        "",
        "Математическое ядро и stop-at-budget соответствуют разделу 4 статьи. "
        "Действия — F+B/F+P/R+B/R+P; контекст — 17 нормированных признаков из доступных скоростей, возраста "
        "датчика, предыдущего действия и прошлых ошибок. Стоимости — известные allocation credits "
        f"{meta['config']['action_costs']}, НЕ реальные байты или секунды. Жёсткая гарантия относится только к этим кредитам. "
        "Фактическое время и трафик регистрируются отдельно. Контексты Criteo и линейный симулятор reward не используются.",
        "",
        reward_description
        + "Обновление MLP/проверка синхронны, только передача метки в bandit искусственно задерживается на "
        f"{meta['config']['pd_feedback_delay']} шага. Задержка датчика отдельная. Нулевые и положительные метки "
        "задерживаются одинаково; positive-only censoring из Criteo здесь не воспроизводится. "
        "Операционные ошибки предыдущего раунда остаются доступным контекстом.",
        "",
        "## Контрольное сравнение",
        "",
        comparison_description
        + "Преимущество из исходной статьи на Criteo не переносится автоматически в этот стенд.",
        "",
        "| Бюджет | Пар | PD−LinUCB, reward | Описательный bootstrap 95% |",
        "|---:|---:|---:|---:|",
    ]
    for p in comparisons:
        lines.append(
            f"| {p['budget_credits']} | {p['pairs']} | {p['reward_delta_PD_minus_LinUCB']:+.3f} | [{p['bootstrap95_low']}; {p['bootstrap95_high']}] |"
        )
    lines += [
        "",
        f"![Награда]({output}/pd_linucb_rewards.png)",
        "",
        "## Аудит и ограничения",
        "",
        "Независимо пересчитаны A=regularizer×I+sum(xx^T) и b=sum(xr) только по доставленным меткам; "
        "проверены прямое решение линейной системы, UCB-индексы, двойственный множитель, точная "
        "рациональная сумма известных затрат, отсутствие будущих меток, первые достижения loss, сценарии, "
        "снимки исходников и журналы. Последние pending-награды после остановки не добавляются задним числом.",
        "",
        "Алгоритм статьи максимизирует reward при ресурсе/горизонте. Минимизация физического T_epsilon "
        "является отдельной задачей: бинарная награда её не гарантирует. "
        + stop_description
        + "Fixed-базовый метод имеет единственное действие "
        "и останавливается, когда не может оплатить именно его; остаток не перераспределяется. "
        "Общий предварительный warmup вынесен за границы T и квоты bandit. Один Mac, честные участники; "
        "нет универсальной гарантии качества, реального traffic-budget или WAN.",
        "",
        "## Запуск",
        "",
        "```bash",
        ".venv/bin/python -m adaptive_planning smoke --study pd-linucb",
        ".venv/bin/python -m adaptive_planning study pd-linucb",
        ".venv/bin/python -m adaptive_planning verify RESULTS_FOLDER",
        "```",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    result = dict(
        status="passed",
        runs=len(summary),
        rounds=len(rows),
        accepted=sum(int(r["accepted"]) for r in rows),
        budget_violations=0,
        independent_equation_audit=True,
        chosen_actions_only=True,
        delayed_reward_audit=True,
        reached=sum(int(s["reached"]) for s in summary),
        primary=[p for p in comparisons if p["primary"]],
    )
    (output / "validation.json").write_text(json.dumps(result, indent=2))
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(result), flush=True)
    return output


if __name__ == "__main__":
    main()
