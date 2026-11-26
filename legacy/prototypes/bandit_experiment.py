"""Третий эксперимент: выбор стратегии по результатам предыдущих раундов.

Использует main.py. Проверка суммы выполняется доверенным эталоном,
а НЕ криптографическим протоколом.

Контекст планировщика: известная пропускная способность (250/350/500 КБ/с).
Он не наблюдает будущие сбои, тип сценария и результат второго действия.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import main as sim

BANDWIDTHS = (250_000, 350_000, 500_000)
ACTIONS = ("Быстрое", "С резервированием")
TRAIN_ROUNDS = 300
TEST_ROUNDS = 600
REPEATS = 8
BLOCK_LENGTH = 30
SWITCH_SECONDS = 0.03
# Небольшой дополнительный штраф за смену конфигурации в момент выбора.
SWITCH_PENALTY = 0.035

POLICIES = (
    "Всегда быстрое",
    "Всегда резервирование",
    "Порог 400 КБ/с",
    "Обучающийся планировщик",
)


class ThompsonPlanner:
    """Контекстный бандит: по две Beta-оценки для каждого уровня канала."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.good = defaultdict(lambda: 1)
        self.bad = defaultdict(lambda: 1)

    def choose(self, bandwidth, previous_action):
        scores = {}
        for action in ACTIONS:
            sample = self.rng.beta(
                self.good[(bandwidth, action)],
                self.bad[(bandwidth, action)],
            )
            if previous_action is not None and action != previous_action:
                sample -= SWITCH_PENALTY
            scores[action] = sample
        return max(scores, key=scores.get)

    def observe(self, bandwidth, action, success):
        # Только фактический результат выбранного действия!
        key = (bandwidth, action)
        if success:
            self.good[key] += 1
        else:
            self.bad[key] += 1


def choose_fixed(policy, bandwidth):
    if policy == "Всегда быстрое":
        return ACTIONS[0]
    if policy == "Всегда резервирование":
        return ACTIONS[1]
    return ACTIONS[1] if bandwidth >= 400_000 else ACTIONS[0]


def network_stream(rng, rounds):
    """Независимо задаём условия блоками; сценарий скрыт от агента."""
    for offset in range(0, rounds, BLOCK_LENGTH):
        bandwidth = int(rng.choice(BANDWIDTHS))
        scenario = str(rng.choice(sim.SCENARIOS))
        for _ in range(min(BLOCK_LENGTH, rounds - offset)):
            yield bandwidth, scenario


def evaluate_round(updates, context, action, bandwidth, switched):
    sim.SHARED_BANDWIDTH = bandwidth
    result = sim.run_round(updates, context, action)
    latency = result.latency
    if latency is not None and switched:
        latency += SWITCH_SECONDS
    return {
        "success": latency is not None and latency <= sim.DEADLINE,
        "recovered": latency is not None,
        "bytes": result.sent_bytes,
    }


def run_one(seed):
    planner = ThompsonPlanner(seed + 100_000)
    previous = {policy: None for policy in POLICIES}
    test_rows = {policy: [] for policy in POLICIES}
    switches = {policy: 0 for policy in POLICIES}
    learned_choices = defaultdict(lambda: defaultdict(int))

    for stage, count in (("обучение", TRAIN_ROUNDS), ("тест", TEST_ROUNDS)):
        # Новая выборка условий и отказов для теста.
        rng = np.random.default_rng(seed * 100 + (1 if stage == "обучение" else 2))
        for bandwidth, scenario in network_stream(rng, count):
            updates = rng.integers(
                -5, 6, size=(sim.WORKERS, sim.VECTOR_SIZE), dtype=np.int64
            )
            context = sim.make_context(rng, scenario)

            # Выбор сделан ДО вычисления результата раунда.
            selections = {}
            for policy in POLICIES:
                if policy == "Обучающийся планировщик":
                    selections[policy] = planner.choose(bandwidth, previous[policy])
                else:
                    selections[policy] = choose_fixed(policy, bandwidth)

            for policy, action in selections.items():
                switched = previous[policy] is not None and previous[policy] != action
                previous[policy] = action
                outcome = evaluate_round(updates, context, action, bandwidth, switched)

                if policy == "Обучающийся планировщик":
                    planner.observe(bandwidth, action, outcome["success"])

                if stage == "тест":
                    switches[policy] += switched
                    test_rows[policy].append(outcome)
                    if policy == "Обучающийся планировщик":
                        learned_choices[bandwidth][action] += 1

    summary = {}
    for policy in POLICIES:
        rows = test_rows[policy]
        summary[policy] = {
            "ontime": np.mean([r["success"] for r in rows]),
            "recovery": np.mean([r["recovered"] for r in rows]),
            "mb": np.mean([r["bytes"] for r in rows]) / 1_000_000,
            "switches": switches[policy],
        }
    return summary, learned_choices


def main():
    summaries = []
    choices = defaultdict(lambda: defaultdict(int))
    for i in range(REPEATS):
        summary, learned = run_one(2028 + i)
        summaries.append(summary)
        for bandwidth in BANDWIDTHS:
            for action in ACTIONS:
                choices[bandwidth][action] += learned[bandwidth][action]

    print(f"Обучение: {TRAIN_ROUNDS} раундов; тест: {TEST_ROUNDS} раундов")
    print(f"Независимых запусков: {REPEATS}; дедлайн: {sim.DEADLINE} с")
    print("Алгоритм видит только пропускную способность и результат выбранного действия.\n")

    means = []
    for policy in POLICIES:
        p = np.array([summary[policy]["ontime"] for summary in summaries])
        recovery = np.mean([summary[policy]["recovery"] for summary in summaries])
        mb = np.mean([summary[policy]["mb"] for summary in summaries])
        swaps = np.mean([summary[policy]["switches"] for summary in summaries])
        print(f"{policy}: в срок {p.mean():.1%} ± {p.std(ddof=1):.1%} "
              f"(ст. откл. по запускам), восстановлено {recovery:.1%}, "
              f"трафик {mb:.3f} МБ/раунд, переключений {swaps:.1f}")
        means.append(p.mean())

    print("\nВыбор обучающегося планировщика на тесте:")
    for bandwidth in BANDWIDTHS:
        counts = choices[bandwidth]
        total = sum(counts.values())
        reserve = counts[ACTIONS[1]] / total if total else float("nan")
        print(f"  {bandwidth // 1000} КБ/с: "
              f"резервирование в {reserve:.1%} раундов ({total} наблюдений)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(POLICIES, means)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Доля раундов с корректной суммой до срока")
    ax.set_title("Тест: планирование с обучением по обратной связи")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    path = out / "bandit_comparison.png"
    fig.savefig(path, dpi=180)
    print(f"\nГрафик: {path}")
    plt.show()


if __name__ == "__main__":
    main()