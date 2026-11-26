"""Первый онлайн-эксперимент: правило выбора по наблюдаемой полосе.

Использует main.py из первого стенда. Это диагностическая симуляция,
а не криптографический протокол или модель обучения нейросети.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import main as sim

RNG_SEED = 2027
EPISODE_ROUNDS = 50
SWITCH_SECONDS = 0.03
THRESHOLD = 400_000  # байт/с, вручную выбранный порог

# Пропускная способность и тип отказов меняются между эпизодами.
EPISODES = [
    (500_000, "Стабильная сеть"),
    (250_000, "Стабильная сеть"),
    (500_000, "Перегрузки"),
    (350_000, "Перегрузки"),
    (500_000, "Отказ площадки"),
    (250_000, "Отказ площадки"),
    (350_000, "Стабильная сеть"),
    (250_000, "Перегрузки"),
    (500_000, "Перегрузки"),
]

POLICIES = ("Всегда быстрое", "Всегда резервирование", "По состоянию сети")


def choose(policy: str, bandwidth: int) -> str:
    # Алгоритму доступна только текущая полоса, но не будущие сбои.
    if policy == "Всегда быстрое":
        return "Быстрое"
    if policy == "Всегда резервирование":
        return "С резервированием"
    return "С резервированием" if bandwidth >= THRESHOLD else "Быстрое"


def experiment():
    rng = np.random.default_rng(RNG_SEED)
    outcomes = {policy: [] for policy in POLICIES}
    previous = {policy: None for policy in POLICIES}
    switches = {policy: 0 for policy in POLICIES}

    for bandwidth, scenario in EPISODES:
        sim.SHARED_BANDWIDTH = bandwidth
        for _ in range(EPISODE_ROUNDS):
            updates = rng.integers(
                -5, 6, size=(sim.WORKERS, sim.VECTOR_SIZE), dtype=np.int64
            )
            context = sim.make_context(rng, scenario)
            for policy in POLICIES:
                strategy = choose(policy, bandwidth)
                switched = previous[policy] is not None and previous[policy] != strategy
                switches[policy] += int(switched)
                previous[policy] = strategy

                result = sim.run_round(updates, context, strategy)
                # Переход добавляется к задержке, но только если сумма получена.
                latency = None if result.latency is None else (
                        result.latency + (SWITCH_SECONDS if switched else 0)
                )
                outcomes[policy].append((
                    latency, result.sent_bytes, strategy, bandwidth, scenario
                ))
    return outcomes, switches


def report(outcomes, switches):
    total = len(next(iter(outcomes.values())))
    print(f"Всего раундов: {total}; срок: {sim.DEADLINE} с")
    print(f"Стоимость смены стратегии: {SWITCH_SECONDS} с")
    print(f"Порог выбора резервирования: {THRESHOLD // 1000} КБ/с\n")
    for policy in POLICIES:
        rows = outcomes[policy]
        within = [v for v in rows if v[0] is not None and v[0] <= sim.DEADLINE]
        completed = [v[0] for v in rows if v[0] is not None]
        avg_mb = sum(v[1] for v in rows) / total / 1_000_000
        print(f"{policy}: в срок {len(within) / total:.1%}, "
              f"восстановлено {len(completed) / total:.1%}, "
              f"трафик {avg_mb:.3f} МБ/раунд, "
              f"переключений {switches[policy]}")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    labels = list(POLICIES)
    success = [sum(1 for v in outcomes[p] if v[0] is not None and v[0] <= sim.DEADLINE)
               / total for p in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, success)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Доля раундов, завершённых в срок")
    ax.set_title("Размещение при изменяющейся пропускной способности")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = out_dir / "online_comparison.png"
    fig.savefig(path, dpi=160)
    print(f"\nГрафик: {path}")
    plt.show()


if __name__ == "__main__":
    report(*experiment())