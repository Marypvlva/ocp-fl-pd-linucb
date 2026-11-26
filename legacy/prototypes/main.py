
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ========================================
# Параметры эксперимента
# ========================================

WORKERS = 4
NODES = 4
VECTOR_SIZE = 2400
BLOCKS = 12

ROUNDS = 300
DEADLINE = 0.65

# Общая пропускная способность канала
# передачи обновлений, байт/с
SHARED_BANDWIDTH = 500_000

# Узлы 0 и 1 — площадка A (быстрые)
# Узлы 2 и 3 — площадка B (медленные)

SPEED = (
    75_000,
    65_000,
    26_000,
    22_000,
)

RTT = (
    0.004,
    0.005,
    0.010,
    0.011,
)

SCENARIOS = (
    "Стабильная сеть",
    "Перегрузки",
    "Отказ площадки",
)

STRATEGIES = (
    "Быстрое",
    "С резервированием",
)


# ========================================
# Состояние сети
# ========================================

@dataclass
class Context:
    failed: tuple[bool, ...]
    slowdown: tuple[float, ...]
    corrupted: np.ndarray


@dataclass
class Result:
    latency: float | None
    sent_bytes: int
    rejected: int

    @property
    def on_time(self) -> bool:
        return (
            self.latency is not None
            and self.latency <= DEADLINE
        )


def make_context(rng, scenario):
    failed = list(
        rng.random(NODES) < 0.01
    )

    slowdown = [1.0] * NODES

    if scenario == "Перегрузки":
        if rng.random() < 0.60:
            factor = float(
                rng.uniform(9.0, 13.0)
            )
            slowdown[0] = factor
            slowdown[1] = factor

    if scenario == "Отказ площадки":
        if rng.random() < 0.30:
            failed[0] = True
            failed[1] = True

        if rng.random() < 0.05:
            failed[2] = True
            failed[3] = True

    corrupted = (
        rng.random((BLOCKS, NODES))
        < 0.002
    )

    return Context(
        tuple(failed),
        tuple(slowdown),
        corrupted,
    )


# ========================================
# Стратегии размещения
# ========================================

def placements(strategy):

    if strategy == "Быстрое":
        # По одной копии на быстрых узлах
        return [
            [block % 2]
            for block in range(BLOCKS)
        ]

    if strategy == "С резервированием":
        # По одной копии на каждой площадке
        return [
            [block % 2, 2 + block % 2]
            for block in range(BLOCKS)
        ]

    raise ValueError(
        f"Неизвестная стратегия: {strategy}"
    )


# ========================================
# Один раунд
# ========================================

def run_round(updates, context, strategy):

    block_indices = np.array_split(
        np.arange(VECTOR_SIZE),
        BLOCKS,
    )

    # Доверенная эталонная сумма
    truth = updates.sum(
        axis=0,
        dtype=np.int64,
    )

    assigned = placements(strategy)

    tasks = [
        [] for _ in range(NODES)
    ]

    sent_bytes = 0

    for block, nodes in enumerate(assigned):
        for node in nodes:
            tasks[node].append(block)

            sent_bytes += (
                WORKERS
                * len(block_indices[block])
                * 8
            )

    # Упрощённая модель:
    # сначала передача по общему каналу,
    # затем обработка очередей на узлах.
    upload_time = (
        sent_bytes / SHARED_BANDWIDTH
    )

    verified_finishes = [
        [] for _ in range(BLOCKS)
    ]

    rejected = 0

    for node, node_tasks in enumerate(tasks):

        if context.failed[node]:
            continue

        queue_time = 0.0

        for block in node_tasks:

            index = block_indices[block]

            operations = (
                WORKERS * len(index)
            )

            service_time = (
                RTT[node]
                + operations
                / SPEED[node]
                * context.slowdown[node]
            )

            queue_time += service_time

            # Узел вычисляет сумму
            # назначенного фрагмента.
            aggregate = updates[:, index].sum(
                axis=0,
                dtype=np.int64,
            )

            # Моделируем подмену ответа
            if context.corrupted[block, node]:
                aggregate[0] += 1

            # ВАЖНО:
            # Это проверка по доверенному
            # эталону, а НЕ криптография.
            correct = np.array_equal(
                aggregate,
                truth[index],
            )

            if correct:
                verified_finishes[block].append(
                    upload_time + queue_time
                )
            else:
                rejected += 1

    # Для восстановления всей суммы
    # нужен каждый блок.
    if not all(verified_finishes):
        return Result(
            None,
            sent_bytes,
            rejected,
        )

    latency = max(
        min(times)
        for times in verified_finishes
    )

    return Result(
        latency,
        sent_bytes,
        rejected,
    )


# ========================================
# Серия экспериментов
# ========================================

def simulate():

    rng = np.random.default_rng(2026)

    results = {}

    for scenario in SCENARIOS:

        results[scenario] = {
            strategy: []
            for strategy in STRATEGIES
        }

        for _ in range(ROUNDS):

            updates = rng.integers(
                -5,
                6,
                size=(WORKERS, VECTOR_SIZE),
                dtype=np.int64,
            )

            context = make_context(
                rng,
                scenario,
            )

            # Обе стратегии получают
            # одинаковые данные и сбои.
            for strategy in STRATEGIES:

                result = run_round(
                    updates,
                    context,
                    strategy,
                )

                results[scenario][strategy].append(
                    result
                )

    return results


# ========================================
# Таблица результатов
# ========================================

def report(results):

    print(
        f"Раундов на сценарий: {ROUNDS}"
    )
    print(
        f"Срок завершения: {DEADLINE} с"
    )
    print()

    for scenario in SCENARIOS:

        print(f"\n{scenario}")

        for strategy in STRATEGIES:

            records = results[scenario][strategy]

            on_time = np.mean([
                r.on_time
                for r in records
            ])

            latencies = [
                r.latency
                for r in records
                if r.latency is not None
            ]

            mean_latency = (
                np.mean(latencies)
                if latencies
                else float("nan")
            )

            sent_mb = np.mean([
                r.sent_bytes
                for r in records
            ]) / 1_000_000

            rejected = sum(
                r.rejected
                for r in records
            )

            print(
                f"  {strategy}: "
                f"в срок {on_time:.1%}, "
                f"время {mean_latency:.3f} с, "
                f"передано {sent_mb:.3f} МБ, "
                f"отброшено {rejected}"
            )

    print()
    print(
        "Среднее время рассчитано только "
        "для раундов с полученной суммой."
    )


# ========================================
# График
# ========================================

def plot(results):

    x = np.arange(len(SCENARIOS))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for i, strategy in enumerate(STRATEGIES):

        success = [
            np.mean([
                r.on_time
                for r in results[scenario][strategy]
            ])
            for scenario in SCENARIOS
        ]

        ax.bar(
            x + (i - 0.5) * width,
            success,
            width,
            label=strategy,
        )

    ax.set_xticks(
        x,
        SCENARIOS,
    )

    ax.set_ylim(0, 1.05)

    ax.set_ylabel(
        "Доля раундов, завершённых в срок"
    )

    ax.set_title(
        "Сравнение стратегий размещения"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    output_dir = (
        Path(__file__).resolve().parent
        / "results"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_file = (
        output_dir / "deadline_success.png"
    )

    fig.savefig(
        output_file,
        dpi=180,
    )

    print(
        f"\nГрафик сохранён: {output_file}"
    )

    plt.show()


# ========================================
# Запуск
# ========================================


if __name__ == "__main__":

    for bandwidth in (
        500_000,
        350_000,
        250_000,
    ):
        SHARED_BANDWIDTH = bandwidth

        print(
            f"\n{'=' * 50}"
        )
        print(
            f"Пропускная способность: "
            f"{bandwidth / 1000:.0f} КБ/с"
        )
        print(
            f"{'=' * 50}"
        )

        results = simulate()
        report(results)
