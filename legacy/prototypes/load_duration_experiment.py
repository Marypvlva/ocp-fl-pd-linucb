
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import load_bandit_experiment as base


BLOCK_LENGTHS = (3, 20, 60)
SENSOR_ERROR = 0.20

TRAIN_ROUNDS = 400
TEST_ROUNDS = 600
REPEATS = 8
BOOTSTRAPS = 3000

POLICIES = (
    "Только канал",
    "Окно 1",
    "Окно 3",
    "Окно 5",
)


def one_run(seed, block_length):

    # Длительность состояния сети.
    base.BLOCK_LENGTH = block_length

    planners = {
        "Только канал":
            base.OnlinePlanner(False, seed + 100),

        "Окно 1":
            base.OnlinePlanner(True, seed + 101),

        "Окно 3":
            base.OnlinePlanner(True, seed + 102),

        "Окно 5":
            base.OnlinePlanner(True, seed + 103),
    }

    previous = {
        name: None
        for name in POLICIES
    }

    stats = {
        name: {
            "utility": 0.0,
            "success": 0,
            "switches": 0,
        }
        for name in POLICIES
    }

    stages = (
        ("train", TRAIN_ROUNDS, 1),
        ("test", TEST_ROUNDS, 2),
    )

    for stage, rounds, suffix in stages:

        if stage == "test":
            previous = {
                name: None
                for name in POLICIES
            }

        histories = {
            width: deque(maxlen=width)
            for width in (1, 3, 5)
        }

        last_bandwidth = None

        for (
            bandwidth,
            noisy_load,
            outcomes,
        ) in base.generate_rounds(
            seed * 100 + suffix,
            rounds,
            SENSOR_ERROR,
        ):

            # Сбрасываем историю при
            # наблюдаемой смене канала.
            # Момент изменения реальной
            # нагрузки алгоритму неизвестен.
            if bandwidth != last_bandwidth:
                for history in histories.values():
                    history.clear()

                last_bandwidth = bandwidth

            for history in histories.values():
                history.append(
                    int(noisy_load)
                )

            for name, planner in planners.items():

                if name == "Только канал":
                    observed_load = False

                else:
                    width = int(
                        name.split()[-1]
                    )

                    history = histories[width]
                    positive = sum(history)

                    if (
                        2 * positive
                        == len(history)
                    ):
                        observed_load = bool(
                            noisy_load
                        )
                    else:
                        observed_load = (
                            2 * positive
                            > len(history)
                        )

                action = planner.choose(
                    bandwidth,
                    observed_load,
                    previous[name],
                    training=(stage == "train"),
                )

                changed = (
                    previous[name] is not None
                    and previous[name] != action
                )

                previous[name] = action

                result = outcomes[action]

                latency = result.latency

                if (
                    latency is not None
                    and changed
                ):
                    latency += base.SWITCH_SECONDS

                success = (
                    latency is not None
                    and latency <= base.DEADLINE
                )

                # Только результат
                # выбранного действия.
                planner.observe(
                    bandwidth,
                    observed_load,
                    action,
                    success,
                )

                if stage != "test":
                    continue

                record = stats[name]

                record["success"] += int(success)
                record["switches"] += int(changed)

                record["utility"] += (
                    int(success)
                    - base.TRAFFIC_PRICE
                    * (action == base.RESERVE)
                    - base.SWITCH_PRICE
                    * int(changed)
                )

    for record in stats.values():
        record["success"] /= TEST_ROUNDS
        record["utility"] /= TEST_ROUNDS

    return stats


def paired_interval(differences, seed):

    values = np.asarray(
        differences,
        dtype=float,
    )

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAPS, len(values)),
    )

    samples = values[
        indices
    ].mean(axis=1)

    low, high = np.quantile(
        samples,
        [0.025, 0.975],
    )

    return (
        float(values.mean()),
        float(low),
        float(high),
    )


def main():

    print(
        "Устойчивость сглаживания "
        "при смене нагрузки",
        flush=True,
    )

    print(
        f"Ошибка датчика: "
        f"{SENSOR_ERROR:.0%}"
    )

    print(
        f"Обучение: {TRAIN_ROUNDS}; "
        f"тест: {TEST_ROUNDS}; "
        f"запусков: {REPEATS}\n"
    )

    graph = {
        name: []
        for name in POLICIES
    }

    for block_length in BLOCK_LENGTHS:

        print(
            f"\n{'=' * 60}"
        )

        print(
            "Длительность состояния: "
            f"{block_length} раундов",
            flush=True,
        )

        runs = []

        for i in range(REPEATS):

            runs.append(
                one_run(
                    12000 + i,
                    block_length,
                )
            )

            print(
                f"Готово {i + 1}/{REPEATS}",
                flush=True,
            )

        print("\nРезультаты:\n")

        for name in POLICIES:

            success = np.mean([
                r[name]["success"]
                for r in runs
            ])

            utility = np.mean([
                r[name]["utility"]
                for r in runs
            ])

            switches = np.mean([
                r[name]["switches"]
                for r in runs
            ])

            graph[name].append(
                utility
            )

            print(
                f"{name}: "
                f"в срок {success:.1%}, "
                f"полезность {utility:.4f}, "
                f"переключений {switches:.1f}"
            )

        print(
            "\nПарные сравнения "
            "полезности:"
        )

        for comparator in (
            "Окно 1",
            "Только канал",
        ):

            differences = [
                r["Окно 5"]["utility"]
                - r[comparator]["utility"]
                for r in runs
            ]

            mean, low, high = paired_interval(
                differences,
                seed=(
                    500
                    + block_length
                    + len(comparator)
                ),
            )

            print(
                f"Окно 5 минус {comparator}: "
                f"{mean:+.4f}; "
                f"95% интервал "
                f"[{low:+.4f}, {high:+.4f}]"
            )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for name in POLICIES:

        ax.plot(
            BLOCK_LENGTHS,
            graph[name],
            marker="o",
            label=name,
        )

    ax.set_xticks(
        BLOCK_LENGTHS
    )

    ax.set_xlabel(
        "Длительность состояния сети, раундов"
    )

    ax.set_ylabel(
        "Средняя полезность"
    )

    ax.set_title(
        "Влияние частоты изменений нагрузки"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()
    fig.tight_layout()

    folder = (
        Path(__file__).resolve().parent
        / "results"
    )

    folder.mkdir(
        exist_ok=True
    )

    output = (
        folder
        / "load_duration_comparison.png"
    )

    fig.savefig(
        output,
        dpi=170,
    )

    print(
        f"\nГрафик: {output}",
        flush=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
