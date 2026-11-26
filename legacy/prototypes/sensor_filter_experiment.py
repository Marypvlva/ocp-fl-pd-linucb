
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import load_bandit_experiment as base


SEEDS = 12
NOISE_LEVELS = (0.10, 0.20)
FILTER_LENGTH = 5
BOOTSTRAPS = 5000

FAST = base.FAST
RESERVE = base.RESERVE

POLICIES = (
    "Всегда резервирование",
    "Порог 400 КБ/с",
    "Только канал",
    "Сырой датчик",
    "Сглаженный датчик",
)


def one_run(seed, error_rate):

    learners = {
        "Только канал":
            base.OnlinePlanner(False, seed + 17),

        "Сырой датчик":
            base.OnlinePlanner(True, seed + 19),

        "Сглаженный датчик":
            base.OnlinePlanner(True, seed + 23),
    }

    previous = {
        name: None for name in POLICIES
    }

    results = {
        name: {
            "success": 0,
            "recovered": 0,
            "bytes": 0,
            "switches": 0,
            "utility": 0.0,
        }
        for name in POLICIES
    }

    history = deque(maxlen=FILTER_LENGTH)
    previous_bandwidth = None

    frequencies = defaultdict(
        lambda: [0, 0]
    )

    stages = (
        ("train", base.TRAIN, 11),
        ("test", base.TEST, 12),
    )

    for stage, count, offset in stages:

        if stage == "test":

            previous = {
                name: None
                for name in POLICIES
            }

            history.clear()
            previous_bandwidth = None

        for (
            bandwidth,
            noisy_high,
            outcomes,
        ) in base.generate_rounds(
            seed * 100 + offset,
            count,
            error_rate,
        ):

            # При изменении скорости канала
            # сбрасываем старые показания.
            if bandwidth != previous_bandwidth:
                history.clear()
                previous_bandwidth = bandwidth

            # Используем только текущие
            # и предыдущие измерения.
            history.append(int(noisy_high))

            positive = sum(history)

            if 2 * positive == len(history):
                filtered_high = bool(noisy_high)
            else:
                filtered_high = (
                    2 * positive > len(history)
                )

            for name in POLICIES:

                if name == "Всегда резервирование":
                    action = RESERVE

                elif name == "Порог 400 КБ/с":
                    action = (
                        RESERVE
                        if bandwidth >= 400_000
                        else FAST
                    )

                else:

                    observation = (
                        filtered_high
                        if name == "Сглаженный датчик"
                        else noisy_high
                    )

                    action = learners[name].choose(
                        bandwidth,
                        observation,
                        previous[name],
                        stage == "train",
                    )

                changed = (
                    previous[name] is not None
                    and action != previous[name]
                )

                previous[name] = action

                outcome = outcomes[action]

                latency = (
                    outcome.latency
                    + base.SWITCH_SECONDS * changed
                    if outcome.latency is not None
                    else None
                )

                success = (
                    latency is not None
                    and latency <= base.DEADLINE
                )

                # Обучаемся только
                # по выбранному действию.
                if name in learners:

                    learners[name].observe(
                        bandwidth,
                        observation,
                        action,
                        success,
                    )

                if stage != "test":
                    continue

                result = results[name]

                result["success"] += int(success)

                result["recovered"] += int(
                    latency is not None
                )

                result["bytes"] += (
                    outcome.sent_bytes
                )

                result["switches"] += int(changed)

                result["utility"] += (
                    int(success)
                    - base.TRAFFIC_PRICE
                    * (action == RESERVE)
                    - base.SWITCH_PRICE * changed
                )

                if name in (
                    "Сырой датчик",
                    "Сглаженный датчик",
                ):

                    key = (
                        name,
                        bandwidth,
                        observation,
                    )

                    frequencies[key][0] += int(
                        action == RESERVE
                    )

                    frequencies[key][1] += 1

    return results, frequencies


def bootstrap_interval(values, seed):

    values = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(seed)

    samples = rng.integers(
        0,
        len(values),
        size=(
            BOOTSTRAPS,
            len(values),
        ),
    )

    means = values[samples].mean(
        axis=1
    )

    low, high = np.quantile(
        means,
        [0.025, 0.975],
    )

    return (
        float(values.mean()),
        float(low),
        float(high),
    )


def main():

    print(
        "Проверка сглаживания "
        "шумного датчика нагрузки",
        flush=True,
    )

    print(
        f"Обучение: {base.TRAIN}; "
        f"тест: {base.TEST}; "
        f"запусков: {SEEDS}"
    )

    print(
        f"Дедлайн: {base.DEADLINE} с; "
        f"окно сглаживания: "
        f"{FILTER_LENGTH} отсчётов"
    )

    print(
        f"Цена резервирования: "
        f"{base.TRAFFIC_PRICE}; "
        f"цена переключения: "
        f"{base.SWITCH_SECONDS} с + "
        f"{base.SWITCH_PRICE} "
        f"в полезности"
    )

    plot_data = {}

    for noise in NOISE_LEVELS:

        print(
            f"\n=== Ошибка датчика "
            f"{noise:.0%} ===",
            flush=True,
        )

        runs = []

        frequencies = defaultdict(
            lambda: [0, 0]
        )

        for i in range(SEEDS):

            result, counts = one_run(
                6000 + i,
                noise,
            )

            runs.append(result)

            for key, (yes, total) in counts.items():
                frequencies[key][0] += yes
                frequencies[key][1] += total

            print(
                f"Готов запуск {i + 1}/{SEEDS}",
                flush=True,
            )

        print("\nРезультаты теста:\n")

        for name in POLICIES:

            success = np.array([
                r[name]["success"] / base.TEST
                for r in runs
            ])

            utility = np.array([
                r[name]["utility"] / base.TEST
                for r in runs
            ])

            recovered = np.mean([
                r[name]["recovered"] / base.TEST
                for r in runs
            ])

            traffic = np.mean([
                r[name]["bytes"]
                / base.TEST / 1e6
                for r in runs
            ])

            switches = np.mean([
                r[name]["switches"]
                for r in runs
            ])

            plot_data[noise, name] = float(
                utility.mean()
            )

            print(
                f"{name}:\n"
                f"  В срок: {success.mean():.1%}\n"
                f"  Полезность: "
                f"{utility.mean():.4f}\n"
                f"  Восстановлено: "
                f"{recovered:.1%}\n"
                f"  Трафик: "
                f"{traffic:.3f} МБ/раунд\n"
                f"  Переключений: "
                f"{switches:.1f}\n"
            )

        print("Парные сравнения полезности:")

        for other in (
            "Сырой датчик",
            "Только канал",
        ):

            differences = [
                (
                    r["Сглаженный датчик"]["utility"]
                    - r[other]["utility"]
                ) / base.TEST
                for r in runs
            ]

            mean, low, high = bootstrap_interval(
                differences,
                seed=(
                    100
                    + int(1000 * noise)
                    + len(other)
                ),
            )

            print(
                f"Сглаженный минус {other}: "
                f"{mean:+.4f}; "
                f"95% интервал "
                f"[{low:+.4f}, {high:+.4f}]"
            )

        print(
            "\nВыбор резервирования "
            "при разной загрузке:"
        )

        for name in (
            "Сырой датчик",
            "Сглаженный датчик",
        ):

            for bandwidth in (
                350_000,
                450_000,
            ):

                descriptions = []

                for state in (
                    False,
                    True,
                ):

                    yes, total = frequencies[
                        name,
                        bandwidth,
                        state,
                    ]

                    label = (
                        "высокая"
                        if state
                        else "низкая"
                    )

                    if total:
                        descriptions.append(
                            f"{label}: "
                            f"{yes / total:.1%} "
                            f"(n={total})"
                        )

                print(
                    f"{name}, "
                    f"{bandwidth // 1000} КБ/с: "
                    + "; ".join(descriptions)
                )

    # ======================================
    # График
    # ======================================

    x = np.arange(len(POLICIES))

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    for index, noise in enumerate(NOISE_LEVELS):

        values = [
            plot_data[noise, name]
            for name in POLICIES
        ]

        ax.bar(
            x + (index - 0.5) * 0.38,
            values,
            0.38,
            label=(
                f"Ошибка датчика "
                f"{noise:.0%}"
            ),
        )

    ax.set_xticks(
        x,
        POLICIES,
        rotation=15,
        ha="right",
    )

    ax.set_ylabel(
        "Средняя полезность на тесте"
    )

    ax.set_title(
        "Сглаживание измерений нагрузки"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.legend()
    fig.tight_layout()

    folder = (
        Path(__file__).resolve().parent
        / "results"
    )

    folder.mkdir(exist_ok=True)

    output = (
        folder
        / "sensor_filter_comparison.png"
    )

    fig.savefig(
        output,
        dpi=160,
    )

    print(
        f"\nГрафик: {output}",
        flush=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
