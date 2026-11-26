
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import training_experiment as exp


RUNS = 24
BOOTSTRAPS = 10000
FIRST_SEED = 10000

THRESHOLD = "Порог 400 КБ/с"
ONLINE = "Онлайн-обучение"
RESERVE = "Всегда резервирование"
FAST = "Всегда быстрое"

rng = np.random.default_rng(2026)


def paired_interval(first, second):
    """
    Положительная разница:
    первая стратегия медленнее второй.
    """
    differences = first - second

    indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAPS, len(differences)),
    )

    means = differences[indices].mean(axis=1)

    low, high = np.percentile(
        means,
        [2.5, 97.5],
    )

    return (
        differences.mean(),
        np.median(differences),
        low,
        high,
    )


def main():
    workers, validation, test = exp.get_data()

    print(
        "Парное сравнение времени до качества",
        flush=True,
    )
    print(f"Порог loss: {exp.TARGET_LOSS}")
    print(f"Новых запусков: {RUNS}\n")

    rows = []

    for i in range(RUNS):
        seed = FIRST_SEED + i

        result = exp.one_run(
            seed,
            workers,
            validation,
            test,
        )

        row = {"seed": seed}

        for policy in exp.POLICIES:
            row[policy] = result[policy]["hit"]

        rows.append(row)

        print(
            f"Готово {i + 1}/{RUNS}",
            flush=True,
        )

    print("\nВремя до порога:\n")

    arrays = {}

    for policy in exp.POLICIES:

        times = np.array([
            (
                row[policy]
                if row[policy] is not None
                else np.nan
            )
            for row in rows
        ])

        arrays[policy] = times

        reached = np.isfinite(times)

        print(
            f"{policy}: "
            f"достигли {reached.sum()}/{RUNS}",
            end="",
        )

        if reached.any():
            print(
                f"; медиана среди достигших "
                f"{np.median(times[reached]):.2f} с"
            )
        else:
            print()

    print("\nПарные сравнения:\n")

    comparisons = (
        (ONLINE, THRESHOLD),
        (ONLINE, RESERVE),
        (ONLINE, FAST),
        (THRESHOLD, RESERVE),
    )

    for first_name, second_name in comparisons:

        first = arrays[first_name]
        second = arrays[second_name]

        # Времена сравниваем только
        # там, где обе стратегии
        # достигли порога.
        valid = (
            np.isfinite(first)
            & np.isfinite(second)
        )

        if not valid.all():
            print(
                f"{first_name} против {second_name}: "
                f"оба достигли порога "
                f"в {valid.sum()}/{RUNS} запусках."
            )
            print(
                "Сравнение времен только среди "
                "достигших может быть смещено. "
                "Для полного анализа потребуется "
                "единый горизонт наблюдения.\n"
            )
            continue

        mean, median, low, high = paired_interval(
            first,
            second,
        )

        faster = np.sum(first < second)
        equal = np.sum(
            np.isclose(first, second)
        )

        print(
            f"{first_name} МИНУС {second_name}"
        )
        print(
            f"  Средняя разница: {mean:+.2f} с"
        )
        print(
            f"  Медиана парной разницы: "
            f"{median:+.2f} с"
        )
        print(
            f"  95% bootstrap-интервал: "
            f"[{low:+.2f}, {high:+.2f}] с"
        )
        print(
            f"  Первая стратегия быстрее "
            f"в {faster}/{RUNS} запусках"
        )
        print(
            f"  Практически равное время: "
            f"{equal}/{RUNS}\n"
        )

    # Сохраняем исходные результаты.
    folder = (
        Path(__file__).resolve().parent
        / "results"
    )
    folder.mkdir(exist_ok=True)

    csv_path = folder / "training_paired.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["seed", *exp.POLICIES],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Таблица сохранена: {csv_path}")

    # Парный график: порог против
    # онлайн-планировщика.
    threshold_times = arrays[THRESHOLD]
    online_times = arrays[ONLINE]

    valid = (
        np.isfinite(threshold_times)
        & np.isfinite(online_times)
    )

    if valid.any():
        x = threshold_times[valid]
        y = online_times[valid]

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        ax.scatter(x, y, alpha=0.75)

        lower = min(x.min(), y.min()) - 2
        upper = max(x.max(), y.max()) + 2

        ax.plot(
            [lower, upper],
            [lower, upper],
            "--",
            label="Одинаковое время",
        )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

        ax.set_xlabel(
            "Пороговое правило: T_eps, с"
        )
        ax.set_ylabel(
            "Онлайн-планировщик: T_eps, с"
        )

        ax.set_title(
            "Парное сравнение времени обучения"
        )

        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()

        output = (
            folder / "training_paired.png"
        )
        fig.savefig(output, dpi=170)

        print(f"График: {output}")

        plt.show()


if __name__ == "__main__":
    main()
