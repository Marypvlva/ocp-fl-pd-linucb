
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tcp_process_experiment as tcp


WORKER_COUNTS = (4, 8, 12, 16)
DIMENSIONS = (50_000, 200_000)

INDEPENDENT_RUNS = 6
TRIALS_PER_RUN = 10
WARMUP = 2
BOOTSTRAPS = 5000

tcp.REPETITIONS = TRIALS_PER_RUN
tcp.WARMUP = WARMUP


def confidence_interval(values, rng):
    values = np.asarray(
        values,
        dtype=float,
    )

    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAPS, len(values)),
    )

    means = values[indices].mean(axis=1)

    low, high = np.percentile(
        means,
        [2.5, 97.5],
    )

    return (
        float(values.mean()),
        float(low),
        float(high),
    )


def main():
    rng = np.random.default_rng(2026)

    configs = [
        (workers, dimension)
        for dimension in DIMENSIONS
        for workers in WORKER_COUNTS
    ]

    observations = defaultdict(list)
    raw_rows = []

    print(
        "TCP: граница выбора механизма проверки",
        flush=True,
    )

    print(
        f"Процессов: {WORKER_COUNTS}"
    )

    print(
        f"Размеры: {DIMENSIONS}"
    )

    print(
        f"Независимых запусков: "
        f"{INDEPENDENT_RUNS}"
    )

    print(
        f"Измерений на запуск: "
        f"{TRIALS_PER_RUN}\n",
        flush=True,
    )

    for repeat in range(INDEPENDENT_RUNS):

        # Перемешиваем порядок конфигураций,
        # чтобы влияние прогрева компьютера
        # не было привязано к одной из них.
        order = rng.permutation(len(configs))

        print(
            f"Серия {repeat + 1}"
            f"/{INDEPENDENT_RUNS}",
            flush=True,
        )

        for index in order:

            workers, dimension = configs[
                int(index)
            ]

            measurements = tcp.run_case(
                workers,
                dimension,
            )

            by_method = {
                row["method"]: row
                for row in measurements
            }

            baseline = by_method[
                "baseline"
            ]["median_ms"]

            projection = by_method[
                "projection"
            ]["median_ms"]

            difference = (
                projection - baseline
            )

            key = (workers, dimension)

            observations[key].append(
                difference
            )

            raw_rows.append({
                "run": repeat + 1,
                "workers": workers,
                "dimension": dimension,
                "baseline_ms": baseline,
                "projection_ms": projection,
                "difference_ms": difference,
                "baseline_mb": by_method[
                    "baseline"
                ]["worker_wire_mb"],
                "projection_mb": by_method[
                    "projection"
                ]["worker_wire_mb"],
            })

            print(
                f"  {workers:2d} участников, "
                f"{dimension:6,d} чисел: "
                f"baseline {baseline:.2f} мс; "
                f"projection {projection:.2f} мс; "
                f"разница {difference:+.2f} мс",
                flush=True,
            )

        print()

    print("\nИтоговые сравнения\n")

    result_rows = []
    interval_rng = np.random.default_rng(77)

    for dimension in DIMENSIONS:

        print(
            f"Размер обновления: "
            f"{dimension:,}"
        )

        for workers in WORKER_COUNTS:

            key = (workers, dimension)

            mean, low, high = (
                confidence_interval(
                    observations[key],
                    interval_rng,
                )
            )

            if high < 0:
                interpretation = (
                    "проекции быстрее"
                )
            elif low > 0:
                interpretation = (
                    "пересчёт быстрее"
                )
            else:
                interpretation = (
                    "различие неустойчиво"
                )

            print(
                f"  {workers:2d} участников: "
                f"{mean:+.2f} мс; "
                f"95% CI "
                f"[{low:+.2f}, {high:+.2f}]; "
                f"{interpretation}"
            )

            result_rows.append({
                "workers": workers,
                "dimension": dimension,
                "mean_difference_ms": mean,
                "ci_low_ms": low,
                "ci_high_ms": high,
            })

        print()

    # =====================================
    # Рисуем разницу задержек
    # =====================================

    fig, ax = plt.subplots(
        figsize=(9, 5),
    )

    for dimension in DIMENSIONS:

        subset = [
            row
            for row in result_rows
            if row["dimension"] == dimension
        ]

        x = np.array([
            row["workers"]
            for row in subset
        ])

        y = np.array([
            row["mean_difference_ms"]
            for row in subset
        ])

        lower = np.array([
            row["ci_low_ms"]
            for row in subset
        ])

        upper = np.array([
            row["ci_high_ms"]
            for row in subset
        ])

        ax.errorbar(
            x,
            y,
            yerr=[
                y - lower,
                upper - y,
            ],
            marker="o",
            capsize=4,
            label=f"{dimension:,} чисел",
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(
        WORKER_COUNTS
    )

    ax.set_xlabel(
        "Число участников"
    )

    ax.set_ylabel(
        "Проекции − полный пересчёт, мс"
    )

    ax.set_title(
        "Граница выбора механизма проверки"
    )

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    folder = (
        Path(__file__).resolve().parent
        / "results"
    )

    folder.mkdir(exist_ok=True)

    figure_path = (
        folder / "tcp_crossover.png"
    )

    raw_path = (
        folder / "tcp_crossover_raw.csv"
    )

    summary_path = (
        folder / "tcp_crossover_summary.csv"
    )

    fig.savefig(
        figure_path,
        dpi=170,
    )

    for path, rows in (
        (raw_path, raw_rows),
        (summary_path, result_rows),
    ):
        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0]),
            )

            writer.writeheader()
            writer.writerows(rows)

    print(f"График: {figure_path}")
    print(f"Измерения: {raw_path}")
    print(f"Сводка: {summary_path}")

    plt.show()


if __name__ == "__main__":
    main()
