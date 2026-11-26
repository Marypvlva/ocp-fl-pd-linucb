
import csv
from pathlib import Path

import numpy as np


INPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "joint_planning.csv"
)

BOOTSTRAPS = 10000

BEST_FIXED = {
    4: "R+B",
    12: "R+P",
    16: "F+P",
}


def bootstrap_interval(values, rng):
    values = np.asarray(
        values,
        dtype=float,
    )

    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAPS, len(values)),
    )

    means = values[
        indices
    ].mean(axis=1)

    low, high = np.percentile(
        means,
        [2.5, 97.5],
    )

    return (
        float(values.mean()),
        float(low),
        float(high),
    )


def compare(
    records,
    workers,
    first,
    second,
    rng,
):
    runs = sorted({
        run
        for count, run, policy in records
        if count == workers
    })

    time_diff = []
    traffic_diff = []

    for run in runs:
        a = records[
            (workers, run, first)
        ]

        b = records[
            (workers, run, second)
        ]

        if not (
            a["reached"]
            and b["reached"]
        ):
            print(
                "Есть недостигнутые пороги. "
                "Сравнение времени требует "
                "учёта цензурирования."
            )
            return

        time_diff.append(
            a["time_s"] - b["time_s"]
        )

        traffic_diff.append(
            a["bytes_mb"] - b["bytes_mb"]
        )

    mean, low, high = bootstrap_interval(
        time_diff,
        rng,
    )

    traffic_mean, _, _ = (
        bootstrap_interval(
            traffic_diff,
            rng,
        )
    )

    print(
        f"\n{first} МИНУС {second}"
    )

    print(
        f"  Парных запусков: "
        f"{len(time_diff)}"
    )

    print(
        f"  Средняя разница времени: "
        f"{mean:+.3f} с"
    )

    print(
        "  95% bootstrap-интервал: "
        f"[{low:+.3f}, {high:+.3f}] с"
    )

    print(
        "  Первая стратегия быстрее: "
        f"{sum(x < 0 for x in time_diff)}"
        f"/{len(time_diff)}"
    )

    print(
        f"  Средняя разница трафика: "
        f"{traffic_mean:+.2f} МБ"
    )


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Не найден файл: {INPUT}"
        )

    records = {}

    with INPUT.open(
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            key = (
                int(row["workers"]),
                int(row["run"]),
                row["policy"],
            )

            records[key] = {
                "reached": (
                    row["reached"].lower()
                    == "true"
                ),
                "time_s": (
                    float(row["time_s"])
                    if row["time_s"]
                    not in ("", "None")
                    else None
                ),
                "bytes_mb": float(
                    row["bytes_mb"]
                ),
            }

    rng = np.random.default_rng(
        2026
    )

    print(
        "Парный статистический анализ "
        "совместного планирования"
    )

    print(
        "Отрицательная разница времени "
        "означает преимущество "
        "первой стратегии."
    )

    print(
        "Интервалы предварительные: "
        "всего восемь запусков "
        "на конфигурацию."
    )

    for workers in (
        4,
        12,
        16,
    ):

        print(
            "\n" + "=" * 55
        )

        print(
            f"Участников: {workers}"
        )

        compare(
            records,
            workers,
            "Онлайн",
            "Правило",
            rng,
        )

        compare(
            records,
            workers,
            "Онлайн",
            BEST_FIXED[workers],
            rng,
        )

        compare(
            records,
            workers,
            "Правило",
            BEST_FIXED[workers],
            rng,
        )

    print(
        "\nАнализ завершён."
    )


if __name__ == "__main__":
    main()
