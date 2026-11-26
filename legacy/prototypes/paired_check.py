
import numpy as np
import context_experiment as exp

RUNS = 16
BOOTSTRAPS = 10000

CHANNEL = "Обучение: только канал"
CONTEXT = "Обучение: канал + нагрузка"
THRESHOLD = "Порог 400 КБ/с"

rng = np.random.default_rng(2026)

records = []

for i in range(RUNS):

    result, _ = exp.one_run(15000 + i)

    row = {}

    for policy in (CHANNEL, CONTEXT, THRESHOLD):
        metrics = result[policy]

        row[policy] = {
            "success":
                metrics["on_time"] / exp.TEST_ROUNDS,
            "utility":
                metrics["utility"] / exp.TEST_ROUNDS,
            "switches":
                metrics["switches"],
        }

    records.append(row)

    print(
        f"Готово {i + 1}/{RUNS}",
        flush=True,
    )


def paired_comparison(first, second, metric, scale=1):

    differences = np.array([
        (
            row[first][metric]
            - row[second][metric]
        ) * scale
        for row in records
    ])

    sample_indices = rng.integers(
        0,
        RUNS,
        size=(BOOTSTRAPS, RUNS),
    )

    bootstrap_means = differences[
        sample_indices
    ].mean(axis=1)

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    print(
        f"Средняя разница: "
        f"{differences.mean():+.3f}"
    )

    print(
        f"95% bootstrap-интервал: "
        f"[{lower:+.3f}, {upper:+.3f}]"
    )

    print(
        "Интервал включает ноль: ",
        lower <= 0 <= upper,
    )


print(
    "\nКанал + нагрузка МИНУС только канал"
)
print("Разница успешности, процентные пункты:")
paired_comparison(
    CONTEXT, CHANNEL, "success", 100
)

print("\nРазница числа переключений:")
paired_comparison(
    CONTEXT, CHANNEL, "switches"
)

print("\nРазница полезности:")
paired_comparison(
    CONTEXT, CHANNEL, "utility"
)

print(
    "\nКанал + нагрузка МИНУС порог 400 КБ/с"
)
print("Разница успешности, процентные пункты:")
paired_comparison(
    CONTEXT, THRESHOLD, "success", 100
)

print("\nРазница полезности:")
paired_comparison(
    CONTEXT, THRESHOLD, "utility"
)
