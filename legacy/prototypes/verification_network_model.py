
# Модель стоимости двух способов проверки.
# Все CPU-измерения взяты из наших
# предыдущих запусков на MacBook.
#
# Это расчётная модель сети,
# а НЕ измерение реальной передачи.

CASES = {
    (4, 200_000): {
        "baseline_sign": 4.745,
        "baseline_verify": 5.922,
        "projection_prepare": 22.087,
        "projection_begin": 1.550,
        "projection_verify": 6.112,
    },
    (16, 200_000): {
        "baseline_sign": 19.797,
        "baseline_verify": 22.413,
        "projection_prepare": 99.808,
        "projection_begin": 1.703,
        "projection_verify": 8.256,
    },
}

BANDWIDTHS = (
    500_000,
    5_000_000,
    50_000_000,
)

EXTRA_RTT_MS = (
    20,
    100,
)

CHECKS = 2


def estimate(
    workers,
    dimension,
    measurements,
    bandwidth,
    extra_rtt,
):
    vector_bytes = dimension * 8

    # В обоих вариантах проверяющий
    # получает агрегированную сумму.
    #
    # При полном пересчёте он
    # дополнительно получает
    # все исходные векторы.

    baseline_bytes = (
        vector_bytes
        + workers * vector_bytes
        + workers * 64
    )

    # При линейной проверке:
    # сумма, подписи, проекции
    # и короткие seed для вызова.
    projection_bytes = (
        vector_bytes
        + workers * (64 + 4 * CHECKS)
        + workers * 32 * CHECKS
    )

    baseline_network_ms = (
        1000 * baseline_bytes / bandwidth
    )

    projection_network_ms = (
        1000 * projection_bytes / bandwidth
    )

    # Оптимистическое предположение:
    # участники выполняют независимые
    # операции параллельно.
    baseline_worker_ms = (
        measurements["baseline_sign"]
        / workers
    )

    projection_worker_ms = (
        measurements["projection_prepare"]
        / workers
    )

    baseline_cpu_ms = (
        baseline_worker_ms
        + measurements["baseline_verify"]
    )

    projection_cpu_ms = (
        projection_worker_ms
        + measurements["projection_begin"]
        + measurements["projection_verify"]
    )

    baseline_total = (
        baseline_network_ms
        + baseline_cpu_ms
    )

    projection_total = (
        projection_network_ms
        + projection_cpu_ms
        + extra_rtt
    )

    # Дополнительная оценка при
    # последовательной обработке
    # ответов участников.
    projection_serial = (
        projection_network_ms
        + measurements["projection_prepare"]
        + measurements["projection_begin"]
        + measurements["projection_verify"]
        + extra_rtt
    )

    return {
        "baseline_bytes": baseline_bytes,
        "projection_bytes": projection_bytes,
        "baseline_ms": baseline_total,
        "projection_parallel_ms": projection_total,
        "projection_serial_ms": projection_serial,
    }


def main():
    print(
        "Модель задержки криптографической "
        "проверки с учётом сети"
    )

    print(
        "Проекции: 2; "
        "CPU по прошлым измерениям"
    )

    print(
        "Общий трафик передачи обновлений "
        "к агрегатору не учитывается: "
        "он одинаков для двух схем.\n"
    )

    for (workers, dimension), values in CASES.items():

        print("=" * 64)
        print(
            f"Участников: {workers}, "
            f"размер обновления: {dimension:,}"
        )

        for bandwidth in BANDWIDTHS:
            for rtt in EXTRA_RTT_MS:

                result = estimate(
                    workers,
                    dimension,
                    values,
                    bandwidth,
                    rtt,
                )

                baseline = result["baseline_ms"]
                projection = result[
                    "projection_parallel_ms"
                ]

                winner = (
                    "ПРОЕКЦИИ"
                    if projection < baseline
                    else "ПОЛНЫЙ ПЕРЕСЧЁТ"
                )

                print(
                    f"\nКанал: "
                    f"{bandwidth / 1e6:.1f} МБ/с; "
                    f"доп. RTT: {rtt} мс"
                )

                print(
                    f"  Полный пересчёт: "
                    f"{baseline:.2f} мс"
                )

                print(
                    f"  Проекции, параллельно: "
                    f"{projection:.2f} мс"
                )

                print(
                    f"  Проекции, последовательно: "
                    f"{result['projection_serial_ms']:.2f} мс"
                )

                print(
                    f"  Доп. трафик к проверяющему: "
                    f"{result['baseline_bytes'] / 1e6:.2f} "
                    f"против "
                    f"{result['projection_bytes'] / 1e6:.2f} МБ"
                )

                print(
                    f"  Быстрее по параллельной "
                    f"модели: {winner}"
                )

        print()

    print("Расчёт завершён.")


if __name__ == "__main__":
    main()
