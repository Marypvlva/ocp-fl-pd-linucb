
import numpy as np

import main as sim


BANDWIDTHS = (
    250_000,
    350_000,
    450_000,
    500_000,
)

DEADLINES = (
    0.55,
    0.65,
    0.75,
    0.90,
)

RUNS = 8
ROUNDS_PER_CASE = 220

TRAFFIC_PENALTY = 0.08

FAST = "Быстрое"
RESERVE = "С резервированием"


def create_context(rng, high_load):
    # Отказы независимы от загрузки.
    failed = tuple(
        bool(x)
        for x in (
            rng.random(sim.NODES) < 0.01
        )
    )

    slowdown = [1.0] * sim.NODES

    if high_load:
        factor = float(
            rng.uniform(9.0, 13.0)
        )

        # Перегружена быстрая площадка.
        slowdown[0] = factor
        slowdown[1] = factor

    corrupted = (
        rng.random(
            (sim.BLOCKS, sim.NODES)
        ) < 0.002
    )

    return sim.Context(
        failed=failed,
        slowdown=tuple(slowdown),
        corrupted=corrupted,
    )


def collect_case(seed, bandwidth, high_load):
    rng = np.random.default_rng(seed)

    sim.SHARED_BANDWIDTH = bandwidth

    fast_times = []
    reserve_times = []

    for _ in range(ROUNDS_PER_CASE):

        updates = rng.integers(
            -5,
            6,
            size=(
                sim.WORKERS,
                sim.VECTOR_SIZE,
            ),
            dtype=np.int64,
        )

        context = create_context(
            rng, high_load
        )

        fast = sim.run_round(
            updates,
            context,
            FAST,
        )

        reserve = sim.run_round(
            updates,
            context,
            RESERVE,
        )

        fast_times.append(
            fast.latency
            if fast.latency is not None
            else np.inf
        )

        reserve_times.append(
            reserve.latency
            if reserve.latency is not None
            else np.inf
        )

    return (
        np.asarray(fast_times),
        np.asarray(reserve_times),
    )


def report_case(fast_runs, reserve_runs, deadline):

    fast_success = np.mean([
        np.mean(times <= deadline)
        for times in fast_runs
    ])

    reserve_success = np.mean([
        np.mean(times <= deadline)
        for times in reserve_runs
    ])

    # Упрощённая полезность:
    # вероятность завершения минус
    # фиксированный штраф за резерв.
    delta = (
        reserve_success
        - fast_success
        - TRAFFIC_PENALTY
    )

    decision = (
        RESERVE
        if delta > 0
        else FAST
    )

    return (
        fast_success,
        reserve_success,
        delta,
        decision,
    )


def main():

    print(
        "Чувствительность выбора "
        "к дедлайну",
        flush=True,
    )

    print(
        f"Запусков: {RUNS}; "
        f"раундов на состояние: "
        f"{ROUNDS_PER_CASE}"
    )

    print(
        "Штраф резервирования: "
        f"{TRAFFIC_PENALTY}"
    )

    print(
        "Стоимость переключения "
        "пока не учитывается.\n"
    )

    data = {}

    for bandwidth in BANDWIDTHS:

        for high_load in (False, True):

            fast_runs = []
            reserve_runs = []

            for run in range(RUNS):

                # Отдельный воспроизводимый
                # поток случайных величин.
                seed = (
                    50000
                    + bandwidth
                    + run * 1000
                    + int(high_load) * 100
                )

                fast, reserve = collect_case(
                    seed,
                    bandwidth,
                    high_load,
                )

                fast_runs.append(fast)
                reserve_runs.append(reserve)

            data[
                bandwidth,
                high_load,
            ] = (
                fast_runs,
                reserve_runs,
            )

        print(
            f"Собраны данные: "
            f"{bandwidth // 1000} КБ/с",
            flush=True,
        )

    for deadline in DEADLINES:

        print("\n" + "=" * 65)

        print(
            f"ДЕДЛАЙН: {deadline:.2f} с"
        )

        for bandwidth in BANDWIDTHS:

            decisions = {}

            print(
                f"\nКанал {bandwidth // 1000} КБ/с"
            )

            for high_load in (
                False,
                True,
            ):

                fast_runs, reserve_runs = data[
                    bandwidth,
                    high_load,
                ]

                (
                    fast_rate,
                    reserve_rate,
                    delta,
                    decision,
                ) = report_case(
                    fast_runs,
                    reserve_runs,
                    deadline,
                )

                decisions[high_load] = decision

                label = (
                    "Высокая"
                    if high_load
                    else "Низкая"
                )

                print(
                    f"  {label} нагрузка: "
                    f"быстрое {fast_rate:.1%}, "
                    f"резерв {reserve_rate:.1%}, "
                    f"разница {delta:+.3f}, "
                    f"выбор: {decision}"
                )

            if decisions[False] != decisions[True]:
                print(
                    "  >>> ЗАГРУЗКА МЕНЯЕТ "
                    "ПРЕДПОЧТИТЕЛЬНУЮ "
                    "СТРАТЕГИЮ"
                )

    print(
        "\nЭксперимент завершён.",
        flush=True,
    )


if __name__ == "__main__":
    main()
