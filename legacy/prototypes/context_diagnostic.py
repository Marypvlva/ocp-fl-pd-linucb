
from collections import defaultdict

import numpy as np
import main as sim
import context_experiment as exp


RUNS = 12
ROUNDS_PER_RUN = 1200
BOOTSTRAPS = 5000

TRAFFIC_PENALTY = 0.08

FAST = "Быстрое"
RESERVE = "С резервированием"

rng_boot = np.random.default_rng(12345)


def one_run(seed):
    rng = np.random.default_rng(seed)

    # Храним:
    # количество наблюдений,
    # успешность быстрого размещения,
    # успешность резервирования.
    stats = defaultdict(
        lambda: np.zeros(3, dtype=float)
    )

    for obs in exp.network_stream(
        rng, ROUNDS_PER_RUN
    ):
        fast = obs.outcomes[FAST]
        reserve = obs.outcomes[RESERVE]

        fast_ok = int(fast.on_time)
        reserve_ok = int(reserve.on_time)

        # Собираем статистику как
        # по каналу в целом, так и
        # отдельно по уровню нагрузки.
        keys = [
            (obs.bandwidth, None),
            (obs.bandwidth, False),
            (obs.bandwidth, True),
        ]

        for bandwidth, load in keys:

            if (
                load is not None
                and load != obs.load_high
            ):
                continue

            stats[(bandwidth, load)] += [
                1,
                fast_ok,
                reserve_ok,
            ]

    return stats


def analyze(rows, key):
    data = np.array([
        run.get(
            key,
            np.zeros(3),
        )
        for run in rows
    ])

    totals = data.sum(axis=0)

    n, fast_sum, reserve_sum = totals

    if n == 0:
        return None

    fast_rate = fast_sum / n
    reserve_rate = reserve_sum / n

    # Положительное значение означает
    # преимущество резервирования
    # после штрафа за лишний трафик.
    delta = (
        reserve_rate
        - fast_rate
        - TRAFFIC_PENALTY
    )

    # Bootstrap по независимым запускам,
    # а не по отдельным раундам.
    indices = rng_boot.integers(
        0, RUNS,
        size=(BOOTSTRAPS, RUNS),
    )

    sampled = data[indices].sum(axis=1)

    valid = sampled[:, 0] > 0
    sampled = sampled[valid]

    differences = (
        (sampled[:, 2] - sampled[:, 1])
        / sampled[:, 0]
        - TRAFFIC_PENALTY
    )

    lower, upper = np.percentile(
        differences,
        [2.5, 97.5],
    )

    if lower > 0:
        decision = "РЕЗЕРВ"
    elif upper < 0:
        decision = "БЫСТРОЕ"
    else:
        decision = "НЕЯСНО"

    return {
        "n": int(n),
        "fast": fast_rate,
        "reserve": reserve_rate,
        "delta": delta,
        "lower": lower,
        "upper": upper,
        "decision": decision,
    }


def main():
    print(
        "Диагностика информативности загрузки",
        flush=True,
    )

    print(
        f"Запусков: {RUNS}; "
        f"раундов на запуск: {ROUNDS_PER_RUN}"
    )

    print(
        f"Штраф резервирования: "
        f"{TRAFFIC_PENALTY:.2f}"
    )

    print(
        "Штраф переключения здесь "
        "не учитывается.\n"
    )

    runs = []

    for i in range(RUNS):
        runs.append(
            one_run(31000 + i)
        )

        print(
            f"Готово {i + 1}/{RUNS}",
            flush=True,
        )

    print(
        "\nРазность = успешность резерва "
        "- успешность быстрого "
        "- штраф за трафик."
    )

    print(
        "Положительная разность: "
        "лучше резервирование.\n"
    )

    for bandwidth in exp.BANDWIDTHS:
        print("=" * 65)

        print(
            f"Канал: {bandwidth // 1000} КБ/с"
        )

        decisions = {}

        for load, label in (
            (None, "Любая нагрузка"),
            (False, "Низкая нагрузка"),
            (True, "Высокая нагрузка"),
        ):
            result = analyze(
                runs,
                (bandwidth, load),
            )

            if result is None:
                continue

            decisions[load] = result[
                "decision"
            ]

            print(f"\n{label}")
            print(
                f"  Наблюдений: {result['n']}"
            )
            print(
                f"  Быстрое: "
                f"{result['fast']:.1%}"
            )
            print(
                f"  Резерв: "
                f"{result['reserve']:.1%}"
            )
            print(
                f"  Разность полезности: "
                f"{result['delta']:+.3f}"
            )
            print(
                f"  95% интервал: "
                f"[{result['lower']:+.3f}, "
                f"{result['upper']:+.3f}]"
            )
            print(
                f"  Вывод: "
                f"{result['decision']}"
            )

        if (
            decisions.get(False) in (
                "БЫСТРОЕ",
                "РЕЗЕРВ",
            )
            and decisions.get(True) in (
                "БЫСТРОЕ",
                "РЕЗЕРВ",
            )
            and decisions[False]
            != decisions[True]
        ):
            print(
                "\n>>> При этом канале "
                "нагрузка меняет "
                "предпочтительную стратегию."
            )

    print("\nДиагностика завершена.")


if __name__ == "__main__":
    main()
