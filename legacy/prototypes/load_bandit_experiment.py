
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import main as sim


# ==========================================
# Параметры
# ==========================================

BANDWIDTHS = (
    250_000,
    350_000,
    450_000,
    500_000,
)

DEADLINE = 0.75

TRAIN = 800
TEST = 1000
REPEATS = 8

BLOCK_LENGTH = 20
HIGH_LOAD_PROB = 0.40

SENSOR_ERRORS = (0.0, 0.10)

SWITCH_SECONDS = 0.03
TRAFFIC_PRICE = 0.08
SWITCH_PRICE = 0.03

EXPLORE_FIRST = 8

FAST = "Быстрое"
RESERVE = "С резервированием"

POLICIES = (
    "Всегда быстрое",
    "Всегда резервирование",
    "Порог 400 КБ/с",
    "Обучение: канал",
    "Обучение: канал + нагрузка",
)


# ==========================================
# Онлайн-планировщик
# ==========================================

class OnlinePlanner:

    def __init__(self, use_load, seed):
        self.use_load = use_load

        self.rng = np.random.default_rng(seed)

        self.success = defaultdict(int)
        self.count = defaultdict(int)

    def state(self, bandwidth, load_signal):

        if self.use_load:
            return bandwidth, load_signal

        return bandwidth

    def choose(
        self,
        bandwidth,
        load_signal,
        previous,
        training,
    ):

        state = self.state(
            bandwidth,
            load_signal,
        )

        # Первичное исследование действий
        # в каждом наблюдаемом состоянии.
        for action in (FAST, RESERVE):

            if (
                training
                and self.count[(state, action)]
                < EXPLORE_FIRST
            ):
                return action

        # Небольшая вероятность исследования.
        epsilon = (
            0.06 if training else 0.01
        )

        if self.rng.random() < epsilon:
            return str(
                self.rng.choice(
                    [FAST, RESERVE]
                )
            )

        scores = {}

        for action in (FAST, RESERVE):

            key = state, action

            # Сглаженная оценка
            # вероятности успеха.
            probability = (
                self.success[key] + 1
            ) / (
                self.count[key] + 2
            )

            traffic_cost = (
                TRAFFIC_PRICE
                if action == RESERVE
                else 0.0
            )

            change_cost = (
                SWITCH_PRICE
                if (
                    previous is not None
                    and action != previous
                )
                else 0.0
            )

            scores[action] = (
                probability
                - traffic_cost
                - change_cost
            )

        return max(
            scores,
            key=scores.get,
        )

    def observe(
        self,
        bandwidth,
        load_signal,
        action,
        success,
    ):

        key = (
            self.state(
                bandwidth,
                load_signal,
            ),
            action,
        )

        self.count[key] += 1
        self.success[key] += int(success)


# ==========================================
# Генерация условий сети
# ==========================================

def generate_rounds(
    seed,
    number,
    sensor_error,
):

    env_rng = np.random.default_rng(
        seed
    )

    sensor_rng = np.random.default_rng(
        seed + 900_000
    )

    for start in range(
        0,
        number,
        BLOCK_LENGTH,
    ):

        bandwidth = int(
            env_rng.choice(BANDWIDTHS)
        )

        actual_high = bool(
            env_rng.random()
            < HIGH_LOAD_PROB
        )

        block_size = min(
            BLOCK_LENGTH,
            number - start,
        )

        for _ in range(block_size):

            failed = tuple(
                bool(x)
                for x in (
                    env_rng.random(sim.NODES)
                    < 0.01
                )
            )

            slowdown = (
                [1.0] * sim.NODES
            )

            if actual_high:

                factor = float(
                    env_rng.uniform(
                        9.0,
                        13.0,
                    )
                )

                slowdown[0] = factor
                slowdown[1] = factor

            context = sim.Context(
                failed=failed,
                slowdown=tuple(slowdown),
                corrupted=(
                    env_rng.random(
                        (
                            sim.BLOCKS,
                            sim.NODES,
                        )
                    ) < 0.002
                ),
            )

            updates = env_rng.integers(
                -5,
                6,
                size=(
                    sim.WORKERS,
                    sim.VECTOR_SIZE,
                ),
                dtype=np.int64,
            )

            # Сигнал датчика.
            # Иногда отличается
            # от фактической загрузки.
            measured_high = actual_high

            if (
                sensor_rng.random()
                < sensor_error
            ):
                measured_high = (
                    not measured_high
                )

            sim.SHARED_BANDWIDTH = bandwidth

            # Симулятор рассчитывает
            # оба потенциальных исхода.
            # Планировщику сообщается
            # только результат
            # выбранного действия.
            outcomes = {
                action: sim.run_round(
                    updates,
                    context,
                    action,
                )
                for action in (
                    FAST,
                    RESERVE,
                )
            }

            yield (
                bandwidth,
                measured_high,
                outcomes,
            )


# ==========================================
# Один независимый запуск
# ==========================================

def run_once(seed, sensor_error):

    learners = {
        "Обучение: канал":
            OnlinePlanner(
                False,
                seed + 17,
            ),

        "Обучение: канал + нагрузка":
            OnlinePlanner(
                True,
                seed + 19,
            ),
    }

    previous = {
        policy: None
        for policy in POLICIES
    }

    results = {
        policy: {
            "success": 0,
            "recovered": 0,
            "bytes": 0,
            "switches": 0,
            "utility": 0.0,
        }
        for policy in POLICIES
    }

    decisions = defaultdict(
        lambda: [0, 0]
    )

    stages = (
        ("train", TRAIN, 11),
        ("test", TEST, 12),
    )

    for stage, count, offset in stages:

        # Оценки сохраняются после обучения,
        # но тест начинается без
        # предыдущей конфигурации.
        if stage == "test":
            previous = {
                policy: None
                for policy in POLICIES
            }

        for (
            bandwidth,
            load_signal,
            outcomes,
        ) in generate_rounds(
            seed * 100 + offset,
            count,
            sensor_error,
        ):

            for policy in POLICIES:

                if policy == "Всегда быстрое":
                    action = FAST

                elif policy == "Всегда резервирование":
                    action = RESERVE

                elif policy == "Порог 400 КБ/с":
                    action = (
                        RESERVE
                        if bandwidth >= 400_000
                        else FAST
                    )

                else:
                    action = learners[
                        policy
                    ].choose(
                        bandwidth,
                        load_signal,
                        previous[policy],
                        training=(
                            stage == "train"
                        ),
                    )

                changed = (
                    previous[policy]
                    is not None
                    and action != previous[policy]
                )

                previous[policy] = action

                result = outcomes[action]

                latency = (
                    result.latency
                    + SWITCH_SECONDS * changed
                    if result.latency is not None
                    else None
                )

                ontime = (
                    latency is not None
                    and latency <= DEADLINE
                )

                # Каждый обучающийся алгоритм
                # получает только собственный
                # фактический результат.
                if policy in learners:

                    learners[policy].observe(
                        bandwidth,
                        load_signal,
                        action,
                        ontime,
                    )

                if stage != "test":
                    continue

                record = results[policy]

                record["success"] += int(
                    ontime
                )

                record["recovered"] += int(
                    latency is not None
                )

                record["bytes"] += (
                    result.sent_bytes
                )

                record["switches"] += int(
                    changed
                )

                record["utility"] += (
                    int(ontime)
                    - TRAFFIC_PRICE * (
                        action == RESERVE
                    )
                    - SWITCH_PRICE * changed
                )

                if policy in learners:

                    key = (
                        policy,
                        bandwidth,
                        load_signal,
                    )

                    decisions[key][0] += int(
                        action == RESERVE
                    )

                    decisions[key][1] += 1

    return results, decisions


# ==========================================
# Основной эксперимент
# ==========================================

def main():

    print(
        "Онлайн-обучение: "
        "роль информации о загрузке",
        flush=True,
    )

    print(
        f"Дедлайн: {DEADLINE:.2f} с; "
        f"обучение: {TRAIN}; "
        f"тест: {TEST}; "
        f"запусков: {REPEATS}"
    )

    print(
        f"Цена трафика: {TRAFFIC_PRICE}; "
        f"стоимость смены: "
        f"{SWITCH_SECONDS} с "
        f"+ штраф {SWITCH_PRICE}"
    )

    figure_rows = []

    for sensor_error in SENSOR_ERRORS:

        print(
            f"\n{'=' * 65}"
        )

        print(
            f"Ошибка датчика: "
            f"{sensor_error:.0%}",
            flush=True,
        )

        runs = []

        decisions_all = defaultdict(
            lambda: [0, 0]
        )

        for i in range(REPEATS):

            summary, decisions = run_once(
                5000 + i,
                sensor_error,
            )

            runs.append(summary)

            for key, value in decisions.items():

                decisions_all[key][0] += value[0]
                decisions_all[key][1] += value[1]

            print(
                f"Готов запуск "
                f"{i + 1}/{REPEATS}",
                flush=True,
            )

        print(
            "\nРезультаты теста:"
        )

        for policy in POLICIES:

            success = np.array([
                r[policy]["success"] / TEST
                for r in runs
            ])

            utility = np.array([
                r[policy]["utility"] / TEST
                for r in runs
            ])

            switches = np.mean([
                r[policy]["switches"]
                for r in runs
            ])

            traffic = np.mean([
                r[policy]["bytes"]
                / TEST / 1e6
                for r in runs
            ])

            print(
                f"{policy}: "
                f"в срок {success.mean():.1%}, "
                f"полезность "
                f"{utility.mean():.3f}, "
                f"трафик {traffic:.3f} "
                f"МБ/раунд, "
                f"переключений "
                f"{switches:.1f}"
            )

            figure_rows.append(
                (
                    sensor_error,
                    policy,
                    utility.mean(),
                )
            )

        # Парное сравнение алгоритмов
        # на одинаковых сценариях.
        contextual = np.array([
            r[
                "Обучение: канал + нагрузка"
            ]["utility"] / TEST
            for r in runs
        ])

        channel = np.array([
            r[
                "Обучение: канал"
            ]["utility"] / TEST
            for r in runs
        ])

        difference = (
            contextual - channel
        )

        boot_rng = np.random.default_rng(
            300 + int(
                sensor_error * 1000
            )
        )

        indices = boot_rng.integers(
            0,
            REPEATS,
            size=(
                5000,
                REPEATS,
            ),
        )

        low, high = np.quantile(
            difference[
                indices
            ].mean(axis=1),
            [0.025, 0.975],
        )

        print(
            "\nРазница полезности "
            "(канал + нагрузка) "
            "минус (канал):"
        )

        print(
            f"{difference.mean():+.4f}; "
            f"95% bootstrap-интервал "
            f"[{low:+.4f}, {high:+.4f}]"
        )

        print(
            "\nДоли резервирования "
            "контекстного планировщика:"
        )

        for bandwidth in (
            350_000,
            450_000,
        ):

            for load in (
                False,
                True,
            ):

                key = (
                    "Обучение: канал + нагрузка",
                    bandwidth,
                    load,
                )

                yes, total = decisions_all[key]

                if total == 0:
                    continue

                label = (
                    "высокая"
                    if load
                    else "низкая"
                )

                print(
                    f"{bandwidth // 1000} КБ/с, "
                    f"{label} загрузка: "
                    f"{yes / total:.1%} "
                    f"(n={total})"
                )

    # ======================================
    # График
    # ======================================

    labels = (
        "Быстрое",
        "Резерв",
        "Порог",
        "Канал",
        "Канал + загрузка",
    )

    x = np.arange(
        len(POLICIES)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for i, noise in enumerate(
        SENSOR_ERRORS
    ):

        values = [
            next(
                value
                for error, name, value
                in figure_rows
                if (
                    error == noise
                    and name == policy
                )
            )
            for policy in POLICIES
        ]

        ax.bar(
            x + (i - 0.5) * 0.38,
            values,
            0.38,
            label=(
                f"Ошибка датчика "
                f"{noise:.0%}"
            ),
        )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Средняя полезность на тесте"
    )

    ax.set_title(
        "Влияние информации о загрузке"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    output_dir = (
        Path(__file__).resolve().parent
        / "results"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output = (
        output_dir
        / "load_bandit_comparison.png"
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
