
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import main as sim


BANDWIDTHS = (250_000, 350_000, 450_000, 500_000)

TRAIN_ROUNDS = 720
TEST_ROUNDS = 960
REPEATS = 8
BLOCK_LENGTH = 20

SWITCH_SECONDS = 0.03
TRAFFIC_PRICE = 0.08
SWITCH_PRICE = 0.03

EXPLORE_PER_ACTION = 6
SENSOR_FLIP_PROB = 0.10

FAST = "Быстрое"
RESERVE = "С резервированием"

POLICIES = (
    "Всегда быстрое",
    "Всегда резервирование",
    "Порог 400 КБ/с",
    "Обучение: только канал",
    "Обучение: канал + нагрузка",
)


@dataclass
class Observation:
    bandwidth: int
    load_high: bool
    outcomes: dict


class MeanPlanner:

    def __init__(self, uses_load):
        self.uses_load = uses_load
        self.good = defaultdict(int)
        self.total = defaultdict(int)

    def state(self, obs):
        if self.uses_load:
            return obs.bandwidth, obs.load_high
        return obs.bandwidth

    def choose(self, obs, previous, training):
        state = self.state(obs)

        if training:
            for action in (FAST, RESERVE):
                key = state, action

                if (
                    self.total[key]
                    < EXPLORE_PER_ACTION
                ):
                    return action

        scores = {}

        for action in (FAST, RESERVE):
            key = state, action

            probability = (
                self.good[key] + 1
            ) / (
                self.total[key] + 2
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
            (FAST, RESERVE),
            key=lambda a: scores[a],
        )

    def observe(self, obs, action, success):
        key = self.state(obs), action

        self.total[key] += 1
        self.good[key] += int(success)


def network_stream(rng, rounds):

    for offset in range(
        0, rounds, BLOCK_LENGTH
    ):
        bandwidth = int(
            rng.choice(BANDWIDTHS)
        )

        scenario = str(
            rng.choice(sim.SCENARIOS)
        )

        block_size = min(
            BLOCK_LENGTH,
            rounds - offset,
        )

        for _ in range(block_size):

            updates = rng.integers(
                -5,
                6,
                size=(
                    sim.WORKERS,
                    sim.VECTOR_SIZE,
                ),
                dtype=np.int64,
            )

            context = sim.make_context(
                rng,
                scenario,
            )

            # Мониторинг текущей нагрузки.
            # Измерение содержит шум.
            measured_load = (
                context.slowdown[0]
                * rng.lognormal(0, 0.2)
            )

            load_high = bool(
                measured_load >= 3.0
            )

            # В 10% случаев
            # датчик ошибается.
            if (
                rng.random()
                < SENSOR_FLIP_PROB
            ):
                load_high = not load_high

            sim.SHARED_BANDWIDTH = bandwidth

            # Исходы обеих стратегий нужны
            # для одинакового тестового стенда.
            # Алгоритму будет показан только
            # результат выбранного действия.
            outcomes = {
                action: sim.run_round(
                    updates,
                    context,
                    action,
                )
                for action in (FAST, RESERVE)
            }

            yield Observation(
                bandwidth,
                load_high,
                outcomes,
            )


def choose_action(
    policy,
    obs,
    previous,
    learners,
    training,
):

    if policy == "Всегда быстрое":
        return FAST

    if policy == "Всегда резервирование":
        return RESERVE

    if policy == "Порог 400 КБ/с":
        return (
            RESERVE
            if obs.bandwidth >= 400_000
            else FAST
        )

    return learners[policy].choose(
        obs,
        previous,
        training,
    )


def one_run(seed):

    learners = {
        "Обучение: только канал":
            MeanPlanner(False),

        "Обучение: канал + нагрузка":
            MeanPlanner(True),
    }

    previous = {
        p: None for p in POLICIES
    }

    metrics = {
        p: {
            "on_time": 0,
            "recovered": 0,
            "bytes": 0,
            "switches": 0,
            "utility": 0.0,
        }
        for p in POLICIES
    }

    choices = defaultdict(
        lambda: [0, 0]
    )

    stages = (
        ("train", TRAIN_ROUNDS),
        ("test", TEST_ROUNDS),
    )

    for stage, count in stages:

        rng = np.random.default_rng(
            seed * 100
            + (1 if stage == "train" else 2)
        )

        for obs in network_stream(
            rng,
            count,
        ):

            for policy in POLICIES:

                action = choose_action(
                    policy,
                    obs,
                    previous[policy],
                    learners,
                    training=(stage == "train"),
                )

                switched = (
                    previous[policy] is not None
                    and previous[policy] != action
                )

                previous[policy] = action

                result = obs.outcomes[action]

                latency = (
                    result.latency
                    + SWITCH_SECONDS * int(switched)
                    if result.latency is not None
                    else None
                )

                success = (
                    latency is not None
                    and latency <= sim.DEADLINE
                )

                # Обучение только по результату
                # выбранной стратегии.
                if policy in learners:
                    learners[policy].observe(
                        obs,
                        action,
                        success,
                    )

                if stage != "test":
                    continue

                m = metrics[policy]

                m["on_time"] += int(success)

                m["recovered"] += int(
                    result.latency is not None
                )

                m["bytes"] += result.sent_bytes

                m["switches"] += int(switched)

                m["utility"] += (
                    int(success)
                    - TRAFFIC_PRICE * int(
                        action == RESERVE
                    )
                    - SWITCH_PRICE * int(switched)
                )

                if policy in learners:

                    key = (
                        policy,
                        obs.bandwidth,
                        obs.load_high,
                    )

                    choices[key][0] += int(
                        action == RESERVE
                    )

                    choices[key][1] += 1

    return metrics, choices


def main():

    print(
        "Эксперимент: влияние "
        "наблюдаемой загрузки",
        flush=True,
    )

    print(
        f"Обучение: {TRAIN_ROUNDS}; "
        f"тест: {TEST_ROUNDS}; "
        f"запусков: {REPEATS}"
    )

    print(
        f"Стоимость смены: "
        f"{SWITCH_SECONDS} с; "
        f"дедлайн: {sim.DEADLINE} с"
    )

    print(
        f"Ошибка датчика: "
        f"{SENSOR_FLIP_PROB:.0%}\n"
    )

    statistics = {
        p: [] for p in POLICIES
    }

    decisions = defaultdict(
        lambda: [0, 0]
    )

    for i in range(REPEATS):

        report, choices = one_run(
            9200 + i
        )

        for policy in POLICIES:
            statistics[policy].append(
                report[policy]
            )

        for key, count in choices.items():
            decisions[key][0] += count[0]
            decisions[key][1] += count[1]

        print(
            f"Готово {i + 1}/{REPEATS}",
            flush=True,
        )

    success_means = []

    print("\nТестовые результаты:\n")

    for policy in POLICIES:

        data = statistics[policy]

        success = np.array([
            x["on_time"] / TEST_ROUNDS
            for x in data
        ])

        recovered = np.mean([
            x["recovered"] / TEST_ROUNDS
            for x in data
        ])

        traffic = np.mean([
            x["bytes"] / TEST_ROUNDS / 1e6
            for x in data
        ])

        switches = np.mean([
            x["switches"]
            for x in data
        ])

        utility = np.mean([
            x["utility"] / TEST_ROUNDS
            for x in data
        ])

        success_means.append(
            float(success.mean())
        )

        print(
            f"{policy}:\n"
            f"  В срок: "
            f"{success.mean():.1%} ± "
            f"{success.std(ddof=1):.1%}\n"
            f"  Восстановлено: {recovered:.1%}\n"
            f"  Трафик: {traffic:.3f} МБ/раунд\n"
            f"  Переключений: {switches:.1f}\n"
            f"  Полезность: {utility:.3f}\n"
        )

    print(
        "Доля выбора резервирования "
        "при 450 и 500 КБ/с:\n"
    )

    for bandwidth in (
        450_000,
        500_000,
    ):

        for policy in (
            "Обучение: только канал",
            "Обучение: канал + нагрузка",
        ):

            for high in (False, True):

                reserve, total = decisions[
                    policy,
                    bandwidth,
                    high,
                ]

                share = (
                    reserve / total
                    if total
                    else float("nan")
                )

                load = (
                    "высокая"
                    if high
                    else "низкая"
                )

                print(
                    f"{bandwidth // 1000} КБ/с, "
                    f"{policy}, "
                    f"нагрузка {load}: "
                    f"{share:.1%} "
                    f"({total} раундов)"
                )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        range(len(POLICIES)),
        success_means,
    )

    ax.set_xticks(
        range(len(POLICIES)),
        POLICIES,
        rotation=17,
        ha="right",
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel(
        "Доля раундов, завершённых в срок"
    )

    ax.set_title(
        "Влияние наблюдаемой загрузки "
        "на онлайн-планирование"
    )

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
        / "context_comparison.png"
    )

    fig.savefig(
        output,
        dpi=180,
    )

    print(
        f"\nГрафик: {output}",
        flush=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
