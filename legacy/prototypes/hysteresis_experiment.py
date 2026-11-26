
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import main as sim
import bandit_experiment as base


REPEATS = 8
EXPLORATION_PER_ACTION = 8
HYSTERESIS_MARGIN = 0.05

POLICIES = (
    "Всегда быстрое",
    "Всегда резервирование",
    "Порог 400 КБ/с",
    "Thompson Sampling",
    "Среднее без гистерезиса",
    "Среднее с гистерезисом",
)

LEARNERS = (
    "Thompson Sampling",
    "Среднее без гистерезиса",
    "Среднее с гистерезисом",
)


class MeanPlanner:
    def __init__(self, seed, margin):
        self.rng = np.random.default_rng(seed)
        self.margin = margin
        self.good = defaultdict(lambda: 1)
        self.bad = defaultdict(lambda: 1)
        self.count = defaultdict(int)

    def choose(self, bandwidth, previous, training):

        # На обучающей части пробуем
        # обе стратегии для каждого канала.
        if training:
            few = [
                action for action in base.ACTIONS
                if self.count[(bandwidth, action)]
                < EXPLORATION_PER_ACTION
            ]

            if few:
                low = min(
                    self.count[(bandwidth, action)]
                    for action in few
                )
                options = [
                    action for action in few
                    if self.count[(bandwidth, action)]
                    == low
                ]
                return str(self.rng.choice(options))

        # Оценка вероятности успеха
        # по ранее полученной обратной связи.
        means = {}

        for action in base.ACTIONS:
            key = bandwidth, action
            means[action] = (
                self.good[key]
                / (self.good[key] + self.bad[key])
            )

        if previous is None:
            return max(
                base.ACTIONS,
                key=lambda a: means[a]
            )

        challenger = next(
            action for action in base.ACTIONS
            if action != previous
        )

        # Переключаемся лишь при достаточно
        # большом преимуществе нового действия.
        if (
            means[challenger]
            > means[previous] + self.margin
        ):
            return challenger

        return previous

    def observe(self, bandwidth, action, success):
        key = bandwidth, action
        self.count[key] += 1

        if success:
            self.good[key] += 1
        else:
            self.bad[key] += 1


def one_run(seed):

    learners = {
        "Thompson Sampling":
            base.ThompsonPlanner(seed + 100_000),

        "Среднее без гистерезиса":
            MeanPlanner(seed + 200_000, 0.0),

        "Среднее с гистерезисом":
            MeanPlanner(
                seed + 200_000,
                HYSTERESIS_MARGIN
            ),
    }

    previous = {
        p: None for p in POLICIES
    }

    test = {
        p: [] for p in POLICIES
    }

    switches = {
        p: 0 for p in POLICIES
    }

    counts_350 = {
        p: [0, 0] for p in learners
    }

    stages = (
        ("train", base.TRAIN_ROUNDS),
        ("test", base.TEST_ROUNDS),
    )

    for stage, rounds in stages:

        rng = np.random.default_rng(
            seed * 100
            + (1 if stage == "train" else 2)
        )

        for bandwidth, scenario in base.network_stream(
            rng, rounds
        ):

            updates = rng.integers(
                -5, 6,
                size=(sim.WORKERS, sim.VECTOR_SIZE),
                dtype=np.int64,
            )

            context = sim.make_context(
                rng, scenario
            )

            # Выбор до получения результата.
            choices = {}

            for policy in POLICIES:

                if policy == "Thompson Sampling":
                    choices[policy] = (
                        learners[policy].choose(
                            bandwidth,
                            previous[policy],
                        )
                    )

                elif policy in learners:
                    choices[policy] = (
                        learners[policy].choose(
                            bandwidth,
                            previous[policy],
                            training=(stage == "train"),
                        )
                    )

                else:
                    choices[policy] = (
                        base.choose_fixed(
                            policy, bandwidth
                        )
                    )

            for policy, action in choices.items():

                changed = (
                    previous[policy] is not None
                    and previous[policy] != action
                )

                previous[policy] = action

                outcome = base.evaluate_round(
                    updates,
                    context,
                    action,
                    bandwidth,
                    changed,
                )

                # Обучение только по выбранному
                # действию, без контрфактического
                # результата другой стратегии.
                if policy in learners:
                    learners[policy].observe(
                        bandwidth,
                        action,
                        outcome["success"],
                    )

                if stage == "test":

                    switches[policy] += int(changed)
                    test[policy].append(outcome)

                    if (
                        bandwidth == 350_000
                        and policy in learners
                    ):
                        counts_350[policy][0] += int(
                            action == base.ACTIONS[1]
                        )
                        counts_350[policy][1] += 1

    summary = {}

    for policy in POLICIES:
        rows = test[policy]

        summary[policy] = {
            "ontime": float(np.mean([
                r["success"] for r in rows
            ])),
            "recovery": float(np.mean([
                r["recovered"] for r in rows
            ])),
            "mb": float(
                np.mean([
                    r["bytes"] for r in rows
                ]) / 1_000_000
            ),
            "switches": switches[policy],
        }

    return summary, counts_350


def main():

    print(
        f"Эксперимент: {REPEATS} запусков; "
        f"обучение {base.TRAIN_ROUNDS}, "
        f"тест {base.TEST_ROUNDS} раундов",
        flush=True,
    )

    print(
        f"Гистерезис: {HYSTERESIS_MARGIN:.2f}; "
        f"стоимость смены: "
        f"{base.SWITCH_SECONDS:.2f} с; "
        f"дедлайн: {sim.DEADLINE:.2f} с\n",
        flush=True,
    )

    results = []
    choices = defaultdict(lambda: [0, 0])

    for k in range(REPEATS):

        row, counts = one_run(2028 + k)
        results.append(row)

        for policy, (reserve, total) in counts.items():
            choices[policy][0] += reserve
            choices[policy][1] += total

        print(
            f"Готов запуск {k + 1}/{REPEATS}",
            flush=True,
        )

    averages = []

    print("\nТестовые результаты:")

    for policy in POLICIES:

        ontime = np.array([
            row[policy]["ontime"]
            for row in results
        ])

        recovery = np.mean([
            row[policy]["recovery"]
            for row in results
        ])

        mb = np.mean([
            row[policy]["mb"]
            for row in results
        ])

        sw = np.mean([
            row[policy]["switches"]
            for row in results
        ])

        averages.append(
            float(ontime.mean())
        )

        print(
            f"{policy}: "
            f"в срок {ontime.mean():.1%} "
            f"± {ontime.std(ddof=1):.1%} "
            f"(ст. откл.); "
            f"восстановлено {recovery:.1%}; "
            f"трафик {mb:.3f} МБ/раунд; "
            f"переключений {sw:.1f}"
        )

    print(
        "\nДоля резервирования при 350 КБ/с:"
    )

    for policy in LEARNERS:

        reserve, total = choices[policy]

        print(
            f"{policy}: {reserve / total:.1%} "
            f"({total} раундов)"
        )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        range(len(POLICIES)),
        averages,
    )

    ax.set_xticks(
        range(len(POLICIES)),
        POLICIES,
        rotation=18,
        ha="right",
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel(
        "Доля раундов с корректной суммой до срока"
    )

    ax.set_title(
        "Онлайн-планирование: влияние гистерезиса"
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
    output_dir.mkdir(exist_ok=True)

    path = (
        output_dir / "hysteresis_comparison.png"
    )

    fig.savefig(path, dpi=180)

    print(
        f"\nГрафик: {path}",
        flush=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
