
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# =========================================
# Параметры
# =========================================

WORKERS = 4
NODES = 4
BLOCKS = 12
VECTOR_SIZE = 2400

NODE_SPEED = np.array([
    75_000.0,
    65_000.0,
    26_000.0,
    22_000.0,
])

NODE_RTT = np.array([
    0.004,
    0.005,
    0.010,
    0.011,
])

BANDWIDTHS = (
    250_000,
    350_000,
    450_000,
    500_000,
)

DEADLINE = 0.75
SWITCH_DELAY = 0.03
TRAIN_DELAY = 0.06

TARGET_LOSS = 0.27
LEARNING_RATE = 1.0
Q_SCALE = 1_000_000

ROUNDS = 160
REPEATS = 8
STATE_LENGTH = 20
NOISE_PROB = 0.10

FAST = "Быстрое"
RESERVE = "Резервирование"

POLICIES = (
    "Всегда быстрое",
    "Всегда резервирование",
    "Порог 400 КБ/с",
    "Контекстное правило",
    "Онлайн-обучение",
)


# =========================================
# Данные
# =========================================

def get_data():

    X, y = load_digits(
        return_X_y=True
    )

    X = X.astype(np.float64) / 16.0

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=7,
            stratify=y,
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=8,
            stratify=y_train,
        )
    )

    # Одинаковое число примеров
    # у каждого участника.
    n = (
        len(X_train) // WORKERS
    ) * WORKERS

    X_train = X_train[:n]
    y_train = y_train[:n]

    workers = []

    for i in range(WORKERS):

        start = i * n // WORKERS
        end = (i + 1) * n // WORKERS

        workers.append((
            X_train[start:end],
            y_train[start:end],
        ))

    return (
        workers,
        (X_val, y_val),
        (X_test, y_test),
    )


# =========================================
# Нейросеть 64 -> 30 -> 10
# =========================================

class MLP:

    def __init__(self, seed):

        rng = np.random.default_rng(seed)

        self.params = [
            rng.normal(
                0,
                np.sqrt(1 / 64),
                (64, 30),
            ),
            np.zeros(30),
            rng.normal(
                0,
                np.sqrt(1 / 30),
                (30, 10),
            ),
            np.zeros(10),
        ]

        self.nparam = sum(
            p.size for p in self.params
        )

        assert self.nparam <= VECTOR_SIZE

    def gradient(self, X, y):

        w1, b1, w2, b2 = self.params

        hidden = np.tanh(
            X @ w1 + b1
        )

        logits = (
            hidden @ w2 + b2
        )

        logits -= logits.max(
            axis=1,
            keepdims=True,
        )

        probabilities = np.exp(logits)

        probabilities /= probabilities.sum(
            axis=1,
            keepdims=True,
        )

        probabilities[
            np.arange(len(y)), y
        ] -= 1

        probabilities /= len(y)

        grad_w2 = (
            hidden.T @ probabilities
        )

        grad_b2 = probabilities.sum(
            axis=0
        )

        grad_hidden = (
            probabilities @ w2.T
        ) * (1 - hidden * hidden)

        grad_w1 = X.T @ grad_hidden

        grad_b1 = grad_hidden.sum(
            axis=0
        )

        return np.concatenate([
            grad_w1.ravel(),
            grad_b1,
            grad_w2.ravel(),
            grad_b2,
        ])

    def update(self, integer_sum):

        learning_rate = getattr(self, "learning_rate", LEARNING_RATE)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")

        gradient = (
            integer_sum[:self.nparam]
            / Q_SCALE
        )

        position = 0

        for parameter in self.params:

            size = parameter.size

            part = gradient[
                position:position + size
            ].reshape(parameter.shape)

            parameter -= (
                learning_rate * part
            )

            position += size

    def score(self, X, y):

        w1, b1, w2, b2 = self.params

        hidden = np.tanh(
            X @ w1 + b1
        )

        logits = hidden @ w2 + b2

        logits -= logits.max(
            axis=1,
            keepdims=True,
        )

        probabilities = np.exp(logits)

        probabilities /= probabilities.sum(
            axis=1,
            keepdims=True,
        )

        loss = -np.log(
            probabilities[
                np.arange(len(y)), y
            ] + 1e-12
        ).mean()

        accuracy = np.mean(
            probabilities.argmax(axis=1) == y
        )

        return float(loss), float(accuracy)


# =========================================
# Условия сети
# =========================================

def network_trace(seed):

    rng = np.random.default_rng(seed)

    scenarios = (
        "stable",
        "overload",
        "site_failure",
    )

    blocks = (
        ROUNDS + STATE_LENGTH - 1
    ) // STATE_LENGTH

    for block in range(blocks):

        bandwidth = int(
            rng.choice(BANDWIDTHS)
        )

        scenario = str(
            rng.choice(scenarios)
        )

        count = min(
            STATE_LENGTH,
            ROUNDS - block * STATE_LENGTH,
        )

        for _ in range(count):

            failed = (
                rng.random(NODES) < 0.01
            )

            slowdown = np.ones(NODES)

            if (
                scenario == "overload"
                and rng.random() < 0.60
            ):

                slowdown[:2] = (
                    rng.uniform(9, 13)
                )

            if scenario == "site_failure":

                if rng.random() < 0.30:
                    failed[:2] = True

                if rng.random() < 0.05:
                    failed[2:] = True

            corrupted = (
                rng.random(
                    (BLOCKS, NODES)
                ) < 0.002
            )

            high_load = bool(
                slowdown[0] >= 3.0
            )

            measured_load = high_load

            if rng.random() < NOISE_PROB:
                measured_load = (
                    not measured_load
                )

            yield (
                bandwidth,
                failed,
                slowdown,
                corrupted,
                measured_load,
            )


# =========================================
# Модель передачи и проверки суммы
# =========================================


def aggregate(updates, action, network):
    (
        bandwidth,
        failed,
        slowdown,
        corrupted,
        _,
    ) = network

    if action == FAST:
        placements = [
            [block % 2]
            for block in range(BLOCKS)
        ]
    else:
        placements = [
            [block % 2, 2 + block % 2]
            for block in range(BLOCKS)
        ]

    indices = np.array_split(
        np.arange(VECTOR_SIZE),
        BLOCKS,
    )

    sent_bytes = sum(
        len(indices[block])
        * WORKERS
        * 8
        * len(nodes)
        for block, nodes in enumerate(placements)
    )

    upload_time = sent_bytes / bandwidth

    # Доверенный эталон пока нужен
    # исключительно для проверки ячеек.
    # Позже заменим проверку отдельным
    # механизмом криптографической верификации.
    truth = updates.sum(
        axis=0,
        dtype=np.int64,
    )

    # Сохраняем полученные корректные
    # ячейки и время их поступления.
    verified_cells = [
        [] for _ in range(BLOCKS)
    ]

    for node in range(NODES):

        if failed[node]:
            continue

        queue_time = 0.0

        for block in range(BLOCKS):

            if node not in placements[block]:
                continue

            index = indices[block]

            queue_time += (
                NODE_RTT[node]
                + WORKERS
                * len(index)
                * slowdown[node]
                / NODE_SPEED[node]
            )

            # Реальный ответ симулируемого
            # агрегатора для своего блока.
            cell = updates[:, index].sum(
                axis=0,
                dtype=np.int64,
            )

            # Моделируем подмену результата.
            if corrupted[block, node]:
                cell[0] += 1

            # Пока используется доверенная
            # проверка вместо криптографии.
            if np.array_equal(
                cell,
                truth[index],
            ):
                verified_cells[block].append(
                    (
                        upload_time + queue_time,
                        cell.copy(),
                    )
                )

    # Если хотя бы один блок не получен,
    # сумму собрать невозможно.
    if not all(verified_cells):
        return None, sent_bytes, None

    reconstructed = np.empty(
        VECTOR_SIZE,
        dtype=np.int64,
    )

    completion_times = []

    for block, candidates in enumerate(
        verified_cells
    ):
        # Берём первую корректную ячейку.
        finish_time, cell = min(
            candidates,
            key=lambda item: item[0],
        )

        reconstructed[
            indices[block]
        ] = cell

        completion_times.append(
            finish_time
        )

    latency = max(completion_times)

    # Контрольный инвариант симуляции.
    assert np.array_equal(
        reconstructed,
        truth,
    ), "Ошибка восстановления суммы"

    return (
        latency,
        sent_bytes,
        reconstructed,
    )



# =========================================
# Обучающийся онлайн-планировщик
# =========================================

class Planner:

    def __init__(self, seed):

        self.rng = np.random.default_rng(seed)

        self.success = defaultdict(int)
        self.count = defaultdict(int)

    def choose(self, bandwidth, load, previous):

        state = bandwidth, load

        # Первичное исследование
        # каждого доступного действия.
        for action in (FAST, RESERVE):

            if self.count[state, action] < 3:
                return action

        # Небольшая вероятность
        # пробовать другое действие.
        if self.rng.random() < 0.05:

            return str(
                self.rng.choice([
                    FAST,
                    RESERVE,
                ])
            )

        def score(action):

            key = state, action

            probability = (
                self.success[key] + 1
            ) / (
                self.count[key] + 2
            )

            traffic_penalty = (
                0.08
                if action == RESERVE
                else 0.0
            )

            switch_penalty = (
                0.03
                if (
                    previous is not None
                    and previous != action
                )
                else 0.0
            )

            return (
                probability
                - traffic_penalty
                - switch_penalty
            )

        return max(
            (FAST, RESERVE),
            key=score,
        )

    def observe(
        self,
        bandwidth,
        load,
        action,
        success,
    ):

        key = (bandwidth, load), action

        self.count[key] += 1
        self.success[key] += int(success)


def choose_action(
    policy,
    bandwidth,
    load,
    previous,
    planner,
):

    if policy == "Всегда быстрое":
        return FAST

    if policy == "Всегда резервирование":
        return RESERVE

    if policy == "Порог 400 КБ/с":

        return (
            RESERVE
            if bandwidth >= 400_000
            else FAST
        )

    if policy == "Контекстное правило":

        return (
            RESERVE
            if (
                load
                and bandwidth >= 350_000
            )
            else FAST
        )

    return planner.choose(
        bandwidth,
        load,
        previous,
    )


# =========================================
# Один независимый запуск
# =========================================

def one_run(seed, workers, validation, test):

    models = {
        name: MLP(seed)
        for name in POLICIES
    }

    planners = {
        name: Planner(seed + 13)
        for name in POLICIES
    }

    previous = {
        name: None
        for name in POLICIES
    }

    stats = {
        name: {
            "time": 0.0,
            "hit": None,
            "success": 0,
            "switches": 0,
            "bytes": 0,
            "losses": [],
            "test_accuracy": 0.0,
        }
        for name in POLICIES
    }

    for network in network_trace(
        seed * 100 + 17
    ):

        (
            bandwidth,
            _,
            _,
            _,
            load,
        ) = network

        for name in POLICIES:

            model = models[name]
            stat = stats[name]

            # После достижения порога
            # дополнительное обучение
            # этой стратегии не требуется.
            if stat["hit"] is not None:
                continue

            action = choose_action(
                name,
                bandwidth,
                load,
                previous[name],
                planners[name],
            )

            changed = (
                previous[name] is not None
                and previous[name] != action
            )

            previous[name] = action

            # Каждый участник вычисляет
            # градиент по своей части данных.
            updates = np.zeros(
                (WORKERS, VECTOR_SIZE),
                dtype=np.int64,
            )

            for i, (X, y) in enumerate(workers):

                gradient = (
                    model.gradient(X, y)
                    / WORKERS
                )

                updates[
                    i, :model.nparam
                ] = np.rint(
                    gradient * Q_SCALE
                ).astype(np.int64)

            (
                latency,
                sent_bytes,
                integer_sum,
            ) = aggregate(
                updates,
                action,
                network,
            )

            actual_latency = (
                latency + SWITCH_DELAY * changed
                if latency is not None
                else None
            )

            success = (
                actual_latency is not None
                and actual_latency <= DEADLINE
            )

            # После дедлайна перестаём
            # ждать данный раунд.
            aggregation_time = (
                min(DEADLINE, actual_latency)
                if actual_latency is not None
                else DEADLINE
            )

            stat["time"] += (
                TRAIN_DELAY
                + aggregation_time
            )

            stat["bytes"] += sent_bytes

            stat["switches"] += int(changed)

            planners[name].observe(
                bandwidth,
                load,
                action,
                success,
            )

            if success:

                # Обновляем параметры только
                # после успешной проверки.
                model.update(integer_sum)

                stat["success"] += 1

            loss, _ = model.score(
                *validation
            )

            stat["losses"].append((
                stat["time"],
                loss,
            ))

            if loss <= TARGET_LOSS:
                stat["hit"] = stat["time"]

    for name in POLICIES:

        _, accuracy = models[name].score(
            *test
        )

        stats[name]["test_accuracy"] = accuracy

    return stats


# =========================================
# Основной эксперимент
# =========================================

def main():

    workers, validation, test = get_data()

    print(
        "Обучение MLP 64-30-10 "
        "на рукописных цифрах"
    )

    print(
        f"Параметров модели: 2260; "
        f"участников: {WORKERS}; "
        f"edge-узлов: {NODES}"
    )

    print(
        f"Целевой validation loss: "
        f"{TARGET_LOSS}"
    )

    print(
        f"Дедлайн: {DEADLINE} с; "
        f"максимум раундов: {ROUNDS}; "
        f"запусков: {REPEATS}"
    )

    print(
        f"Условное время вычисления "
        f"градиентов: {TRAIN_DELAY} с/раунд"
    )

    print(
        "Время моделируется; "
        "это не измерение реальной сети.\n",
        flush=True,
    )

    runs = []

    for i in range(REPEATS):

        result = one_run(
            2026 + i,
            workers,
            validation,
            test,
        )

        runs.append(result)

        print(
            f"Готов запуск {i + 1}/{REPEATS}",
            flush=True,
        )

    print("\nРезультаты:\n")

    for name in POLICIES:

        data = [
            result[name]
            for result in runs
        ]

        reached = [
            row["hit"]
            for row in data
            if row["hit"] is not None
        ]

        print(name)

        print(
            f"  Достигли порога: "
            f"{len(reached)}/{REPEATS}"
        )

        if reached:

            print(
                f"  Медиана T_eps "
                f"среди достигших: "
                f"{np.median(reached):.2f} с"
            )

        print(
            f"  Среднее число успешных "
            f"раундов: "
            f"{np.mean([x['success'] for x in data]):.1f}"
        )

        print(
            f"  Тестовая accuracy: "
            f"{np.mean([x['test_accuracy'] for x in data]):.3f}"
        )

        print(
            f"  Переключений: "
            f"{np.mean([x['switches'] for x in data]):.1f}"
        )

        print(
            f"  Передано данных: "
            f"{np.mean([x['bytes'] for x in data]) / 1e6:.2f} МБ\n"
        )

    # =====================================
    # Траектория обучения
    # =====================================

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    first_run = runs[0]

    for name in POLICIES:

        points = first_run[name]["losses"]

        ax.plot(
            [t for t, loss in points],
            [loss for t, loss in points],
            label=name,
        )

    ax.axhline(
        TARGET_LOSS,
        linestyle="--",
        label="Целевой loss",
    )

    ax.set_xlabel(
        "Смоделированное время, с"
    )

    ax.set_ylabel(
        "Validation loss"
    )

    ax.set_title(
        "Скорость обучения MLP "
        "при разных стратегиях"
    )

    ax.set_ylim(0, 1.2)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    output_dir = (
        Path(__file__).resolve().parent
        / "results"
    )

    output_dir.mkdir(exist_ok=True)

    output = (
        output_dir
        / "training_quality.png"
    )

    fig.savefig(output, dpi=160)

    print(
        f"График: {output}",
        flush=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
