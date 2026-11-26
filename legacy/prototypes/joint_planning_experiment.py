
"""Совместное планирование размещения и проверки.

Градиенты и криптографические проверки
выполняются реально при построении
траектории обучения.

Сеть, размещение, отказы и задержки
проверки внутри раундов симулируются.
"""

from collections import defaultdict
from pathlib import Path
from time import perf_counter
import csv
import secrets

import matplotlib.pyplot as plt
import numpy as np

import training_experiment as train
import crypto_baseline as base
import randomized_check as proj


# ========================================
# Параметры эксперимента
# ========================================

PARTICIPANTS = (4, 12, 16)

HIDDEN = 670
TARGET_LOSS = 0.22

MAX_UPDATES = 95
NETWORK_ROUNDS = 150
SEEDS = 8

DEADLINE = 0.55
SWITCH_SECONDS = 0.025

EDGE_BW = (
    30e6,
    60e6,
    120e6,
)

VERIFY_BW = (
    20e6,
    60e6,
    160e6,
)

NODE_SPEED = (
    15e6,
    13e6,
    8e6,
    7e6,
)

NODE_RTT = (
    0.003,
    0.004,
    0.007,
    0.008,
)

# F = быстрое размещение
# R = резервирование
# B = полный пересчёт
# P = линейные проекции

ACTIONS = (
    "F+B",
    "F+P",
    "R+B",
    "R+P",
)

POLICIES = (
    *ACTIONS,
    "Правило",
    "Онлайн",
)

# Медианы предыдущих localhost TCP
# измерений, в миллисекундах.
#
# Размер обновления в измерениях:
# около 50 000 чисел.
#
# Для 12 участников используется
# приближённая калибровка.

CALIB_MS = {
    4: (2.65, 3.53),
    12: (6.72, 6.07),
    16: (9.66, 7.62),
}


# ========================================
# Нейросеть
# ========================================

class WideMLP(train.MLP):

    def __init__(self, seed):

        rng = np.random.default_rng(seed)

        self.params = [
            rng.normal(
                0,
                np.sqrt(1 / 64),
                (64, HIDDEN),
            ),
            np.zeros(HIDDEN),
            rng.normal(
                0,
                np.sqrt(1 / HIDDEN),
                (HIDDEN, 10),
            ),
            np.zeros(10),
        ]

        self.nparam = sum(
            p.size
            for p in self.params
        )


def make_workers(count):

    chunks, validation, test = (
        train.get_data()
    )

    X = np.concatenate([
        x for x, _ in chunks
    ])

    y = np.concatenate([
        labels
        for _, labels in chunks
    ])

    indices = np.array_split(
        np.arange(len(y)),
        count,
    )

    workers = [
        (X[ii], y[ii])
        for ii in indices
    ]

    return workers, validation, test


# ========================================
# Реальная траектория обучения
# и две криптографические проверки
# ========================================

def build_learning_curve(count):

    groups, validation, test = (
        make_workers(count)
    )

    model = WideMLP(2026)

    dimension = model.nparam

    base.VECTOR_SIZE = dimension

    proj.MAX_ABS_UPDATE = 200_000

    crypto_workers = [
        proj.Worker(
            i,
            np.zeros(
                dimension,
                dtype=np.int64,
            ),
        )
        for i in range(count)
    ]

    public_keys = {
        w.worker_id: w.public_key
        for w in crypto_workers
    }

    session = secrets.token_bytes(16)

    total_examples = sum(
        len(y)
        for _, y in groups
    )

    initial_loss, _ = model.score(
        *validation
    )

    losses = [initial_loss]
    gradient_times = []

    checked_tampering = False

    for round_id in range(
        1,
        MAX_UPDATES + 1,
    ):

        updates = []
        worker_times = []

        for X, y in groups:

            start = perf_counter()

            gradient = model.gradient(
                X, y
            )

            worker_times.append(
                perf_counter() - start
            )

            weighted = (
                gradient
                * (len(y) / total_examples)
                * train.Q_SCALE
            )

            updates.append(
                np.rint(weighted).astype(
                    np.int64
                )
            )

        updates = np.stack(
            updates
        )

        gradient_times.append(
            max(worker_times)
        )

        if (
            np.max(np.abs(updates))
            > proj.MAX_ABS_UPDATE
        ):
            raise ValueError(
                "Градиент вне диапазона проверки"
            )

        for worker, update in zip(
            crypto_workers,
            updates,
        ):
            worker.update = update.copy()

        # Кандидат формируется тестовым
        # агрегатором из обновлений.
        candidate = updates.sum(
            axis=0,
            dtype=np.int64,
        )

        # --------------------------------
        # B: подписи и полный пересчёт
        # --------------------------------

        packets = [
            base.make_packet(
                worker.private_key,
                worker.worker_id,
                worker.update,
                session,
                1,
                round_id,
            )
            for worker in crypto_workers
        ]

        cells = [
            part.copy()
            for part in np.array_split(
                candidate,
                base.BLOCKS,
            )
        ]

        verified_b = (
            base.verify_and_reconstruct(
                packets,
                public_keys,
                session,
                1,
                round_id,
                cells,
            )
        )

        # --------------------------------
        # P: линейные проекции
        # --------------------------------

        verifier = proj.Verifier(
            public_keys,
            session,
            1,
            round_id,
        )

        # Кандидат фиксируется
        # ДО генерации случайного вызова.

        challenge = verifier.begin(
            candidate
        )

        replies = [
            worker.reply(challenge)
            for worker in crypto_workers
        ]

        verified_p = verifier.finish(
            replies
        )

        assert np.array_equal(
            verified_b,
            verified_p,
        ), "Механизмы дали разные суммы"

        # --------------------------------
        # Контроль подмены результата
        # --------------------------------

        if not checked_tampering:

            changed = candidate.copy()

            changed[0] += 1

            changed_cells = [
                part.copy()
                for part in np.array_split(
                    changed,
                    base.BLOCKS,
                )
            ]

            try:
                base.verify_and_reconstruct(
                    packets,
                    public_keys,
                    session,
                    1,
                    round_id,
                    changed_cells,
                )

            except ValueError:
                pass

            else:
                raise AssertionError(
                    "Полный пересчёт принял "
                    "подменённую сумму"
                )

            bad_verifier = proj.Verifier(
                public_keys,
                session,
                1,
                round_id,
            )

            bad_challenge = (
                bad_verifier.begin(changed)
            )

            bad_replies = [
                worker.reply(bad_challenge)
                for worker in crypto_workers
            ]

            try:
                bad_verifier.finish(
                    bad_replies
                )

            except ValueError:
                pass

            else:
                raise AssertionError(
                    "Проекции приняли "
                    "подменённую сумму"
                )

            checked_tampering = True

        # --------------------------------
        # Шаг обучения
        # --------------------------------

        model.update(
            verified_b
        )

        loss, _ = model.score(
            *validation
        )

        losses.append(loss)

        if loss <= TARGET_LOSS:
            break

    _, accuracy = model.score(
        *test
    )

    return (
        np.asarray(losses),
        float(accuracy),
        dimension,
        float(
            np.median(gradient_times)
        ),
    )


# ========================================
# Генерация состояний сети
# ========================================

def make_trace(seed, count):

    rng = np.random.default_rng(seed)

    trace = []

    for t in range(NETWORK_ROUNDS):

        if t % 10 == 0:

            ebw = float(
                rng.choice(EDGE_BW)
            )

            vbw = float(
                rng.choice(VERIFY_BW)
            )

            high_load = bool(
                rng.random() < 0.30
            )

            dropout_rate = float(
                rng.choice(
                    (
                        0.0,
                        0.008,
                        0.025,
                    )
                )
            )

        failed = (
            rng.random(4) < 0.012
        )

        if rng.random() < 0.07:
            failed[:2] = True

        slowdown = np.ones(4)

        if high_load:

            slowdown[:2] = (
                rng.uniform(6, 10)
            )

        corrupted = (
            rng.random((12, 4)) < 0.002
        )

        # Участник может отключиться
        # после передачи обновления
        # агрегатору, но до проверки.

        available = bool(
            np.all(
                rng.random(count)
                >= dropout_rate
            )
        )

        # Датчик ошибается с
        # вероятностью 10%.

        measured_load = (
            high_load
            if rng.random() > 0.10
            else not high_load
        )

        extra_rtt = float(
            rng.uniform(
                0.015,
                0.055,
            )
        )

        trace.append((
            ebw,
            vbw,
            measured_load,
            failed,
            slowdown,
            corrupted,
            available,
            extra_rtt,
        ))

    return trace


# ========================================
# Размещение фрагментов
# ========================================

def placement(
    edge_bw,
    failed,
    slowdown,
    corruption,
    copies,
    count,
    dimension,
):

    slices = np.array_split(
        np.arange(dimension),
        12,
    )

    upload = (
        count
        * dimension
        * 8
        * copies
        / edge_bw
    )

    queues = np.zeros(4)

    block_finish = []

    any_corrupt = False

    for block, indices in enumerate(
        slices
    ):

        if copies == 1:

            nodes = (
                block % 2,
            )

        else:

            nodes = (
                block % 2,
                2 + block % 2,
            )

        candidates = []

        for node in nodes:

            if failed[node]:
                continue

            queues[node] += (
                NODE_RTT[node]
                + count
                * len(indices)
                * slowdown[node]
                / NODE_SPEED[node]
            )

            candidates.append((
                queues[node] + upload,
                node,
            ))

        if not candidates:
            return None, False

        finished, node = min(
            candidates
        )

        block_finish.append(
            finished
        )

        # Упрощение:
        # выбираем первую реплику.
        # Повторная попытка после
        # повреждения не моделируется.

        any_corrupt |= bool(
            corruption[block, node]
        )

    return (
        max(block_finish),
        any_corrupt,
    )


# ========================================
# Результат одного сетевого раунда
# ========================================

def outcome(
    action,
    sample,
    prev,
    count,
    dimension,
):

    (
        ebw,
        vbw,
        _,
        failed,
        slowdown,
        corrupted,
        available,
        extra_rtt,
    ) = sample

    copies = (
        2
        if action.startswith("R")
        else 1
    )

    uses_projection = (
        action.endswith("P")
    )

    edge_time, tampered = placement(
        ebw,
        failed,
        slowdown,
        corrupted,
        copies,
        count,
        dimension,
    )

    # Передача исходных обновлений
    # на агрегаторы.

    sent = (
        count
        * dimension
        * 8
        * copies
    )

    switched = (
        prev is not None
        and prev != action
    )

    if edge_time is None:

        return (
            False,
            DEADLINE,
            sent,
            switched,
        )

    # ====================================
    # ИСПРАВЛЕННЫЙ УЧЁТ ПРОВЕРКИ
    # ====================================

    # В ОБОИХ механизмах агрегированная
    # сумма передаётся проверяющему.

    sum_bytes = dimension * 8

    sent += sum_bytes

    if uses_projection:

        # P: агрегированная сумма,
        # подписанные проекции и
        # дополнительный RTT.

        sent += count * (
            64 + 8 + 64
        )

        verification_seconds = (
            sum_bytes / vbw
            + CALIB_MS[count][1] / 1000
            + extra_rtt
        )

    else:

        # B: агрегированная сумма
        # плюс count исходных обновлений.

        sent += count * (
            sum_bytes + 64
        )

        verification_seconds = (
            (count + 1)
            * sum_bytes
            / vbw
            + CALIB_MS[count][0] / 1000
        )

    total = (
        edge_time
        + verification_seconds
        + SWITCH_SECONDS * switched
    )

    success = (
        not tampered
        and (
            available
            or not uses_projection
        )
        and total <= DEADLINE
    )

    # При недоступных участниках P
    # не получает ответы на вызов и
    # ожидает до дедлайна.

    if (
        uses_projection
        and not available
    ):

        elapsed = DEADLINE

    else:

        elapsed = min(
            DEADLINE,
            total,
        )

    return (
        success,
        elapsed,
        sent,
        switched,
    )


# ========================================
# Онлайн-планировщик
# ========================================

class OnlinePlanner:

    def __init__(self, seed):

        self.rng = (
            np.random.default_rng(seed)
        )

        self.n = defaultdict(int)

        self.success = defaultdict(int)

        self.total_time = defaultdict(
            float
        )

    def state(self, sample):

        ebw, vbw, overload, *_ = sample

        return (
            ebw <= 30e6,
            vbw <= 60e6,
            overload,
        )

    def choose(self, sample, prev):

        state = self.state(sample)

        # Вероятность исследования
        # случайного действия — 12%.

        if self.rng.random() < 0.12:

            return str(
                self.rng.choice(
                    ACTIONS
                )
            )

        def score(action):

            key = (
                state,
                action,
            )

            n = self.n[key]

            p_success = (
                self.success[key] + 1
            ) / (
                n + 2
            )

            avg_time = (
                self.total_time[key]
                + 0.35 * 2
            ) / (
                n + 2
            )

            penalty = (
                SWITCH_SECONDS
                if (
                    prev is not None
                    and prev != action
                )
                else 0.0
            )

            # Оцениваем количество
            # успешных раундов
            # на единицу времени.

            return (
                p_success
                / (avg_time + penalty)
            )

        return max(
            ACTIONS,
            key=score,
        )

    def observe(
        self,
        sample,
        action,
        success,
        elapsed,
    ):

        key = (
            self.state(sample),
            action,
        )

        self.n[key] += 1

        self.success[key] += int(
            success
        )

        self.total_time[key] += elapsed


# ========================================
# Контекстное правило
# ========================================

def simple_rule(
    sample,
    count,
    dimension,
):

    ebw, vbw, high, *_ = sample

    place_choice = (
        "R"
        if high or ebw >= 60e6
        else "F"
    )

    cpu_b, cpu_p = (
        CALIB_MS[count]
    )

    # Формулы согласованы с outcome().

    baseline_time = (
        (count + 1)
        * dimension
        * 8
        / vbw
        + cpu_b / 1000
    )

    projection_time = (
        dimension
        * 8
        / vbw
        + 0.035
        + cpu_p / 1000
    )

    check_choice = (
        "P"
        if projection_time < baseline_time
        else "B"
    )

    return (
        f"{place_choice}"
        f"+{check_choice}"
    )


# ========================================
# Выполнение стратегии
# ========================================

def run_policy(
    name,
    trace,
    required,
    count,
    dimension,
    gradient_seconds,
    seed,
):

    planner = OnlinePlanner(seed)

    prev = None

    total_time = 0.0

    accepted = 0

    bytes_sent = 0

    switches = 0

    successes = 0

    for sample in trace:

        if name in ACTIONS:

            action = name

        elif name == "Правило":

            action = simple_rule(
                sample,
                count,
                dimension,
            )

        else:

            action = planner.choose(
                sample,
                prev,
            )

        (
            success,
            elapsed,
            sent,
            switched,
        ) = outcome(
            action,
            sample,
            prev,
            count,
            dimension,
        )

        total_time += (
            elapsed
            + gradient_seconds
        )

        bytes_sent += sent

        switches += int(
            switched
        )

        successes += int(
            success
        )

        if name == "Онлайн":

            planner.observe(
                sample,
                action,
                success,
                elapsed,
            )

        prev = action

        if success:

            accepted += 1

            if accepted >= required:

                return (
                    total_time,
                    bytes_sent,
                    switches,
                    successes,
                )

    return (
        None,
        bytes_sent,
        switches,
        successes,
    )


# ========================================
# Основной эксперимент
# ========================================

def main():

    print(
        "Совместное планирование "
        "размещения и проверки"
    )

    print(
        "Градиенты и криптопроверки: "
        "реальные на траектории обучения"
    )

    print(
        "WAN, сбои и edge-задержки: "
        "симуляция"
    )

    print(
        f"Дедлайн: {DEADLINE} с; "
        f"целевой loss: {TARGET_LOSS}; "
        f"повторов: {SEEDS}\n",
        flush=True,
    )

    outrows = []

    for count in PARTICIPANTS:

        (
            losses,
            accuracy,
            dimension,
            gradient_seconds,
        ) = build_learning_curve(
            count
        )

        if losses[-1] > TARGET_LOSS:

            print(
                f"{count} участников: "
                f"цель не достигнута за "
                f"{MAX_UPDATES} шагов"
            )

            continue

        required = len(losses) - 1

        print(
            f"\nУчастников: {count}; "
            f"параметров: {dimension}"
        )

        print(
            "Для достижения цели нужно "
            f"{required} успешных SGD-шагов"
        )

        print(
            f"Итоговый loss: "
            f"{losses[-1]:.4f}; "
            f"test accuracy: {accuracy:.3f}"
        )

        print(
            f"Градиент, max worker: "
            f"{gradient_seconds * 1000:.3f} мс",
            flush=True,
        )

        for i in range(SEEDS):

            trace = make_trace(
                1000 * count + i,
                count,
            )

            for policy in POLICIES:

                (
                    time_to_target,
                    sent,
                    switches,
                    successes,
                ) = run_policy(
                    policy,
                    trace,
                    required,
                    count,
                    dimension,
                    gradient_seconds,
                    900 + 10 * i,
                )

                outrows.append({
                    "workers": count,
                    "run": i,
                    "policy": policy,
                    "time_s": time_to_target,
                    "bytes_mb": sent / 1e6,
                    "switches": switches,
                    "successes": successes,
                    "reached": (
                        time_to_target
                        is not None
                    ),
                })

            print(
                f"  Готово {i + 1}/{SEEDS}",
                flush=True,
            )

        print("\nРезультаты:")

        for policy in POLICIES:

            group = [
                row
                for row in outrows
                if (
                    row["workers"] == count
                    and row["policy"] == policy
                )
            ]

            times = [
                row["time_s"]
                for row in group
                if row["reached"]
            ]

            if times:

                median = (
                    f"{np.median(times):.2f} с"
                )

            else:

                median = "не достигнута"

            traffic = np.mean([
                row["bytes_mb"]
                for row in group
            ])

            avg_switches = np.mean([
                row["switches"]
                for row in group
            ])

            print(
                f"  {policy}: "
                f"{len(times)}/{SEEDS} достигли; "
                f"медиана {median}; "
                f"трафик {traffic:.1f} МБ; "
                f"переключений "
                f"{avg_switches:.1f}"
            )

    if not outrows:

        print(
            "Нет результатов: "
            "порог качества не достигнут"
        )

        return

    # ====================================
    # CSV
    # ====================================

    directory = (
        Path(__file__).resolve().parent
        / "results"
    )

    directory.mkdir(
        exist_ok=True
    )

    output = (
        directory
        / "joint_planning.csv"
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(outrows[0]),
        )

        writer.writeheader()

        writer.writerows(
            outrows
        )

    print(
        f"\nРезультаты: {output}"
    )

    # ====================================
    # График
    # ====================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for policy in POLICIES:

        x = []
        y = []

        for count in PARTICIPANTS:

            matches = [
                row["time_s"]
                for row in outrows
                if (
                    row["workers"] == count
                    and row["policy"] == policy
                    and row["reached"]
                )
            ]

            if matches:

                x.append(count)

                y.append(
                    float(np.median(matches))
                )

        if x:

            ax.plot(
                x,
                y,
                marker="o",
                label=policy,
            )

    ax.set_xlabel(
        "Число участников"
    )

    ax.set_ylabel(
        "Медиана времени до качества, с"
    )

    ax.set_title(
        "Совместное планирование "
        "размещения и проверки"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        ncol=2,
        fontsize=9,
    )

    fig.tight_layout()

    figure = (
        directory
        / "joint_planning.png"
    )

    fig.savefig(
        figure,
        dpi=160,
    )

    print(
        f"График: {figure}",
        flush=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
