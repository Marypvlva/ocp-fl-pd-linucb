
"""Замкнутый цикл совместного планирования.

Обучение, сборка суммы и криптопроверка
выполняются в каждом выбранном раунде.

Сетевые задержки и отказы моделируются.
"""

import csv
import secrets

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import joint_planning_experiment as exp
import crypto_baseline as base
import randomized_check as proj
import training_experiment as train


COUNT = 12
RUNS = 3

POLICIES = (
    "F+B",
    "F+P",
    "R+B",
    "R+P",
    "Правило",
    "Онлайн",
)


# =====================================
# Сборка результата из edge-фрагментов
# =====================================

def edge_aggregate(updates, action, sample):

    (
        edge_bw,
        _,
        _,
        failed,
        slowdown,
        corruption,
        _,
        _,
    ) = sample

    count, dimension = updates.shape

    copies = (
        2 if action.startswith("R") else 1
    )

    upload = (
        count * dimension * 8
        * copies / edge_bw
    )

    queues = np.zeros(4)

    result = np.empty(
        dimension,
        dtype=np.int64,
    )

    blocks = np.array_split(
        np.arange(dimension),
        12,
    )

    for block, indices in enumerate(blocks):

        if copies == 1:
            nodes = (block % 2,)
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
                exp.NODE_RTT[node]
                + count
                * len(indices)
                * slowdown[node]
                / exp.NODE_SPEED[node]
            )

            # Фрагмент действительно
            # вычисляется из обновлений.
            cell = updates[
                :, indices
            ].sum(
                axis=0,
                dtype=np.int64,
            )

            if corruption[block, node]:
                cell = cell.copy()
                cell[0] += 1

            candidates.append((
                upload + queues[node],
                cell,
            ))

        if not candidates:
            return None

        # Берём самую раннюю реплику,
        # заранее не зная её корректности.
        _, selected = min(
            candidates,
            key=lambda item: item[0],
        )

        result[indices] = selected

    return result


# =====================================
# Модель времени и объёма передачи
# =====================================

def network_delay(
    action,
    sample,
    previous,
    count,
    dimension,
):

    (
        edge_bw,
        verify_bw,
        _,
        failed,
        slowdown,
        corruption,
        available,
        extra_rtt,
    ) = sample

    copies = (
        2 if action.startswith("R") else 1
    )

    is_projection = action.endswith("P")

    switched = (
        previous is not None
        and action != previous
    )

    payload = dimension * 8

    transferred = (
        count * payload * copies
    )

    # Время размещения и передачи
    # моделируется отдельно от
    # фактического вычисления суммы.
    edge_time, _ = exp.placement(
        edge_bw,
        failed,
        slowdown,
        corruption,
        copies,
        count,
        dimension,
    )

    if edge_time is None:
        return (
            exp.DEADLINE,
            transferred,
            switched,
            False,
        )

    # Общая передача готовой суммы.
    transferred += payload

    if is_projection:

        transferred += count * (
            64 + 8 + 64
        )

        verification_time = (
            payload / verify_bw
            + exp.CALIB_MS[count][1] / 1000
            + extra_rtt
        )

    else:

        transferred += count * (
            payload + 64
        )

        verification_time = (
            (count + 1)
            * payload / verify_bw
            + exp.CALIB_MS[count][0] / 1000
        )

    finish_time = (
        edge_time
        + verification_time
        + exp.SWITCH_SECONDS * switched
    )

    if is_projection and not available:
        return (
            exp.DEADLINE,
            transferred,
            switched,
            False,
        )

    return (
        min(exp.DEADLINE, finish_time),
        transferred,
        switched,
        finish_time <= exp.DEADLINE,
    )


# =====================================
# Настоящая проверка выбранным методом
# =====================================

def cryptographic_check(
    action,
    candidate,
    updates,
    workers,
    public_keys,
    session,
    round_id,
):

    if candidate is None:
        return False, None, 0.0

    start = perf_counter()

    try:

        if action.endswith("B"):

            packets = [
                base.make_packet(
                    worker.private_key,
                    worker.worker_id,
                    vector,
                    session,
                    1,
                    round_id,
                )
                for worker, vector
                in zip(workers, updates)
            ]

            cells = [
                part.copy()
                for part in np.array_split(
                    candidate,
                    base.BLOCKS,
                )
            ]

            verified = (
                base.verify_and_reconstruct(
                    packets,
                    public_keys,
                    session,
                    1,
                    round_id,
                    cells,
                )
            )

        else:

            verifier = proj.Verifier(
                public_keys,
                session,
                1,
                round_id,
            )

            # Кандидат фиксируется
            # ДО получения случайного вызова.
            challenge = verifier.begin(
                candidate
            )

            replies = [
                worker.reply(challenge)
                for worker in workers
            ]

            verified = verifier.finish(
                replies
            )

    except ValueError:
        return (
            False,
            None,
            perf_counter() - start,
        )

    return (
        True,
        verified,
        perf_counter() - start,
    )


# =====================================
# Проверка обнаружения подмены
# =====================================

def selftest(dimension):

    updates = np.zeros(
        (COUNT, dimension),
        dtype=np.int64,
    )

    workers = [
        proj.Worker(i, updates[i])
        for i in range(COUNT)
    ]

    public_keys = {
        worker.worker_id: worker.public_key
        for worker in workers
    }

    session = secrets.token_bytes(16)

    clean = np.zeros(
        (12, 4),
        dtype=bool,
    )

    corrupted = clean.copy()
    corrupted[0, 0] = True

    def make_sample(corruption):

        return (
            60e6,
            160e6,
            False,
            np.zeros(4, dtype=bool),
            np.ones(4),
            corruption,
            True,
            0.03,
        )

    for action in ("F+B", "F+P"):

        candidate_good = edge_aggregate(
            updates,
            action,
            make_sample(clean),
        )

        candidate_bad = edge_aggregate(
            updates,
            action,
            make_sample(corrupted),
        )

        good_ok, _, _ = cryptographic_check(
            action,
            candidate_good,
            updates,
            workers,
            public_keys,
            session,
            1,
        )

        bad_ok, _, _ = cryptographic_check(
            action,
            candidate_bad,
            updates,
            workers,
            public_keys,
            session,
            2,
        )

        assert good_ok
        assert not bad_ok, (
            f"{action}: подмена не обнаружена"
        )

    print(
        "Самотест подмены для B и P: OK",
        flush=True,
    )


# =====================================
# Один запуск одной политики
# =====================================

def run_one(
    policy,
    trace,
    groups,
    validation,
    test,
    seed,
):

    model = exp.WideMLP(2026)

    dimension = model.nparam

    planner = exp.OnlinePlanner(seed)

    workers = [
        proj.Worker(
            i,
            np.zeros(
                dimension,
                dtype=np.int64,
            ),
        )
        for i in range(COUNT)
    ]

    public_keys = {
        worker.worker_id: worker.public_key
        for worker in workers
    }

    session = secrets.token_bytes(16)

    total_examples = sum(
        len(y)
        for _, y in groups
    )

    previous = None

    total_seconds = 0.0
    transferred = 0
    accepted = 0
    rejected = 0
    checked = 0
    switches = 0
    crypto_cpu_seconds = 0.0

    loss, _ = model.score(
        *validation
    )

    attempts = 0

    for round_id, sample in enumerate(
        trace,
        start=1,
    ):

        if loss <= exp.TARGET_LOSS:
            break

        attempts += 1

        # ---------------------------------
        # Выбор стратегии
        # ---------------------------------

        if policy in exp.ACTIONS:

            action = policy

        elif policy == "Правило":

            action = exp.simple_rule(
                sample,
                COUNT,
                dimension,
            )

        else:

            action = planner.choose(
                sample,
                previous,
            )

        # ---------------------------------
        # Настоящие градиенты
        # ---------------------------------

        updates = []
        worker_times = []

        for X, y in groups:

            started = perf_counter()

            gradient = model.gradient(
                X, y
            )

            worker_times.append(
                perf_counter() - started
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

        updates = np.stack(updates)

        if (
            np.max(np.abs(updates))
            > proj.MAX_ABS_UPDATE
        ):
            raise ValueError(
                "Градиент вне допустимого диапазона"
            )

        for worker, vector in zip(
            workers,
            updates,
        ):
            worker.update = vector.copy()

        # ---------------------------------
        # Получение результата агрегаторов
        # ---------------------------------

        candidate = edge_aggregate(
            updates,
            action,
            sample,
        )

        (
            elapsed,
            sent,
            switched,
            in_time,
        ) = network_delay(
            action,
            sample,
            previous,
            COUNT,
            dimension,
        )

        # ---------------------------------
        # Настоящая криптопроверка
        # ---------------------------------

        available = sample[6]

        can_check = (
            candidate is not None
            and (
                not action.endswith("P")
                or available
            )
        )

        if can_check:

            (
                crypto_ok,
                verified,
                crypto_seconds,
            ) = cryptographic_check(
                action,
                candidate,
                updates,
                workers,
                public_keys,
                session,
                round_id,
            )

            checked += 1

            rejected += int(
                not crypto_ok
            )

            crypto_cpu_seconds += (
                crypto_seconds
            )

        else:

            crypto_ok = False
            verified = None

        # Контрольный инвариант стенда:
        # эталон используется ТОЛЬКО
        # после завершения протокола
        # и не влияет на его решение.

        if crypto_ok:

            reference = updates.sum(
                axis=0,
                dtype=np.int64,
            )

            assert np.array_equal(
                verified,
                reference,
            ), (
                "Проверка приняла "
                "неправильную сумму"
            )

        success = (
            crypto_ok and in_time
        )

        # Сверка с предыдущим
        # симулятором — только тест,
        # не основание принятия решения.

        (
            oracle_success,
            oracle_elapsed,
            oracle_bytes,
            oracle_switched,
        ) = exp.outcome(
            action,
            sample,
            previous,
            COUNT,
            dimension,
        )

        assert abs(
            oracle_elapsed - elapsed
        ) < 1e-10

        assert oracle_bytes == sent
        assert oracle_switched == switched

        assert (
            oracle_success == success
        ), (
            "Реальная криптопроверка "
            "и модель расходятся"
        )

        # ---------------------------------
        # Обновление модели
        # ---------------------------------

        if success:

            model.update(
                verified
            )

            accepted += 1

        # Сетевое время модельное,
        # время градиента — измеренное.
        total_seconds += (
            elapsed
            + max(worker_times)
        )

        transferred += sent

        switches += int(
            switched
        )

        if policy == "Онлайн":

            planner.observe(
                sample,
                action,
                success,
                elapsed,
            )

        previous = action

        loss, _ = model.score(
            *validation
        )

    _, accuracy = model.score(
        *test
    )

    reached = (
        loss <= exp.TARGET_LOSS
    )

    return {
        "policy": policy,
        "reached": reached,
        "time_s": (
            total_seconds
            if reached
            else None
        ),
        "attempts": attempts,
        "accepted": accepted,
        "crypto_checks": checked,
        "crypto_rejections": rejected,
        "crypto_cpu_ms": (
            crypto_cpu_seconds * 1000
        ),
        "switches": switches,
        "traffic_mb": transferred / 1e6,
        "validation_loss": loss,
        "test_accuracy": accuracy,
    }


# =====================================
# Эксперимент
# =====================================

def main():

    proj.MAX_ABS_UPDATE = 200_000

    groups, validation, test = (
        exp.make_workers(COUNT)
    )

    dimension = (
        exp.WideMLP(2026).nparam
    )

    base.VECTOR_SIZE = dimension

    selftest(dimension)

    print(
        "\nЗамкнутый цикл: "
        "агрегация и проверка каждого раунда"
    )

    print(
        f"Участников: {COUNT}; "
        f"параметров: {dimension}; "
        f"запусков: {RUNS}"
    )

    print(
        f"Целевой loss: {exp.TARGET_LOSS}; "
        f"дедлайн: {exp.DEADLINE} с"
    )

    print(
        "Время до качества — модель сети "
        "+ измерения градиентов, "
        "не wall-clock.\n",
        flush=True,
    )

    rows = []

    for run in range(RUNS):

        trace = exp.make_trace(
            1000 * COUNT + run,
            COUNT,
        )

        for policy in POLICIES:

            result = run_one(
                policy,
                trace,
                groups,
                validation,
                test,
                seed=900 + run * 10,
            )

            result["run"] = run

            rows.append(
                result
            )

            if result["reached"]:

                status = (
                    f"{result['time_s']:.2f} с"
                )

            else:

                status = (
                    "порог не достигнут"
                )

            print(
                f"Запуск {run + 1}/{RUNS} | "
                f"{policy:8s} | "
                f"{status} | "
                f"принято "
                f"{result['accepted']} | "
                f"крипто-отклонений "
                f"{result['crypto_rejections']}",
                flush=True,
            )

    print("\nСводка:")

    for policy in POLICIES:

        subset = [
            result
            for result in rows
            if result["policy"] == policy
        ]

        times = [
            result["time_s"]
            for result in subset
            if result["reached"]
        ]

        if times:

            median = (
                f"{np.median(times):.2f} с"
            )

        else:

            median = "—"

        mean_traffic = np.mean([
            result["traffic_mb"]
            for result in subset
        ])

        mean_crypto = np.mean([
            result["crypto_cpu_ms"]
            for result in subset
        ])

        print(
            f"{policy:8s}: "
            f"достигли {len(times)}/{RUNS}, "
            f"медиана {median}, "
            f"трафик {mean_traffic:.1f} МБ, "
            f"CPU криптопроверки "
            f"{mean_crypto:.0f} мс"
        )

    # ---------------------------------
    # CSV
    # ---------------------------------

    folder = (
        Path(__file__).resolve().parent
        / "results"
    )

    folder.mkdir(
        exist_ok=True
    )

    csv_path = (
        folder
        / "closed_loop_joint.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nCSV: {csv_path}"
    )

    # ---------------------------------
    # График
    # ---------------------------------

    medians = []

    for policy in POLICIES:

        times = [
            result["time_s"]
            for result in rows
            if (
                result["policy"] == policy
                and result["reached"]
            )
        ]

        medians.append(
            float(np.median(times))
            if times
            else np.nan
        )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        POLICIES,
        medians,
    )

    ax.set_ylabel(
        "Медиана смоделированного "
        "времени до качества, с"
    )

    ax.set_title(
        f"Замкнутый цикл, "
        f"{COUNT} участников"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    figure_path = (
        folder
        / "closed_loop_joint.png"
    )

    fig.savefig(
        figure_path,
        dpi=160,
    )

    print(
        f"График: {figure_path}",
        flush=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
