
from time import perf_counter_ns
import secrets

import numpy as np

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

import crypto_baseline as base


SIZES = (2260, 50_000, 200_000)
WORKER_COUNTS = (4, 16)

TRIALS = 12
WARMUP = 2


def milliseconds(values):
    return np.asarray(
        values, dtype=np.float64
    ) / 1_000_000


def benchmark(workers, dimension, seed):

    # В базовом модуле размер вектора
    # задан глобальной константой.
    # Для этого отдельного эксперимента
    # последовательно изменяем её.
    base.VECTOR_SIZE = dimension

    rng = np.random.default_rng(seed)

    updates = rng.integers(
        -100,
        101,
        size=(workers, dimension),
        dtype=np.int64,
    )

    # Генерация ключей не включается
    # в измерение стоимости раунда.
    keys = [
        Ed25519PrivateKey.generate()
        for _ in range(workers)
    ]

    public_keys = {
        i: key.public_key()
        for i, key in enumerate(keys)
    }

    session = secrets.token_bytes(16)

    expected = updates.sum(
        axis=0,
        dtype=np.int64,
    )

    cells = [
        block.copy()
        for block in np.array_split(
            expected,
            base.BLOCKS,
        )
    ]

    timings = {
        name: []
        for name in (
            "sign",
            "verify_signatures",
            "verify_all",
            "plain_sum",
        )
    }

    for repeat in range(TRIALS + WARMUP):

        # 1. Подписание всех обновлений
        start = perf_counter_ns()

        packets = [
            base.make_packet(
                keys[i],
                i,
                updates[i],
                session,
                1,
                repeat + 1,
            )
            for i in range(workers)
        ]

        sign_time = (
            perf_counter_ns() - start
        )

        # 2. Только проверка подписей
        # Без проверки суммы!
        start = perf_counter_ns()

        for packet in packets:
            public_keys[
                packet.worker_id
            ].verify(
                packet.signature,
                base.message(packet),
            )

        signature_time = (
            perf_counter_ns() - start
        )

        # 3. Проверка подписей,
        # пересчёт суммы и сравнение
        # с результатом агрегаторов
        start = perf_counter_ns()

        reconstructed = (
            base.verify_and_reconstruct(
                packets,
                public_keys,
                session,
                1,
                repeat + 1,
                cells,
            )
        )

        full_time = (
            perf_counter_ns() - start
        )

        # 4. Обычное суммирование
        # без криптографии
        start = perf_counter_ns()

        direct = updates.sum(
            axis=0,
            dtype=np.int64,
        )

        sum_time = (
            perf_counter_ns() - start
        )

        assert np.array_equal(
            reconstructed,
            direct,
        )

        # Первые повторы не учитываем
        # из-за эффекта прогрева.
        if repeat >= WARMUP:

            timings["sign"].append(
                sign_time
            )

            timings[
                "verify_signatures"
            ].append(signature_time)

            timings["verify_all"].append(
                full_time
            )

            timings["plain_sum"].append(
                sum_time
            )

    statistics = {}

    for name, samples in timings.items():

        times = milliseconds(samples)

        statistics[name] = (
            float(np.median(times)),
            float(np.percentile(times, 95)),
        )

    payload_bytes = (
        workers * dimension * 8
    )

    signature_bytes = sum(
        len(packet.signature)
        for packet in packets
    )

    return (
        statistics,
        payload_bytes,
        signature_bytes,
    )


def main():

    print(
        "Масштабирование "
        "криптографической проверки"
    )

    print(
        f"Повторов: {TRIALS}; "
        f"прогревов: {WARMUP}"
    )

    print(
        "Все времена в миллисекундах."
    )

    print(
        "Показываются медиана и p95."
    )

    print(
        "Генерация ключей и передача "
        "по сети не измеряются.\n"
    )

    for workers in WORKER_COUNTS:

        for dimension in SIZES:

            (
                statistics,
                payload_bytes,
                signature_bytes,
            ) = benchmark(
                workers,
                dimension,
                1000 + dimension + workers,
            )

            print("=" * 58)

            print(
                f"Участников: {workers}; "
                f"размер обновления: "
                f"{dimension:,}"
            )

            print(
                f"Входные данные: "
                f"{payload_bytes / 1e6:.2f} МБ"
            )

            print(
                f"Подписи: "
                f"{signature_bytes} байт\n"
            )

            labels = (
                ("plain_sum",
                 "Обычное суммирование"),
                ("sign",
                 "Подписание обновлений"),
                ("verify_signatures",
                 "Только проверка подписей"),
                ("verify_all",
                 "Подписи + пересчёт суммы"),
            )

            for key, label in labels:

                median_ms, p95_ms = (
                    statistics[key]
                )

                print(
                    f"{label}: "
                    f"медиана {median_ms:.3f} мс; "
                    f"p95 {p95_ms:.3f} мс"
                )

            print(flush=True)

    print(
        "\nЭксперимент завершён."
    )


if __name__ == "__main__":
    main()
