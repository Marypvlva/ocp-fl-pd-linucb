
import csv
import json
import multiprocessing as mp
import secrets
import socket

from pathlib import Path

import numpy as np

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

import crypto_baseline as baseline
import randomized_check as projection
import tcp_verification_experiment as tcp


# Те же конфигурации, что и в TCP-тесте.
DIMENSIONS = (50_000, 200_000)
WORKER_COUNTS = (4, 16)

WARMUP = 2
REPETITIONS = 8


def process_worker(
    port,
    worker_id,
    update,
    raw_private_key,
):
    # Каждый процесс создаёт свой объект
    # участника с тем же ключом, который
    # известен проверяющему.
    worker = projection.Worker(
        worker_id,
        update,
    )

    private_key = (
        Ed25519PrivateKey.from_private_bytes(
            raw_private_key
        )
    )

    worker.private_key = private_key
    worker.public_key = (
        private_key.public_key()
    )

    # Используем тот же TCP-протокол,
    # что и в эксперименте с потоками.
    tcp.worker_client(port, worker)


def run_case(workers_n, dimension):

    baseline.VECTOR_SIZE = dimension

    # Тот же seed, что в старом TCP-тесте.
    rng = np.random.default_rng(
        10_000 + workers_n + dimension
    )

    updates = rng.integers(
        -100,
        101,
        size=(workers_n, dimension),
        dtype=np.int64,
    )

    workers = [
        projection.Worker(
            i,
            updates[i],
        )
        for i in range(workers_n)
    ]

    public_keys = {
        w.worker_id: w.public_key
        for w in workers
    }

    candidate = updates.sum(
        axis=0,
        dtype=np.int64,
    )

    session = secrets.token_bytes(16)

    listener = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    listener.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    listener.bind(
        ("127.0.0.1", 0)
    )

    listener.listen(workers_n)
    listener.settimeout(120)

    port = listener.getsockname()[1]

    # spawn совместим с macOS.
    context = mp.get_context("spawn")

    processes = []

    for worker in workers:

        raw_key = worker.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=(
                serialization.NoEncryption()
            ),
        )

        process = context.Process(
            target=process_worker,
            args=(
                port,
                worker.worker_id,
                worker.update,
                raw_key,
            ),
        )

        processes.append(process)

    links = {}

    try:
        for process in processes:
            process.start()

        # Участники подключаются
        # независимо друг от друга.
        for _ in range(workers_n):

            sock, _ = listener.accept()

            sock.settimeout(60)

            sock.setsockopt(
                socket.IPPROTO_TCP,
                socket.TCP_NODELAY,
                1,
            )

            identity, _ = tcp.recv_frame(
                sock
            )

            worker_id = int.from_bytes(
                identity,
                byteorder="big",
            )

            if (
                worker_id not in public_keys
                or worker_id in links
            ):
                sock.close()
                raise ValueError(
                    "Неверный идентификатор участника"
                )

            links[worker_id] = sock

        results = {
            "baseline": [],
            "projection": [],
        }

        for trial in range(
            WARMUP + REPETITIONS
        ):

            # Чередуем порядок проверки.
            if trial % 2 == 0:
                methods = (
                    "baseline",
                    "projection",
                )
            else:
                methods = (
                    "projection",
                    "baseline",
                )

            for index, method in enumerate(
                methods
            ):

                elapsed, transferred = (
                    tcp.one_measurement(
                        method,
                        links,
                        public_keys,
                        candidate,
                        session,
                        100 + trial * 2 + index,
                    )
                )

                if trial >= WARMUP:
                    results[method].append(
                        (elapsed, transferred)
                    )

        rows = []

        for method in (
            "baseline",
            "projection",
        ):

            elapsed = np.array([
                item[0]
                for item in results[method]
            ])

            transferred = np.array([
                item[1]
                for item in results[method]
            ])

            rows.append({
                "workers": workers_n,
                "dimension": dimension,
                "method": method,
                "median_ms": float(
                    np.median(elapsed)
                ),
                "p95_ms": float(
                    np.percentile(elapsed, 95)
                ),
                "worker_wire_mb": float(
                    np.mean(transferred) / 1e6
                ),
            })

        return rows

    finally:

        stop_message = json.dumps({
            "kind": "stop"
        }).encode()

        for sock in links.values():
            try:
                tcp.send_frame(
                    sock,
                    stop_message,
                )
            except OSError:
                pass

            try:
                sock.close()
            except OSError:
                pass

        listener.close()

        for process in processes:
            if process.pid is None:
                continue

            process.join(timeout=5)

            if process.is_alive():
                process.terminate()
                process.join(timeout=2)


def main():

    print(
        "TCP-эксперимент: "
        "отдельные процессы",
        flush=True,
    )

    print(
        "Сеть: localhost, "
        "без ограничения пропускной способности"
    )

    print(
        f"Прогрев: {WARMUP}; "
        f"повторов: {REPETITIONS}"
    )

    print(
        "Время запуска процессов "
        "не учитывается.\n",
        flush=True,
    )

    rows = []

    for workers_n in WORKER_COUNTS:

        for dimension in DIMENSIONS:

            print(
                f"Участников: {workers_n}; "
                f"размер: {dimension:,}",
                flush=True,
            )

            group = run_case(
                workers_n,
                dimension,
            )

            rows.extend(group)

            for result in group:

                print(
                    f"  {result['method']}: "
                    f"медиана "
                    f"{result['median_ms']:.2f} мс; "
                    f"p95 "
                    f"{result['p95_ms']:.2f} мс; "
                    f"трафик "
                    f"{result['worker_wire_mb']:.3f} МБ",
                    flush=True,
                )

            baseline_time = next(
                item["median_ms"]
                for item in group
                if item["method"] == "baseline"
            )

            projection_time = next(
                item["median_ms"]
                for item in group
                if item["method"] == "projection"
            )

            print(
                f"  Разница (проекции - пересчёт): "
                f"{projection_time - baseline_time:+.2f} мс\n",
                flush=True,
            )

    folder = (
        Path(__file__).resolve().parent
        / "results"
    )

    folder.mkdir(exist_ok=True)

    output = (
        folder / "tcp_processes.csv"
    )

    with output.open(
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
        f"Результаты сохранены: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
