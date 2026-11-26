
import csv
import json
import secrets
import socket
import struct
import threading

from pathlib import Path
from time import perf_counter

import numpy as np

import crypto_baseline as baseline
import randomized_check as projection


DIMENSIONS = (50_000, 200_000)
WORKER_COUNTS = (4, 16)

REPETITIONS = 5
WARMUP = 1
EPOCH = 1


# =========================================
# Передача сообщений через TCP
# =========================================

def send_frame(sock, data):
    sock.sendall(
        struct.pack("!I", len(data))
    )
    sock.sendall(data)

    return len(data) + 4


def recv_exact(sock, size):
    result = bytearray()

    while len(result) < size:
        chunk = sock.recv(
            size - len(result)
        )

        if not chunk:
            raise ConnectionError(
                "Соединение закрыто"
            )

        result.extend(chunk)

    return bytes(result)


def recv_frame(sock):
    size = struct.unpack(
        "!I",
        recv_exact(sock, 4)
    )[0]

    if size > 5_000_000:
        raise ValueError(
            "Неожиданно большой пакет"
        )

    data = recv_exact(sock, size)

    return data, size + 4


# =========================================
# Участник
# =========================================

def worker_client(port, worker):

    with socket.create_connection(
        ("127.0.0.1", port),
        timeout=60
    ) as sock:

        sock.settimeout(60)

        # Убираем задержки, связанные
        # с буферизацией малых TCP-пакетов.
        sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )

        send_frame(
            sock,
            struct.pack(
                "!I", worker.worker_id
            )
        )

        while True:

            command_bytes, _ = recv_frame(
                sock
            )

            cmd = json.loads(
                command_bytes
            )

            if cmd["kind"] == "stop":
                break

            session = bytes.fromhex(
                cmd["session"]
            )

            if cmd["kind"] == "baseline":

                packet = baseline.make_packet(
                    worker.private_key,
                    worker.worker_id,
                    worker.update,
                    session,
                    EPOCH,
                    cmd["round"]
                )

                send_frame(
                    sock,
                    packet.payload
                )

                send_frame(
                    sock,
                    packet.signature
                )

            elif cmd["kind"] == "projection":

                challenge = projection.Challenge(
                    session=session,
                    epoch=EPOCH,
                    round_id=cmd["round"],
                    candidate_hash=bytes.fromhex(
                        cmd["hash"]
                    ),
                    dimension=worker.update.size,
                    seeds=tuple(
                        bytes.fromhex(s)
                        for s in cmd["seeds"]
                    )
                )

                reply = worker.reply(
                    challenge
                )

                tags = struct.pack(
                    "!" + "I" * len(reply.tags),
                    *reply.tags
                )

                send_frame(sock, tags)

                send_frame(
                    sock,
                    reply.signature
                )

            else:
                raise ValueError(
                    "Неизвестная команда"
                )


# =========================================
# Один проверочный раунд
# =========================================

def one_measurement(
    method,
    links,
    public_keys,
    candidate,
    session,
    round_id
):

    dimension = len(candidate)

    transferred = 0

    start = perf_counter()

    if method == "baseline":

        command = json.dumps({
            "kind": "baseline",
            "session": session.hex(),
            "round": round_id
        }).encode()

        for sock in links.values():
            transferred += send_frame(
                sock, command
            )

        packets = []

        for worker_id, sock in links.items():

            payload, count = recv_frame(
                sock
            )
            transferred += count

            signature, count = recv_frame(
                sock
            )
            transferred += count

            packets.append(
                baseline.SignedUpdate(
                    worker_id=worker_id,
                    session_id=session,
                    epoch=EPOCH,
                    round_id=round_id,
                    vector_size=dimension,
                    payload=payload,
                    signature=signature
                )
            )

        result = baseline.verify_and_reconstruct(
            packets,
            public_keys,
            session,
            EPOCH,
            round_id,
            [
                x.copy()
                for x in np.array_split(
                    candidate,
                    baseline.BLOCKS
                )
            ]
        )

    elif method == "projection":

        verifier = projection.Verifier(
            public_keys,
            session,
            EPOCH,
            round_id
        )

        # Сначала фиксируем результат,
        # затем создаём случайный вызов.
        challenge = verifier.begin(
            candidate
        )

        command = json.dumps({
            "kind": "projection",
            "session": session.hex(),
            "round": round_id,
            "hash": (
                challenge.candidate_hash.hex()
            ),
            "seeds": [
                s.hex()
                for s in challenge.seeds
            ]
        }).encode()

        for sock in links.values():
            transferred += send_frame(
                sock, command
            )

        replies = []

        for worker_id, sock in links.items():

            tags_bytes, count = recv_frame(
                sock
            )
            transferred += count

            expected = (
                4 * len(challenge.seeds)
            )

            if len(tags_bytes) != expected:
                raise ValueError(
                    "Неверная длина проекций"
                )

            tags = struct.unpack(
                "!" + "I" * len(challenge.seeds),
                tags_bytes
            )

            signature, count = recv_frame(
                sock
            )
            transferred += count

            replies.append(
                projection.Reply(
                    worker_id,
                    tags,
                    signature
                )
            )

        result = verifier.finish(
            replies
        )

    else:
        raise ValueError(method)

    elapsed_ms = (
        perf_counter() - start
    ) * 1000

    assert np.array_equal(
        result,
        candidate
    ), "Ошибка проверки суммы"

    return elapsed_ms, transferred


# =========================================
# Серия измерений
# =========================================

def run_case(workers_n, dimension):

    baseline.VECTOR_SIZE = dimension

    rng = np.random.default_rng(
        10_000 + workers_n + dimension
    )

    updates = rng.integers(
        -100,
        101,
        size=(workers_n, dimension),
        dtype=np.int64
    )

    workers = [
        projection.Worker(
            i,
            updates[i]
        )
        for i in range(workers_n)
    ]

    public_keys = {
        w.worker_id: w.public_key
        for w in workers
    }

    # Сумма подготовлена тестовым стендом.
    # Её формирование не входит
    # в измеряемое время проверки.
    candidate = updates.sum(
        axis=0,
        dtype=np.int64
    )

    session = secrets.token_bytes(16)

    listener = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    listener.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    listener.bind(
        ("127.0.0.1", 0)
    )

    listener.listen(workers_n)

    port = listener.getsockname()[1]

    threads = [
        threading.Thread(
            target=worker_client,
            args=(port, worker),
            daemon=True
        )
        for worker in workers
    ]

    for thread in threads:
        thread.start()

    links = {}

    try:

        for _ in range(workers_n):

            sock, _ = listener.accept()

            sock.settimeout(60)

            sock.setsockopt(
                socket.IPPROTO_TCP,
                socket.TCP_NODELAY,
                1
            )

            worker_id_data, _ = recv_frame(
                sock
            )

            worker_id = struct.unpack(
                "!I", worker_id_data
            )[0]

            links[worker_id] = sock

        assert len(links) == workers_n

        measurements = {
            "baseline": [],
            "projection": []
        }

        for trial in range(
            WARMUP + REPETITIONS
        ):

            # Чередуем порядок методов,
            # чтобы уменьшить влияние прогрева.
            if trial % 2 == 0:
                methods = (
                    "baseline",
                    "projection"
                )
            else:
                methods = (
                    "projection",
                    "baseline"
                )

            for index, method in enumerate(
                methods
            ):

                elapsed, transferred = (
                    one_measurement(
                        method,
                        links,
                        public_keys,
                        candidate,
                        session,
                        100 + trial * 2 + index
                    )
                )

                if trial >= WARMUP:

                    measurements[method].append(
                        (elapsed, transferred)
                    )

        rows = []

        for method in (
            "baseline",
            "projection"
        ):

            times = [
                t
                for t, _ in measurements[method]
            ]

            bytes_transferred = [
                b
                for _, b in measurements[method]
            ]

            rows.append({
                "workers": workers_n,
                "dimension": dimension,
                "method": method,
                "median_ms": float(
                    np.median(times)
                ),
                "p95_ms": float(
                    np.percentile(times, 95)
                ),
                "worker_wire_mb": float(
                    np.mean(bytes_transferred) / 1e6
                )
            })

        return rows

    finally:

        for sock in links.values():

            try:
                send_frame(
                    sock,
                    b'{"kind":"stop"}'
                )
                sock.close()

            except OSError:
                pass

        listener.close()

        for thread in threads:
            thread.join(timeout=3)


# =========================================
# Вывод результатов
# =========================================

def main():

    print(
        "TCP-эксперимент: localhost",
        flush=True
    )

    print(
        "Время включает сетевой обмен "
        "с участниками и CPU проверки."
    )

    print(
        "Передача обновлений агрегатору "
        "и доставка самой суммы "
        "не включены."
    )

    print(
        "Скорость канала и RTT "
        "искусственно не ограничиваются.\n"
    )

    rows = []

    for workers_n in WORKER_COUNTS:

        for dimension in DIMENSIONS:

            print(
                f"Участников: {workers_n}; "
                f"размер: {dimension:,}",
                flush=True
            )

            group = run_case(
                workers_n,
                dimension
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
                    flush=True
                )

            print()

    output = (
        Path(__file__).resolve().parent
        / "results"
        / "tcp_loopback.csv"
    )

    output.parent.mkdir(
        exist_ok=True
    )

    with output.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Результаты сохранены: {output}"
    )


if __name__ == "__main__":
    main()
