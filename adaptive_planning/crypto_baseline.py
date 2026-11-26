from dataclasses import dataclass, replace
import hashlib
import secrets
import struct
from time import perf_counter

import numpy as np

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)


WORKERS = 4
VECTOR_SIZE = 2260
BLOCKS = 12

DOMAIN = b"EDGE-SIGNED-SUM/v1\x00"


def integer_vector(vector, bound=None):
    """Validate before casting: no floats, uint64 wrap or int64 overflow."""
    value = np.asarray(vector)
    if value.ndim != 1 or value.dtype.kind not in "iu" or not value.size:
        raise ValueError("Ожидался непустой целочисленный вектор")
    low, high = int(value.min()), int(value.max())
    limit = np.iinfo(np.int64).max if bound is None else int(bound)
    if limit < 0 or low < -limit or high > limit:
        raise ValueError("Значение вне допустимого диапазона")
    return value.astype("<i8", copy=False)


def validate_context(session, epoch, round_id):
    if not isinstance(session, bytes) or len(session) != 16:
        raise ValueError("Неверная длина идентификатора сессии")
    if any(type(x) is not int or not 0 <= x < 2**32 for x in (epoch, round_id)):
        raise ValueError("Неверная эпоха или раунд")


@dataclass(frozen=True)
class SignedUpdate:
    worker_id: int
    session_id: bytes
    epoch: int
    round_id: int
    vector_size: int
    payload: bytes
    signature: bytes


def message(packet):
    """Сообщение, защищённое подписью."""

    return digest_message(packet.worker_id, packet.session_id, packet.epoch, packet.round_id,
                          packet.vector_size, hashlib.sha256(packet.payload).digest())


def digest_message(worker_id, session, epoch, round_id, dimension, digest):
    validate_context(session, epoch, round_id)
    if any(type(v) is not int or not 0 <= v < 2**32 for v in (worker_id, dimension)):
        raise ValueError("Invalid worker or dimension")
    if not isinstance(digest, bytes) or len(digest) != 32:
        raise ValueError("Expected a SHA-256 digest")
    header = struct.pack(">16sIIII", session, epoch, round_id, worker_id, dimension)
    return DOMAIN + header + digest


def make_packet(
    private_key,
    worker_id,
    vector,
    session_id,
    epoch,
    round_id,
):
    vector = integer_vector(vector)

    unsigned = SignedUpdate(
        worker_id=worker_id,
        session_id=session_id,
        epoch=epoch,
        round_id=round_id,
        vector_size=vector.size,
        payload=vector.tobytes(),
        signature=b"",
    )

    signature = private_key.sign(message(unsigned))

    return replace(
        unsigned,
        signature=signature,
    )


def verify_and_reconstruct(
    packets,
    public_keys,
    session_id,
    epoch,
    round_id,
    cells,
    *,
    vector_size=None,
    blocks=None,
    max_abs_update=None,
):
    validate_context(session_id, epoch, round_id)
    dimension = VECTOR_SIZE if vector_size is None else vector_size
    block_count = BLOCKS if blocks is None else blocks
    if not public_keys or dimension < 1 or block_count < 1:
        raise ValueError("Пустая конфигурация проверки")
    safe_bound = np.iinfo(np.int64).max // len(public_keys)
    bound = safe_bound if max_abs_update is None else min(safe_bound, max_abs_update)
    if len(packets) != len(public_keys):
        raise ValueError("Не хватает обновлений участников")

    if len(cells) != block_count:
        raise ValueError("Неполный набор агрегированных блоков")

    seen = set()
    verified = []

    for packet in packets:
        if packet.worker_id in seen or packet.worker_id not in public_keys:
            raise ValueError("Повторный или неизвестный участник")

        seen.add(packet.worker_id)

        if (
            packet.session_id != session_id
            or packet.epoch != epoch
            or packet.round_id != round_id
            or packet.vector_size != dimension
        ):
            raise ValueError("Неверная сессия, эпоха или раунд")

        if len(packet.payload) != dimension * 8 or len(packet.signature) != 64:
            raise ValueError("Неверная длина обновления")

        try:
            public_keys[packet.worker_id].verify(
                packet.signature,
                message(packet),
            )

        except InvalidSignature as exc:
            raise ValueError("Неверная подпись участника") from exc

        vector = np.frombuffer(
            packet.payload,
            dtype="<i8",
        )

        verified.append(integer_vector(vector, bound))

    # Базовый вариант:
    # проверяющий пересчитывает всю сумму.
    expected = np.stack(verified).sum(
        axis=0,
        dtype=np.int64,
    )

    reference_blocks = np.array_split(
        expected,
        block_count,
    )

    for i, (received, reference) in enumerate(zip(cells, reference_blocks)):
        received = np.asarray(received)
        if received.dtype.kind not in "iu" or received.ndim != 1:
            raise ValueError("Неверный формат агрегированного блока")

        if not np.array_equal(
            received,
            reference,
        ):
            raise ValueError(f"Некорректная агрегированная ячейка: {i}")

    return np.concatenate(cells)


def require_rejection(
    label,
    packets,
    public_keys,
    session_id,
    epoch,
    round_id,
    cells,
):
    try:
        verify_and_reconstruct(
            packets,
            public_keys,
            session_id,
            epoch,
            round_id,
            cells,
        )

    except ValueError as exc:
        print(f"{label}: ОТКЛОНЕНО ({exc})")

    else:
        raise AssertionError(f"{label}: ошибка не обнаружена")


def main():

    rng = np.random.default_rng(101)

    # Отдельный ключ каждого участника.
    keys = {i: Ed25519PrivateKey.generate() for i in range(WORKERS)}

    public_keys = {i: key.public_key() for i, key in keys.items()}

    session_id = secrets.token_bytes(16)

    epoch = 3
    round_id = 7

    updates = rng.integers(
        -100,
        101,
        size=(WORKERS, VECTOR_SIZE),
        dtype=np.int64,
    )

    # ----------------------------------
    # Подписание обновлений
    # ----------------------------------

    start = perf_counter()

    packets = [
        make_packet(
            keys[i],
            i,
            updates[i],
            session_id,
            epoch,
            round_id,
        )
        for i in range(WORKERS)
    ]

    sign_ms = (perf_counter() - start) * 1000

    expected = updates.sum(
        axis=0,
        dtype=np.int64,
    )

    cells = [
        block.copy()
        for block in np.array_split(
            expected,
            BLOCKS,
        )
    ]

    # ----------------------------------
    # 1. Корректные обновления
    # ----------------------------------

    start = perf_counter()

    reconstructed = verify_and_reconstruct(
        packets,
        public_keys,
        session_id,
        epoch,
        round_id,
        cells,
    )

    verify_ms = (perf_counter() - start) * 1000

    assert np.array_equal(
        reconstructed,
        expected,
    )

    print("1. Подписанные обновления и правильная сумма: OK")

    # ----------------------------------
    # 2. Подмена агрегированной суммы
    # ----------------------------------

    changed_cells = [block.copy() for block in cells]

    changed_cells[0][0] += 1

    require_rejection(
        "2. Подмена суммы",
        packets,
        public_keys,
        session_id,
        epoch,
        round_id,
        changed_cells,
    )

    # ----------------------------------
    # 3. Подмена обновления участника
    # ----------------------------------

    damaged = bytearray(packets[0].payload)

    damaged[0] ^= 1

    bad_packets = [
        replace(
            packets[0],
            payload=bytes(damaged),
        ),
        *packets[1:],
    ]

    require_rejection(
        "3. Подмена подписанного обновления",
        bad_packets,
        public_keys,
        session_id,
        epoch,
        round_id,
        cells,
    )

    # ----------------------------------
    # 4. Повтор данных другой эпохи
    # ----------------------------------

    require_rejection(
        "4. Повтор из старой эпохи",
        packets,
        public_keys,
        session_id,
        epoch + 1,
        round_id,
        cells,
    )

    # ----------------------------------
    # 5. Повтор данных другого раунда
    # ----------------------------------

    require_rejection(
        "5. Повтор из старого раунда",
        packets,
        public_keys,
        session_id,
        epoch,
        round_id + 1,
        cells,
    )

    # ----------------------------------
    # 6. Отсутствующий участник
    # ----------------------------------

    require_rejection(
        "6. Пропуск участника",
        packets[:-1],
        public_keys,
        session_id,
        epoch,
        round_id,
        cells,
    )

    # ----------------------------------
    # Стоимость
    # ----------------------------------

    signatures_bytes = sum(len(packet.signature) for packet in packets)

    input_bytes = sum(len(packet.payload) for packet in packets)

    print(f"\nПодписание обновлений: {sign_ms:.3f} мс")

    print(f"Проверка подписей + пересчёт: {verify_ms:.3f} мс")

    print(f"Входные векторы: {input_bytes:,} байт")

    print(f"Подписи: {signatures_bytes} байт")

    print("\nВсе криптографические тесты пройдены.")


if __name__ == "__main__":
    main()
