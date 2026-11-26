from dataclasses import dataclass, replace
from hashlib import sha256, shake_256
from time import perf_counter
import secrets
import struct

import numpy as np
from .crypto_baseline import digest_message, integer_vector, validate_context

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)


# ==========================================
# Параметры
# ==========================================

P = 2**31 - 1

MAX_ABS_UPDATE = 100

# Две независимые случайные проверки
CHECKS = 2

SIZES = (
    2260,
    50_000,
    200_000,
)

WORKER_COUNTS = (4, 16)

REPEATS = 4

DOMAIN = b"EDGE-PROJECTION-v1\x00"


# ==========================================
# Структуры данных
# ==========================================


@dataclass(frozen=True)
class Challenge:
    session: bytes
    epoch: int
    round_id: int
    candidate_hash: bytes
    dimension: int
    seeds: tuple[bytes, ...]


@dataclass(frozen=True)
class Reply:
    worker_id: int
    tags: tuple[int, ...]
    signature: bytes


# ==========================================
# Случайная линейная проекция
# ==========================================


def random_weights(seed, dimension):

    if not isinstance(seed, bytes) or len(seed) != 32 or not 0 < dimension < 2**32:
        raise ValueError("Неверный seed или размер проекции")

    # SHAKE256 разворачивает короткий seed
    # в псевдослучайный вектор.
    #
    # Rejection sampling позволяет
    # избежать смещения при взятии
    # остатка по модулю P.

    length = dimension + 8

    limit = (2**32 // P) * P

    while True:
        stream = shake_256(DOMAIN + seed).digest(4 * length)

        raw = np.frombuffer(
            stream,
            dtype="<u4",
        )

        valid = raw[raw < limit]

        if valid.size >= dimension:
            return valid[:dimension].astype(np.int64) % P

        length += 16


def dot_mod(vector, weights):

    vector = integer_vector(vector, (P - 1) // 2)
    weights = integer_vector(weights, P - 1)
    if vector.shape != weights.shape or np.any(weights < 0):
        raise ValueError("Неверная форма или диапазон весов")

    # Ограничиваем диапазон,
    # чтобы избежать переполнения
    # и неоднозначности по модулю P.
    bound = (P - 1) // 2

    if np.any(vector < -bound) or np.any(vector > bound):
        raise ValueError("Значение вне допустимого диапазона")

    max_abs = int(np.max(np.abs(vector))) if vector.size else 0

    # Короткие блоки исключают
    # переполнение int64 при суммировании.
    chunk = min(
        4096,
        max(
            1,
            2**60
            // max(
                1,
                max_abs * (P - 1),
            ),
        ),
    )

    result = 0

    for start in range(
        0,
        vector.size,
        chunk,
    ):
        part = vector[start : start + chunk] * weights[start : start + chunk]

        result = (result + int(part.sum(dtype=np.int64))) % P

    return result


# ==========================================
# Подписание ответов участников
# ==========================================


def encoded_message(
    challenge,
    worker_id,
    tags,
):

    validate_context(challenge.session, challenge.epoch, challenge.round_id)
    if (
        type(challenge.dimension) is not int
        or not 0 < challenge.dimension < 2**32
        or len(challenge.candidate_hash) != 32
        or len(challenge.seeds) != CHECKS
        or any(not isinstance(s, bytes) or len(s) != 32 for s in challenge.seeds)
        or type(worker_id) is not int
        or not 0 <= worker_id < 2**32
        or len(tags) != CHECKS
        or any(
            isinstance(t, (bool, np.bool_))
            or not isinstance(t, (int, np.integer))
            or not 0 <= t < P
            for t in tags
        )
    ):
        raise ValueError("Неверный формат вызова или ответа")

    header = struct.pack(
        ">III",
        challenge.epoch,
        challenge.round_id,
        challenge.dimension,
    )

    values = b"".join(struct.pack(">I", int(tag)) for tag in tags)

    return (
        DOMAIN
        + challenge.session
        + header
        + challenge.candidate_hash
        + b"".join(challenge.seeds)
        + struct.pack(">I", worker_id)
        + values
    )


class Worker:
    def __init__(self, worker_id, update, *, max_abs_update=None):

        self.worker_id = worker_id

        self.max_abs_update = MAX_ABS_UPDATE if max_abs_update is None else max_abs_update
        self.update = integer_vector(update, self.max_abs_update).copy()

        if np.any(self.update < -self.max_abs_update) or np.any(self.update > self.max_abs_update):
            raise ValueError("Обновление вне допустимого диапазона")

        self.private_key = Ed25519PrivateKey.generate()

        self.public_key = self.private_key.public_key()

    def reply(self, challenge):
        integer_vector(self.update, self.max_abs_update)
        if challenge.dimension != self.update.size:
            raise ValueError("Размер вызова не совпадает с обновлением")
        encoded_message(challenge, self.worker_id, (0,) * CHECKS)

        if challenge.dimension != self.update.size:
            raise ValueError("Неверный размер обновления")

        tags = tuple(
            dot_mod(
                self.update,
                random_weights(
                    seed,
                    challenge.dimension,
                ),
            )
            for seed in challenge.seeds
        )

        signature = self.private_key.sign(
            encoded_message(
                challenge,
                self.worker_id,
                tags,
            )
        )

        return Reply(
            self.worker_id,
            tags,
            signature,
        )


# ==========================================
# Проверяющий
# ==========================================


class Verifier:
    def __init__(
        self,
        public_keys,
        session,
        epoch,
        round_id,
        *,
        max_abs_update=None,
        commitments=None,
    ):

        self.receipts = dict(commitments or {})
        if commitments is not None and set(self.receipts) != set(public_keys):
            raise ValueError("Incomplete commitment roster")
        self.pinned = {}
        self.public_keys = public_keys
        self.session = session
        self.epoch = epoch
        self.round_id = round_id
        validate_context(session, epoch, round_id)
        self.max_abs_update = MAX_ABS_UPDATE if max_abs_update is None else max_abs_update
        if (
            not public_keys
            or self.max_abs_update < 0
            or 2 * len(public_keys) * self.max_abs_update >= P
        ):
            raise ValueError("Диапазон суммы неоднозначен по модулю поля")
        self.consumed = False

        self.candidate = None
        self.challenge = None

    def begin(self, candidate):

        if self.challenge is not None:
            raise ValueError("Кандидат уже зафиксирован")

        # СНАЧАЛА фиксируем сумму.
        # Только ПОСЛЕ этого
        # генерируем случайный вызов.
        self.candidate = integer_vector(
            candidate, len(self.public_keys) * self.max_abs_update
        ).copy()
        self.candidate.flags.writeable = False

        dimension = len(self.candidate)
        for worker_id, (digest, signature) in self.receipts.items():
            msg = digest_message(worker_id, self.session, self.epoch, self.round_id, dimension, digest)
            try:
                self.public_keys[worker_id].verify(signature, msg)
            except InvalidSignature as exc:
                raise ValueError("Invalid commitment for this candidate context") from exc
            self.pinned[worker_id] = digest

        bound = len(self.public_keys) * self.max_abs_update

        if np.any(self.candidate < -bound) or np.any(self.candidate > bound):
            raise ValueError("Сумма вне допустимого диапазона")

        digest = sha256(self.candidate.astype("<i8").tobytes()).digest()

        self.challenge = Challenge(
            session=self.session,
            epoch=self.epoch,
            round_id=self.round_id,
            candidate_hash=digest,
            dimension=dimension,
            seeds=tuple(secrets.token_bytes(32) for _ in range(CHECKS)),
        )

        return self.challenge

    def finish(self, replies, *, openings=None):

        if self.consumed:
            raise ValueError("Повторное использование проверки")
        self.consumed = True

        challenge = self.challenge

        if challenge is None or self.candidate is None:
            raise ValueError("Сумма ещё не зафиксирована")

        openings = {} if openings is None else openings
        if len(replies) + len(openings) != len(self.public_keys):
            raise ValueError("Ответили не все участники")

        totals = np.zeros(
            CHECKS,
            dtype=np.int64,
        )

        seen = set()
        # Only the coordinator supplies openings. Digests must be authenticated
        # by the worker and pinned BEFORE begin() chooses the challenge.
        for worker_id, raw in openings.items():
            if worker_id not in self.public_keys or len(raw) != 8 * challenge.dimension:
                raise ValueError("Invalid opening identity or length")
            if self.pinned.get(worker_id) != sha256(raw).digest():
                raise ValueError("The opening differs from the committed update")
            vector = integer_vector(np.frombuffer(raw, dtype="<i8"), self.max_abs_update)
            for j, seed in enumerate(challenge.seeds):
                weights = random_weights(seed, challenge.dimension)
                totals[j] = (totals[j] + dot_mod(vector, weights)) % P
            seen.add(worker_id)

        for reply in replies:
            worker_id = reply.worker_id

            if worker_id not in self.public_keys or worker_id in seen:
                raise ValueError("Неизвестный или повторный участник")

            if len(reply.tags) != CHECKS:
                raise ValueError("Неверное число проекций")

            seen.add(worker_id)

            try:
                self.public_keys[worker_id].verify(
                    reply.signature,
                    encoded_message(
                        challenge,
                        worker_id,
                        reply.tags,
                    ),
                )

            except InvalidSignature as exc:
                raise ValueError("Неверная подпись") from exc

            for j, value in enumerate(reply.tags):
                if not 0 <= value < P:
                    raise ValueError("Неверная проекция")

                totals[j] = (totals[j] + value) % P

        # Проверяющий использует
        # только агрегированную сумму
        # и подписанные проекции.
        #
        # Исходных векторов участников
        # здесь нет.

        for j, seed in enumerate(challenge.seeds):
            weights = random_weights(
                seed,
                challenge.dimension,
            )

            actual = dot_mod(
                self.candidate,
                weights,
            )

            if actual != totals[j]:
                raise ValueError("Подмена агрегированной суммы")

        return self.candidate.copy()


# ==========================================
# Проверка отклонения ошибок
# ==========================================


def demand_rejection(
    label,
    verifier,
    replies,
):

    try:
        verifier.finish(replies)

    except ValueError as exc:
        print(f"{label}: ОТКЛОНЕНО ({exc})")

    else:
        raise AssertionError(f"{label}: ожидалось отклонение")


# ==========================================
# Один эксперимент
# ==========================================


def benchmark(workers_n, dimension, seed):

    rng = np.random.default_rng(seed)

    updates = rng.integers(
        -100,
        101,
        size=(workers_n, dimension),
        dtype=np.int64,
    )

    workers = [Worker(i, updates[i]) for i in range(workers_n)]

    public_keys = {worker.worker_id: worker.public_key for worker in workers}

    # Истинная сумма вычисляется
    # только в тестовом стенде
    # для имитации честного агрегатора.
    correct = updates.sum(
        axis=0,
        dtype=np.int64,
    )

    session = secrets.token_bytes(16)

    begin_times = []
    reply_times = []
    verify_times = []

    for repeat in range(REPEATS):
        verifier = Verifier(
            public_keys,
            session,
            epoch=2,
            round_id=100 + repeat,
        )

        start = perf_counter()

        challenge = verifier.begin(correct)

        begin_times.append((perf_counter() - start) * 1000)

        start = perf_counter()

        replies = [worker.reply(challenge) for worker in workers]

        reply_times.append((perf_counter() - start) * 1000)

        start = perf_counter()

        reconstructed = verifier.finish(replies)

        verify_times.append((perf_counter() - start) * 1000)

        assert np.array_equal(
            reconstructed,
            correct,
        )

    # Только для учёта размеров:
    # 64 байта подписи и 4 байта
    # на каждый результат проекции.
    reply_bytes = workers_n * (64 + 4 * CHECKS)

    print(f"\nУчастников: {workers_n}, размер: {dimension:,}")

    print(f"  Фиксация и создание вызова: {np.median(begin_times):.3f} мс")

    print(f"  Формирование ответов всех участников: {np.median(reply_times):.3f} мс")

    print(f"  Проверка: {np.median(verify_times):.3f} мс")

    print(f"  Вектор для проверяющего: {correct.nbytes / 1e6:.3f} МБ")

    print(f"  Подписи и проекции: {reply_bytes} байт")

    # Проверки безопасности
    # на самой маленькой конфигурации
    if workers_n == 4 and dimension == 2260:
        print("\nКонтрольные тесты:")

        attacker = correct.copy()
        attacker[0] += 1

        verifier_bad = Verifier(
            public_keys,
            session,
            epoch=2,
            round_id=999,
        )

        challenge_bad = verifier_bad.begin(attacker)

        replies_bad = [worker.reply(challenge_bad) for worker in workers]

        demand_rejection(
            "1. Подмена суммы",
            verifier_bad,
            replies_bad,
        )

        verifier_new = Verifier(
            public_keys,
            session,
            epoch=2,
            round_id=1000,
        )

        challenge_new = verifier_new.begin(correct)

        demand_rejection(
            "2. Повтор старых ответов",
            verifier_new,
            replies_bad,
        )

        # Each negative case needs a fresh one-shot verifier; otherwise rejection
        # could merely be caused by reusing an already consumed verification.
        verifier_new = Verifier(public_keys, session, epoch=2, round_id=1001)
        challenge_new = verifier_new.begin(correct)

        valid_replies = [worker.reply(challenge_new) for worker in workers]

        modified = list(valid_replies)

        modified[0] = replace(
            valid_replies[0],
            tags=(
                valid_replies[0].tags[0] + 1,
                valid_replies[0].tags[1],
            ),
        )

        demand_rejection(
            "3. Подмена подписанной проекции",
            verifier_new,
            modified,
        )

        verifier_new = Verifier(public_keys, session, epoch=2, round_id=1002)
        challenge_new = verifier_new.begin(correct)
        valid_replies = [worker.reply(challenge_new) for worker in workers]

        demand_rejection(
            "4. Отсутствующий участник",
            verifier_new,
            valid_replies[:-1],
        )


def main():

    print(
        "Рандомизированная проверка агрегированной суммы",
        flush=True,
    )

    print(f"Модуль: {P}; независимых проверок: {CHECKS}")

    print("Проверяющий не получает исходные обновления.")

    print("Время передачи по сети не измеряется.")

    print("Ответы участников формируются последовательно в этом тесте.\n")

    for workers_n in WORKER_COUNTS:
        for dimension in SIZES:
            benchmark(
                workers_n,
                dimension,
                seed=2026 + workers_n + dimension,
            )

    print(
        "\nЭксперимент завершён.",
        flush=True,
    )


if __name__ == "__main__":
    main()
