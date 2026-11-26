
import numpy as np
import training_experiment as exp

rng = np.random.default_rng(123)

updates = rng.integers(
    -100,
    101,
    size=(exp.WORKERS, exp.VECTOR_SIZE),
    dtype=np.int64,
)

expected = updates.sum(axis=0)

healthy = np.zeros(exp.NODES, dtype=bool)
broken_site = np.array(
    [True, True, False, False],
    dtype=bool,
)

normal_speed = np.ones(exp.NODES)

no_corruption = np.zeros(
    (exp.BLOCKS, exp.NODES),
    dtype=bool,
)

all_corrupted = np.ones(
    (exp.BLOCKS, exp.NODES),
    dtype=bool,
)


def check(action, failed, corrupted):
    latency, _, reconstructed = exp.aggregate(
        updates,
        action,
        (
            500_000,
            failed,
            normal_speed,
            corrupted,
            False,
        ),
    )

    if reconstructed is not None:
        assert np.array_equal(
            reconstructed,
            expected,
        )

    return latency, reconstructed


# 1. Оба способа работают
# при исправных узлах.
for strategy in (exp.FAST, exp.RESERVE):
    latency, result = check(
        strategy,
        healthy,
        no_corruption,
    )
    assert result is not None

print("1. Исправные узлы: OK")


# 2. Быстрая площадка отказала.
# Без резервирования результата нет,
# с резервированием он восстанавливается.
_, fast_result = check(
    exp.FAST,
    broken_site,
    no_corruption,
)

_, reserve_result = check(
    exp.RESERVE,
    broken_site,
    no_corruption,
)

assert fast_result is None
assert reserve_result is not None

print("2. Отказ площадки: OK")


# 3. Все агрегаторы возвращают
# некорректные ячейки.
_, result = check(
    exp.RESERVE,
    healthy,
    all_corrupted,
)

assert result is None

print("3. Подмена всех ячеек: OK")

print("\nВсе проверки пройдены.")
