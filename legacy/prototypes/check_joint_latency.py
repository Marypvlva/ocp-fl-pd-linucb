
"""Проверка формул задержки и трафика."""

import numpy as np

import joint_planning_experiment as exp


def check_case(
    count,
    vbw,
    available=True,
):

    dimension = 50_260

    sample = (
        60e6,                          # edge bandwidth
        vbw,                           # verification bandwidth
        False,                         # нагрузка
        np.zeros(4, dtype=bool),       # нет отказов
        np.ones(4),                    # нет замедления
        np.zeros((12, 4), dtype=bool), # нет подмен
        available,                     # участники доступны
        0.03,                          # дополнительный RTT
    )

    print(
        f"\nУчастников: {count}; "
        f"канал проверки: "
        f"{vbw / 1e6:.0f} МБ/с"
    )

    for action in exp.ACTIONS:

        copies = (
            2
            if action.startswith("R")
            else 1
        )

        uses_projection = (
            action.endswith("P")
        )

        edge_time, tampered = (
            exp.placement(
                60e6,
                sample[3],
                sample[4],
                sample[5],
                copies,
                count,
                dimension,
            )
        )

        assert edge_time is not None
        assert not tampered

        sum_bytes = dimension * 8

        # Передача исходных обновлений
        # агрегаторам + передача суммы
        # проверяющему.

        expected_bytes = (
            count
            * sum_bytes
            * copies
            + sum_bytes
        )

        if uses_projection:

            expected_bytes += count * (
                64 + 8 + 64
            )

            check_seconds = (
                sum_bytes / vbw
                + exp.CALIB_MS[count][1] / 1000
                + 0.03
            )

        else:

            expected_bytes += count * (
                sum_bytes + 64
            )

            check_seconds = (
                (count + 1)
                * sum_bytes
                / vbw
                + exp.CALIB_MS[count][0] / 1000
            )

        # Проверяем вариант без
        # переключения и вариант
        # с переключением.

        previous_actions = (
            None,
            (
                "R+P"
                if action.startswith("F")
                else "F+B"
            ),
        )

        for previous in previous_actions:

            should_switch = (
                previous is not None
                and previous != action
            )

            full_time = (
                edge_time
                + check_seconds
            )

            if should_switch:

                full_time += (
                    exp.SWITCH_SECONDS
                )

            should_succeed = (
                full_time <= exp.DEADLINE
                and (
                    available
                    or not uses_projection
                )
            )

            if (
                uses_projection
                and not available
            ):

                should_elapsed = (
                    exp.DEADLINE
                )

            else:

                should_elapsed = min(
                    exp.DEADLINE,
                    full_time,
                )

            (
                success,
                elapsed,
                sent,
                switched,
            ) = exp.outcome(
                action,
                sample,
                previous,
                count,
                dimension,
            )

            assert (
                success == should_succeed
            ), (
                f"{action}: "
                f"success={success}, "
                f"ожидалось {should_succeed}"
            )

            assert abs(
                elapsed - should_elapsed
            ) < 1e-9, (
                f"{action}: "
                f"elapsed={elapsed}, "
                f"ожидалось {should_elapsed}"
            )

            assert sent == expected_bytes, (
                f"{action}: "
                f"bytes={sent}, "
                f"ожидалось {expected_bytes}"
            )

            assert (
                switched == should_switch
            )

        # Итоговый статус.
        # Здесь full_time содержит
        # стоимость переключения.

        if (
            uses_projection
            and not available
        ):

            result = (
                "участники недоступны — отказ"
            )

        elif full_time <= exp.DEADLINE:

            result = "в срок"

        else:

            result = "дедлайн превышен"

        print(
            f"  {action}: "
            f"{full_time * 1000:.2f} мс; "
            f"{result}; OK"
        )


def main():

    print(
        "Проверка задержки и трафика "
        "совместного планировщика"
    )

    for count in (
        4,
        12,
        16,
    ):

        check_case(
            count,
            20e6,
        )

        check_case(
            count,
            160e6,
        )

    print(
        "\nПроверка недоступных участников:"
    )

    check_case(
        12,
        160e6,
        available=False,
    )

    print(
        "\nВсе проверки пройдены."
    )


if __name__ == "__main__":
    main()
