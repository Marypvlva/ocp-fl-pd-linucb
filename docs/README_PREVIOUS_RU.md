# adaptive_planning: OCP-FL + PD-LinUCB

В этапе от 19 ноября исправлены наблюдаемость проверок, учёт фактического
прикладного трафика и аудит AF-планировщиков. Короткий запуск исправленной
факторизованной модели доступен через
`train --config configs/tcp_ocp_factored_smoke.json`.
[Модели, формат журналов и байтовый бюджет](docs/experiments/AF_PROTOCOL_REPAIR.md)
описаны отдельно вместе с отрицательными результатами новых абляций.

Исследовательский стенд распределённого обучения с двумя основами:

- **OCP-FL** проверяемо выбирает участников раунда через commit–reveal,
  подписанный beacon и детерминированную выборку, записанную в EVM-контракт.
- **PD-LinUCB** выбирает способ размещения и проверки обновлений для этого
  состава с учётом известных затрат и отложенных наград.

Затем отдельные worker/edge-процессы обмениваются данными по TCP, проверяют сумму
и подписи, применяют обновление MLP/Digits и записывают хеш модели в контракт.
При отказе клиента состав не меняется. [Архитектура и соответствие статьям](docs/experiments/OCP_PD_FOUNDATION.md).

## Запуск

Python 3.13, проверенная среда macOS arm64. В существующем checkout сохраните `.venv`.
Для нового checkout:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m adaptive_planning doctor
.venv/bin/python -m adaptive_planning test
.venv/bin/python -m adaptive_planning smoke
.venv/bin/python -m adaptive_planning train
```

`train` по умолчанию использует OCP-FL **и** PD-LinUCB. Настройки:
[совместное обучение](configs/tcp_ocp_train.json). Для своей конфигурации:
`train --config /полный/путь/config.json`. Новый запуск создаёт новую папку результатов.

```bash
.venv/bin/python -m adaptive_planning study ocp
.venv/bin/python -m adaptive_planning verify RESULTS_FOLDER
.venv/bin/python -m adaptive_planning report RESULTS_FOLDER --output NEW_REPORT_FOLDER
```

Парный пилот сравнивает PD-LinUCB, LinUCB и четыре фиксированных режима
на одинаковых списках клиентов и сценариях сбоя:

```bash
.venv/bin/python -m adaptive_planning.ocp_compare --output results/ocp_paired_pilot
```

В папке сохраняются журналы, проверяемые сертификаты, сводка расхода и качества,
а также доли принятых вкладов по группам. В конфигурации пилота задан один
seed модели. [Протокол и результаты пилота](docs/experiments/OCP_PD_PAIRED_PILOT.md).
Для продолжения после прерывания используйте `--resume` с тем же
конфигом и папкой. Если внутри осталась папка незавершённого режима,
переместите её в другое место перед продолжением.

Для сравнений с остановкой при первом достижении качества
(`stop_at_target=true`) доступна сводка кредитов и gas с аудитом исходных
журналов и указанием недостигнутого порога.

```bash
.venv/bin/python -m adaptive_planning.ocp_compare_report \
  RESULTS_FOLDER --output NEW_REPORT_FOLDER
```

[Проверка на четырёх новых трассах](docs/experiments/OCP_PD_QUALITY_REPLICATION.md)
продолжает сравнение ресурсов до общего качества модели.

[Смешанные отказы и сильные реактивные контроли](docs/experiments/OCP_PD_MIXED_AVAILABILITY.md)
проверяют, когда резервирование полезно, и где PD-LinUCB уступает простому правилу.
Флаг `--reactive-controls` добавляет `reactive_edge` и `reactive_any` в `ocp_compare`.
Флаг `--roster-controls` включает экспериментальные сравнения, в которых
планировщик получает [состав из сертификата OCP-FL](docs/experiments/OCP_PD_ROSTER_CONTEXT.md).
Для запусков с `stop_at_target=false` используйте отчёт при фиксированном бюджете:

```bash
.venv/bin/python -m adaptive_planning.ocp_budget_report \
  RESULTS_FOLDER --output NEW_REPORT_FOLDER
```

Симуляция абляций запускается отдельно:

```bash
.venv/bin/python -m adaptive_planning.ocp_availability_sim --output NEW_FOLDER
```

`--joint-controls --seed-start 9000` добавляет четыре действия, а
`--shared-model --seed-start 10000` — эксперимент с общими параметрами модели
на отдельной серии. Эти варианты не включены в TCP-планировщик по умолчанию.
Симуляция не измеряет обучение модели, TCP или gas.

## Границы реализации

Стандартный стенд: настоящая локальная EVM EthereumTester/PyEVM и BLS-маяк с
известным тестовым ключом. Это не сеть независимых валидаторов и не защищённый
production beacon. Адаптер настоящего drand проверяет подписанные значения по
закреплённому ключу; реальный тест-вектор включён в проверки. В Solidity beacon
передаёт разрешённый oracle, а независимая BLS-проверка происходит **до обучения**.
При неверном сертификате или подписи TCP-раунд не запускается.

Подписи и проверка суммы не доказывают честность градиента. OCP-FL не заменяет
robust aggregation или защиту приватности. Gas EVM считается отдельно от квоты
PD в условных кредитах. Время всех фаз измеряется, выигрыш из статей не предполагается.

## Структура и предыдущие этапы

`adaptive_planning/` — рабочий пакет; `contracts/` — Solidity, ABI и байткод;
`configs/` — настройки; `tests/` — проверки; `docs/` — протоколы и результаты;
`legacy/` — архивные прототипы. `results/` и `work/` хранятся отдельно от Git.

Прежние исследования доступны явно через `study pd-linucb`, `study availability`,
`study freshness`; `train --engine pd-only` воспроизводит вариант без OCP-FL.
Они служат историческими сравнениями. [Предыдущий PD-пилот](docs/experiments/PD_LINUCB_RESULTS.md)
не подтвердил преимущества PD над LinUCB. Отрицательные результаты сохраняются.

Пересборка контракта при изменении Solidity: `cd contracts`, `npm ci`, `npm run build`.
Для обычного Python-запуска используется проверенный artifact, Node.js не требуется.

[Статус](docs/project/PROJECT_STATUS.md) · [План](docs/project/ROADMAP.md) ·
[Разработка](docs/project/CONTRIBUTING.md) · [Документация](docs/README.md).


## Исследование от 21 ноября 2026 года

Версия v4 содержит сравнение методов с одинаковым латентным фильтром, проверку дефицита двух ресурсов, исправленный байесовский ориентир и воспроизводимый парный режим с 16 клиентами. Полные результаты и ограничения приведены в [отчёте](research/study4/REPORT_RU.md), рабочий текст статьи — в [manuscript_v4.md](research/study4/manuscript_v4.md).

Дополнительный эффект AF-L над общим фильтром зависит от условий. В опорном сравнении он составил +0,70 раунда за 480, а заранее выбранный порог практического превосходства не достигнут. Облачный комплект обновлён для отдельного пилота. Инструкция находится в deploy/yandex/RUNBOOK_RU.md.
