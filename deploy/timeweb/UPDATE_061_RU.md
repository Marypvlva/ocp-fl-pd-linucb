# Обновление трёхузлового стенда до v0.6.1

Все команды выполняются в браузерной консоли координатора. Существующие результаты v0.6.0 сохраняются. Новая серия получает новый каталог и новый план с теми же научными параметрами. Не обновляйте код работающих процессов и не смешивайте версии в одном эпизоде.

## 1. Дождаться завершения или штатно остановить текущую серию

Если серия продолжает писать результаты, дайте ей завершиться либо остановите её Ctrl+C в той tmux-сессии, где она запущена. При остановке сохраняются уже записанные раунды, SQLite и незавершённые эпизоды. Перед обновлением проверьте

```bash
cd /opt/research/ocp-fl-pd-linucb
pgrep -af 'manage.py run|live.py.*--plan|adaptive_planning.node' || true
git status --short
```

После остановки связанных процессов список должен быть пуст. При неожиданных оставшихся процессах сначала выясните, к какой серии они относятся. Не удаляйте RUNNING, failure.json или budget.sqlite для обхода защиты. При изменённых tracked-файлах сначала сохраните и перенесите свои правки, Git не должен их перезаписывать при переключении тега.

## 2. Сделать локальную резервную копию

На остановленном стенде

```bash
BACKUP="/opt/research/backups/collector-v060-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP"
chmod 700 "$BACKUP"
git rev-parse HEAD > "$BACKUP/source-commit.txt"
git diff --binary > "$BACKUP/local-changes.patch"
tar -czf "$BACKUP/results-and-state.tgz" results deploy/timeweb/state
sha256sum "$BACKUP/results-and-state.tgz" > "$BACKUP/SHA256SUMS"
echo "$BACKUP"
```

Копия state содержит приватные ключи стенда, она остаётся в защищённом каталоге координатора. Её нельзя загружать в публичный репозиторий или публичную выгрузку. SQLite копируется после остановки писателей вместе с WAL/SHM внутри results.

## 3. Получить исправленный код и развернуть его на тех же ВМ

```bash
git fetch origin --tags
git switch --detach v0.6.1
bash deploy/timeweb/bootstrap.sh
.venv/bin/python deploy/timeweb/manage.py deploy
.venv/bin/python deploy/timeweb/manage.py diagnose
```

Inventory, placement, SSH-ключи, TLS-сертификат и адреса трёх ВМ остаются в state. Deploy копирует код на уже указанные в inventory узлы и исключает results и state. Новые платные ресурсы не создаются. SSH с Mac не требуется. Bootstrap использует прежний проверенный Linux lock, изменения зависимостей отсутствуют.

## 4. Проверить предыдущие данные без перезаписи

Замените путь на фактический каталог серии v0.6.0

```bash
OLD=results/ИМЯ_ПРЕДЫДУЩЕЙ_СЕРИИ
.venv/bin/python deploy/timeweb/audit_budget.py "$OLD" > "$BACKUP/budget-audit-v060.json"
cat "$BACKUP/budget-audit-v060.json"
```

Аудит открывает SQLite только для чтения. Он проверяет диапазоны и сохранение лимита, отдельно показывает незавершённые брони. Успех этой проверки не заменяет полный аудит экспериментальных журналов. Предыдущую summary можно пересчитать прежней версией на сохранённой копии. Сами исходные раунды и планы сохраняйте без правок. BrokenPipeError и отсутствие failure.json сами по себе не определяют достоверность результата.

## 5. Создать новый план с прежними параметрами

```bash
NEW=results/collector-061-repeat
.venv/bin/python deploy/timeweb/clone_plan.py "$OLD/plan.json" "$NEW"
cat "$NEW/plan.json"
```

Копируются 120 раундов, восемь сидов, семь политик, порядок политик, исходная калибровка, раскладка трёх ВМ, окна, квоты и режим сети из вашего исходного плана. Меняются версия исходников, контрольные суммы, время создания и ссылка на родительский план. Старый каталог не изменяется. Новый каталог должен отсутствовать до запуска команды. База новой серии имеет транзакционные счётчики v0.6.1. Не возобновляйте её запись старой версией программы.

Для короткой предварительной проверки создайте отдельный smoke-план через manage.py plan с теми же окнами из исходного плана. Его результаты не включаются в повтор основной серии.

## 6. Запустить повтор и проверить журналы

```bash
tmux new -s collector-061
cd /opt/research/ocp-fl-pd-linucb
set -o pipefail
.venv/bin/python deploy/timeweb/manage.py diagnose
.venv/bin/python deploy/timeweb/manage.py run --out results/collector-061-repeat \
  2>&1 | tee results/collector-061-repeat/console.log
.venv/bin/python deploy/timeweb/summarize.py results/collector-061-repeat
.venv/bin/python deploy/timeweb/audit_budget.py results/collector-061-repeat
.venv/bin/python deploy/timeweb/control_report.py results/collector-061-repeat
```

Ctrl+B, затем D — отсоединиться от tmux. `tmux attach -t collector-061` — вернуться. Приложение публикует control_health, а подробные события сохраняются в файлах

- `<seed>_<policy>/control-server.jsonl` с peer, owner, round, frame_id, этапом ответа, длительностью транзакции/ожидания и нагрузкой;
- `control-client-coordinator.jsonl` и `control-clients/*.jsonl` с повторами, ошибками, ожиданием пула и истечением дедлайна;
- `control-log-fetch.json` с результатом получения клиентских журналов с worker/gateway ВМ;
- `ledger-audit.json` с независимым пересчётом журнала кадров;
- `round-*.json` со всеми раундами, включая отказы, клиентскими счётчиками и состоянием сервиса.

`response_lost` означает обнаруженную ошибку записи HTTPS-ответа. `operation_applied=true` показывает, что операция уже вернулась из SQLite или источника разрешений. `response_written` подтверждает завершение записи в сокет, получение ответа приложением клиента из него не следует. По frame_id можно сопоставить повтор, отмену и settlement. При недоступности сервиса неизвестные брони остаются занятыми.

Если часть клиентских журналов не скопировалась, список ошибок остаётся в control-log-fetch.json. Это следует учитывать в отчёте о полноте наблюдений. Успешные и неуспешные раунды сохраняются и учитываются как прежде. v0.6.0 и v0.6.1 являются разными версиями реализации, их исходы нельзя объединять без явного указания версии.

HTTPS-выгрузка результатов выполняется по [HTTPS_RESULTS_RU.md](HTTPS_RESULTS_RU.md). Новые JSONL входят в экспорт, приватные ключи не входят.

## 7. Дополнительный тест службы

На свободном координаторе, вне сбора научных результатов, можно выполнить

```bash
systemd-run --scope -p MemoryMax=2G -p CPUQuota=100% \
  "$PWD/.venv/bin/python" "$PWD/deploy/timeweb/stress_control.py" \
  --out results/service-load-061 --frames 60480 --owners 9 --concurrency 18 --one-cpu
```

Это нагрузка на реальные локальные HTTPS-запросы и учёт тестовых кредитов. Она не запускает обучение и не имитирует межхостовую сеть. Не выполняйте её одновременно с основной серией. Результаты нагрузки и полные события записываются в отдельный каталог. Она помогает проверить устойчивость службы на одном CPU, но не доказывает отсутствие сетевых проблем на реальных ВМ.
