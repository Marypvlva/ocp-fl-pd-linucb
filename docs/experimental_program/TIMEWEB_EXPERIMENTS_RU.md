# Команды для существующего трёхузлового стенда

Эти команды подготовлены для оператора. Разработчик их на Timeweb не выполнял, к серверам не подключался и конфигурацию ВМ не менял. **Обновление и новые облачные запуски выполняйте только после своего отдельного решения.** Не запускайте всё меню экспериментов автоматически: сначала C0 и пилот информативности. Новые ВМ, новый inventory и новые ключи не нужны.

## 1. Обновление без потери v0.6.1

В браузерной консоли координатора `77.233.212.217`:

```bash
cd /opt/research/ocp-fl-pd-linucb
pgrep -af 'manage.py run|live.py.*--plan|program_suite.py run|adaptive_planning.node' || true
git status --short
df -h /opt/research
```

Если серия работает, дождитесь завершения либо остановите её Ctrl+C в её tmux. Не обновляйте работающие процессы. Не удаляйте RUNNING, базы или failure.json. Если остались процессы ролей, сначала выясните их принадлежность; инструкция не выполняет `pkill` наугад. Изменённые tracked-файлы нужно сохранить отдельно до переключения тега.

На остановленном стенде создайте защищённую локальную копию:

```bash
COLLECTOR_BACKUP="/opt/research/backups/before-v070-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$COLLECTOR_BACKUP"
chmod 700 "$COLLECTOR_BACKUP"
git rev-parse HEAD > "$COLLECTOR_BACKUP/source.txt"
git diff --binary > "$COLLECTOR_BACKUP/local-changes.patch"
tar -czf "$COLLECTOR_BACKUP/previous-results.tgz" \
  results/timeweb-061-natural results/timeweb-061-controlled
tar -czf "$COLLECTOR_BACKUP/private-stand-state.tgz" deploy/timeweb/state
sha256sum "$COLLECTOR_BACKUP/previous-results.tgz" \
  "$COLLECTOR_BACKUP/private-stand-state.tgz" > "$COLLECTOR_BACKUP/SHA256SUMS"
```

`private-stand-state.tgz` содержит секреты стенда. Он остаётся в защищённой папке координатора и не входит в публичную выгрузку. При нехватке места остановитесь, не удаляйте предыдущие данные.

```bash
git fetch origin --tags
git switch --detach v0.7.0
bash deploy/timeweb/bootstrap.sh
.venv/bin/python deploy/timeweb/manage.py deploy
.venv/bin/python deploy/timeweb/manage.py diagnose
```

Deploy использует прежний coordinator SSH key и существующие ВМ, исключает results/state при копировании. Не выполняйте `manage.py init` повторно. Linux dependency lock не изменён. SSH с MacBook не требуется.

Локальная проверка исходного состояния старых данных только для чтения:

```bash
.venv/bin/python deploy/timeweb/audit_budget.py results/timeweb-061-natural
.venv/bin/python deploy/timeweb/audit_budget.py results/timeweb-061-controlled
```

Этот общий CLI может использовать SQLite read-only WAL/SHM для чтения остановленного журнала; он не меняет записи/квоты. Проверка завершённых новых эпизодов использует immutable-режим только для остановленной checkpointed-базы и не создаёт sidecar-файлы. Непустой WAL при таком resume не игнорируется и не удаляется автоматически.

## 2. Рабочая сессия и переменные

```bash
tmux new -s collector070
```

В открывшейся сессии:

```bash
cd /opt/research/ocp-fl-pd-linucb
set -o pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
COLLECTOR_BASE=results/timeweb-061-natural/plan.json
COLLECTOR_KIT=experiments/collector_v3/kit
mkdir -p results/v070

collector_run_suite() {
  .venv/bin/python "$COLLECTOR_KIT/program_suite.py" run "$1" \
    2>&1 | tee -a "$1/console.log"
}
```

Ctrl+B, затем D — отсоединиться; `tmux attach -t collector070` — вернуться. Suite автоматически обновляет диагностику перед блоком, когда предыдущая старше 55 минут. Опыты идут последовательно: один координатор/HTTPS-порт не используются двумя сериями одновременно. При ошибке остановитесь и сохраните диагностику; не переходите к следующему семейству, скрыв отказ.

`plan` не подключается к ВМ и не запускает опыт. Папка назначения должна отсутствовать. План содержит полную конфигурацию, хеши кода, seed, порядок, события, бюджеты и оценку ресурсов. `run` — отдельное явное действие оператора.

## 3. Минимальный пилот перед основной серией

C0, 80 раундов:

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family C0 --phase diagnostic \
  --seeds 901001 901002 --out results/v070/C0
collector_run_suite results/v070/C0
```

Посмотрите complete/failure и аудиты до перехода дальше. Две крайние proof-SLO ячейки E2, 1 440 раундов:

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E2 --phase pilot \
  --seeds 902001 902002 --cells W27-35-46 W80-130-200 \
  --out results/v070/E2-pilot
collector_run_suite results/v070/E2-pilot
```

Одна E3 proof-only смена режима, 720 раундов:

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E3 --phase pilot \
  --seeds 902011 902012 --cells proof_switch --out results/v070/E3-pilot
collector_run_suite results/v070/E3-pilot
```

Эти 2 240 раундов вместе — рекомендуемый небольшой набор, а не подтверждение преимущества. Анализировать после завершения/остановки, вне научного сбора на загруженном координаторе:

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" analyze \
  results/v070/E2-pilot --out results/analysis-v070-E2-pilot
.venv/bin/python deploy/timeweb/audit_budget.py results/v070/E2-pilot
```

Результаты анализа создаются **вне** исходной серии. Существующая папка анализа не перезаписывается: для повторной версии выберите новый путь. Для C0/E3 замените путь suite и путь вывода. Полнота/неуспехи видны в episodes.csv, analysis.json, prior_attempt_records.csv и журналах; отсутствующие исходы не заполняются нулями.

## 4. Меню остальных экспериментов

Каждый блок ниже — отдельный необязательный запуск после соответствующего научного gate. Все политики внутри seed/time-блока получают одинаковые лимиты/события. Сценарии перемешиваются внутри блока, а не исполняются целиком по очереди «сначала весь лёгкий, затем весь трудный».

### E1: шесть бюджетов

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E1 --phase pilot \
  --seeds 902101 902102 --out results/v070/E1
collector_run_suite results/v070/E1
```

Сетка 30/27/24/21/18/15 МБ; 4 320 раундов. Для первого скрининга можно перед созданием плана добавить `--cells B30000000 B24000000 B18000000` — 2 160 раундов. Не менять уже созданный suite.

### E2: вся proof-сетка или отдельный deadline протокола

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E2 --phase pilot \
  --seeds 902111 902112 --out results/v070/E2-full
collector_run_suite results/v070/E2-full
```

2 160 раундов. Отдельная постановка E2-protocol:

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E2-protocol --phase pilot \
  --seeds 902121 902122 --out results/v070/E2-protocol
collector_run_suite results/v070/E2-protocol
```

Также 2 160 раундов; меняется deadline протокола, служебные timeout не уменьшаются. Это не активный SLO финальности EVM.

### E3/E4: нестационарность и неоднородность

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E3 --phase pilot \
  --seeds 902201 902202 --out results/v070/E3
collector_run_suite results/v070/E3

.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E4 --phase pilot \
  --seeds 902211 902212 --out results/v070/E4
collector_run_suite results/v070/E4
```

Соответственно 2 160 и 1 920 раундов. Proof-only service failure, CPU-хеширование и прикладное ожидание не называются естественной сетью. Kernel traffic control и новые аппаратные характеристики ВМ не настраиваются.

### E5: основная исследовательская серия

Начать с одной ячейки и восьми новых блоков, 5 760 раундов:

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E5 --phase exploratory \
  --seeds 903001 903002 903003 903004 903005 903006 903007 903008 \
  --cells booking_windows_cpu --out results/v070/E5-main
collector_run_suite results/v070/E5-main
```

Альтернативная ячейка называется `bytes_windows_outage`. Чтобы спланировать обе, не указывайте `--cells`; это 11 520 раундов, а не прежние 5 760. Выбор ячейки и параметров делается до её новых исходов. Перед E5 следует зафиксировать выбранные по development значения; `--hyperparameters файл.json` меняет общий Joint-набор и наследуемые matched-абляции.

### E6: длина истории

Начать с 250/500, 3 000 раундов:

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E6 --phase pilot \
  --seeds 902301 902302 --cells H250 H500 --out results/v070/E6
collector_run_suite results/v070/E6
```

Полная сетка без `--cells` содержит H120/H250/H500/H1000 и 7 480 раундов. Бюджет масштабируется с H. 1 000 раундов не следует запускать просто ради увеличения выборки.

### E7: чистые абляции

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E7 --phase exploratory \
  --seeds 903101 903102 903103 903104 --out results/v070/E7
collector_run_suite results/v070/E7

.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E7-article --phase exploratory \
  --seeds 903111 903112 903113 903114 --out results/v070/E7-article
collector_run_suite results/v070/E7-article
```

2 400 и 960 раундов. Первый набор — Joint и четыре однофакторные абляции. Второй — буквальное ядро Article PD и то же ядро с η=0 при нулевой начальной цене.

### E8: ограниченный tuning до основных сравнений

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" plan \
  --base "$COLLECTOR_BASE" --family E8 --phase pilot \
  --seeds 902401 902402 --out results/v070/E8
collector_run_suite results/v070/E8
```

2 160 раундов: по три заранее заданные конфигурации Joint, Article PD и LP Mean. Делайте этот этап **до** настройки основной/подтверждающей серии. Не подбирайте параметры по 904xxx. Дополнительная сетка на отдельные seed — новый заранее сохранённый план; глобально оптимальная настройка здесь не обещается.

## 5. Подтверждающий запуск — отдельное решение

Не запускайте его по умолчанию. После анализа development зафиксируйте один сценарий, один сильный comparator, первичную метрику, практический эффект и n по power.json. Шаблон создания одного подтверждающего плана (например, только если выбран `matched_recent`):

```bash
.venv/bin/python "$COLLECTOR_KIT/program.py" plan \
  --base "$COLLECTOR_BASE" \
  --scenario configs/collector_program/confirmation_example.json \
  --phase confirmatory --seeds 904001 904002 904003 \
  --policies joint matched_recent \
  --exclude-plans results/v070/E5-main/booking_windows_cpu/plan.json \
  --out results/v070/confirmation
```

Три seed в этой команде — только минимальный технический пример, **не** обещание мощности. Прежде чем запускать, замените список на заранее рассчитанное число новых 904xxx (потолок 24 в программе стоимости), comparator и гиперпараметры на выбранные до подтверждения. `confirmation_example.json` имеет H=250 и пропорциональные квоты/байты. Это template, не уже утверждённый подтверждающий дизайн.

Сам запуск после отдельного решения оператора:

```bash
.venv/bin/python deploy/timeweb/manage.py diagnose
.venv/bin/python deploy/timeweb/manage.py run --out results/v070/confirmation \
  2>&1 | tee -a results/v070/confirmation/console.log
```

## 6. Ошибка, возобновление и сохранение попыток

После штатной остановки команда `run` пропускает только завершённые эпизоды, прошедшие проверку plan/round hashes/ресурсного аудита. Частично выполненный эпизод по умолчанию блокирует запуск: бюджет не сбрасывается.

Если остался RUNNING после аварии, сначала проверьте процессы. Для v2-плана утилита снимает lock **только когда записанный PID отсутствует на том же хосте**:

```bash
.venv/bin/python "$COLLECTOR_KIT/program.py" unlock results/v070/E5-main
.venv/bin/python "$COLLECTOR_KIT/program.py" unlock results/v070/E5-main/booking_windows_cpu
```

Выполняйте только для реально существующего stale lock; ошибка «файл отсутствует» не требует создания/удаления чего-либо. При живом PID команда откажет. Старый v0.6.1 lock с другим форматом этой командой не снимается.

Явный повтор незавершённого эпизода:

```bash
.venv/bin/python "$COLLECTOR_KIT/program_suite.py" run results/v070/E5-main \
  --retry-incomplete 2>&1 | tee -a results/v070/E5-main/console.log
```

Первая попытка переносится в `attempts/`, её SQLite, отказы и частичные результаты не удаляются. Новая попытка получает новую базу и отдельные служебные журналы. Лимит B относится к **попытке**, не к сумме всех повторов. Анализ пишет прежние раунды/неопределённый исход в `prior_attempt_records.csv`, помечает сравнение последними попытками как условное и возвращает код 2. Такой повтор не становится незаметно независимым подтверждающим успехом.

## 7. Аудит и HTTPS-выгрузка

После завершения выбранного suite:

```bash
.venv/bin/python deploy/timeweb/audit_budget.py results/v070/E5-main
.venv/bin/python deploy/timeweb/export.py --out results/v070/E5-main \
  --cert deploy/timeweb/state/tls.crt --key deploy/timeweb/state/tls.key
```

Открыть TCP 8443 только для текущего IP Mac. Сверить fingerprint сертификата в доверенной консоли, скачать `https://77.233.212.217:8443/results.zip` с логином `research` и напечатанным временным паролем. После скачивания Ctrl+C и закрыть порт. Приватные ключи, SQLite и state не экспортируются. Исходные базы остаются на координаторе. Архив содержит raw-сценарии и suite-план; графики удобнее строить на Mac, вне сбора наблюдений.

## 8. Анализ на Mac без SSH к ВМ

В отдельной локальной копии v0.7.0 установите Python 3.13 и зависимости по README. Для старого Draft при установленном `gh`:

```bash
gh auth switch --user Marypvlva
mkdir -p ../collector-data-061
gh release download experiments-v0.6.1 --repo Marypvlva/ocp-fl-pd-linucb \
  --dir ../collector-data-061
cd ../collector-data-061
shasum -a 256 -c SHA256SUMS
cd -
.venv/bin/python experiments/collector_v3/kit/analysis_program.py unpack \
  ../collector-data-061/natural-v061.zip ../collector-raw-061/natural
.venv/bin/python experiments/collector_v3/kit/analysis_program.py unpack \
  ../collector-data-061/controlled-v061.zip ../collector-raw-061/controlled
.venv/bin/python experiments/collector_v3/kit/analysis_program.py analyze \
  --series Natural=../collector-raw-061/natural \
  --series Controlled=../collector-raw-061/controlled --out ../analysis-061
```

Если `gh` не установлен/не авторизован, скачайте три файла через интерфейс Draft-релиза под аккаунтом владельца и начните с checksum-команды. Не публикуйте пароль GitHub или токены в shell-history/чате.

Для нового suite, скачанного как `E5-main.zip`:

```bash
.venv/bin/python experiments/collector_v3/kit/analysis_program.py unpack \
  ../E5-main.zip ../collector-raw-070/E5-main
.venv/bin/python experiments/collector_v3/kit/program_suite.py analyze \
  ../collector-raw-070/E5-main --out ../analysis-070-E5
```

Вывод: CSV/LaTeX таблицы, JSON предпосылок/пропусков/мощности, PDF/SVG/PNG графики и `figures/availability.json`. Если нужной сетки/метрики нет, соответствующий график пропускается с объяснением. Нет искусственного regret, отсутствующих цен или test accuracy.

Условная реконструкция прежних цен и диагностика стадий:

```bash
.venv/bin/python experiments/collector_v3/kit/replay_program.py \
  ../collector-raw-061/controlled --out ../controller-replay-061-controlled
.venv/bin/python experiments/collector_v3/kit/mechanisms_program.py \
  ../collector-raw-061/controlled --reconstruction ../controller-replay-061-controlled \
  --out ../mechanisms-061-controlled
```

`replay` проверяет точное совпадение исходных действий по причинной истории, не оценивает награду невыбранных действий. Цены такой реконструкции явно отделены от прямых измерений новых серий.
