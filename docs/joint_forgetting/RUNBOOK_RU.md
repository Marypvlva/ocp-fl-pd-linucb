# Воспроизведение аудита и использование v0.7.1-rc.1

Это предварительная исследовательская версия. **Новый Joint не объявлен победителем и не заменяет default.** Ниже команды оператора; разработчик не подключался к Timeweb и не выполнял облачные серии. Новые платные ресурсы не нужны. Старые results, базы, черновики GitHub и теги сохраняются.

## 1. Локально: исходные архивы и аудит

В клоне репозитория с Python 3.13 и установленными зависимостями. `gh` должен быть авторизован в аккаунте с доступом к Draft; токен в текст команд не вставлять.

```bash
cd /путь/к/ocp-fl-pd-linucb
git fetch origin --tags
git switch --detach v0.7.1-rc.1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
J071_ANALYSIS="$PWD/work/joint071-reproduction"
J071_KIT=experiments/collector_v3/kit
mkdir -p "$J071_ANALYSIS/archives" "$J071_ANALYSIS/local"
gh release download experiments-v0.7.0 --repo Marypvlva/ocp-fl-pd-linucb \
  --dir "$J071_ANALYSIS/archives"
(cd "$J071_ANALYSIS/archives" && shasum -a 256 -c SHA256SUMS-v070)
.venv/bin/python "$J071_KIT/analysis_program.py" unpack \
  "$J071_ANALYSIS/archives/E1-v070.zip" "$J071_ANALYSIS/raw/E1-v070"
.venv/bin/python "$J071_KIT/analysis_program.py" unpack \
  "$J071_ANALYSIS/archives/E5-booking-pilot-v070.zip" "$J071_ANALYSIS/raw/E5-booking-pilot-v070"
.venv/bin/python "$J071_KIT/analysis_program.py" unpack \
  "$J071_ANALYSIS/archives/E3-pilot-v070.zip" "$J071_ANALYSIS/raw/E3-pilot-v070"
.venv/bin/python "$J071_KIT/joint_audit.py" "$J071_ANALYSIS/raw/E1-v070" --out "$J071_ANALYSIS/E1-audit"
.venv/bin/python "$J071_KIT/joint_audit.py" "$J071_ANALYSIS/raw/E5-booking-pilot-v070" --out "$J071_ANALYSIS/E5-audit"
.venv/bin/python "$J071_KIT/joint_audit.py" "$J071_ANALYSIS/raw/E3-pilot-v070" --out "$J071_ANALYSIS/E3-audit"
.venv/bin/python "$J071_KIT/joint_shadow.py" "$J071_ANALYSIS/raw/E5-booking-pilot-v070" --out "$J071_ANALYSIS/E5-shadow"
```

На Linux также можно использовать `sha256sum -c`. Внешние суммы должны совпасть все три. `unpack` дополнительно проверяет **каждый** внутренний файл. Аудит проверяет подписи, расходы, roster и точность причинного восстановления; при несоответствии останавливается. Вывод создаётся в новых каталогах; повторный запуск в существующую папку запрещён. Черновик не нужно публиковать для анализа. Если установленный Python называется иначе, используйте полный путь к вашей среде, а не системный Python Ubuntu 3.12.

## 2. Локально: честное сравнение пяти семей

```bash
.venv/bin/python "$J071_KIT/joint_local.py" prepare --out "$J071_ANALYSIS/local/manifest.json"
.venv/bin/python "$J071_KIT/joint_local.py" tune "$J071_ANALYSIS/local/manifest.json" --out "$J071_ANALYSIS/local/tuning"
.venv/bin/python "$J071_KIT/joint_local.py" validate "$J071_ANALYSIS/local/manifest.json" \
  --selection "$J071_ANALYSIS/local/tuning/selection.json" --out "$J071_ANALYSIS/local/validation"
.venv/bin/python "$J071_KIT/joint_statistics.py" "$J071_ANALYSIS" --out "$J071_ANALYSIS/statistics"
.venv/bin/python "$J071_KIT/joint_figures.py" "$J071_ANALYSIS" --out "$J071_ANALYSIS/figures"
.venv/bin/python -m unittest discover -s tests -p test_joint_memory.py -v
.venv/bin/python project.py test
```

На машине разработчика validation занял около 37 секунд; tuning, изображения и полный набор тестов требуют ещё несколько минут. Это **локальные** проверки, не научная облачная серия. Результаты фиксированной проверки находятся в `research/joint_v071/evidence/`; не перезаписывайте их собственным запуском. Не подбирайте γ/W/α по validation; изменение методики требует нового manifest, нового validation-набора и явного раскрытия причин. Данные всех старых серий v0.7.0 используются только как development.

## 3. Опциональное обновление существующего стенда

В браузерной консоли координатора, после остановки/завершения всех предыдущих серий:

```bash
cd /opt/research/ocp-fl-pd-linucb
pgrep -af 'manage.py run|live.py.*--plan|program_suite.py run|adaptive_planning.node' || true
git status --short
df -h /opt/research
```

При работающей серии дождитесь её завершения. Не обновляйте процессы под нагрузкой и не удаляйте RUNNING/failure/SQLite. При изменённых tracked-файлах сначала сохраните их; `git switch` не должен затереть локальную работу.

Защищённая копия состояния и контрольные суммы **без удаления результатов**:

```bash
J071_BACKUP="/opt/research/backups/before-v071rc1-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$J071_BACKUP"
chmod 700 "$J071_BACKUP"
git rev-parse HEAD > "$J071_BACKUP/source.txt"
git diff --binary > "$J071_BACKUP/local-changes.patch"
tar -czf "$J071_BACKUP/private-stand-state.tgz" deploy/timeweb/state
find results -type f -print0 | sort -z | xargs -0 sha256sum > "$J071_BACKUP/results.sha256"
```

Private-stand-state содержит ключи: оставить только на координаторе. Существующие результаты остаются на месте; при необходимости полной резервной копии сначала проверьте доступное место и сделайте отдельную копию. Не включайте state/SQLite в публичные assets.

```bash
git fetch origin --tags
git switch --detach v0.7.1-rc.1
bash deploy/timeweb/bootstrap.sh
.venv/bin/python deploy/timeweb/manage.py deploy
.venv/bin/python deploy/timeweb/manage.py diagnose
sha256sum -c "$J071_BACKUP/results.sha256" > "$J071_BACKUP/results-check.txt"
```

Bootstrap использует прежний Linux hash-lock; версии зависимостей не менялись. Deploy управляет существующими worker-ВМ с координатора, исключает results/state, не создаёт ВМ. Не запускайте `init` повторно. SSH с MacBook не требуется. Проверка hashes относится к остановленным старым результатам; новый bootstrap пишет собственные новые журналы. При несовпадении не запускать науку, сохранить диагностику.

## 4. Минимальный технический пилот — только по решению оператора

```bash
tmux new -s joint071
```

В tmux:

```bash
cd /opt/research/ocp-fl-pd-linucb
set -o pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
J071_BASE=results/timeweb-061-natural/plan.json
J071_KIT=experiments/collector_v3/kit
J071_CFG=research/joint_v071/configs
mkdir -p results/v071rc1
.venv/bin/python "$J071_KIT/program.py" plan --base "$J071_BASE" \
  --scenario "$J071_CFG/technical.json" --phase diagnostic --seeds 901701 \
  --policies joint joint_discounted joint_window matched_recent article_pd \
  --policy-overrides "$J071_CFG/frozen-primary.json" --out results/v071rc1/technical
.venv/bin/python deploy/timeweb/manage.py diagnose
.venv/bin/python deploy/timeweb/manage.py run --out results/v071rc1/technical \
  2>&1 | tee -a results/v071rc1/technical/console.log
.venv/bin/python deploy/timeweb/audit_budget.py results/v071rc1/technical
.venv/bin/python "$J071_KIT/analysis_program.py" analyze \
  --series technical=results/v071rc1/technical --out results/analysis-v071rc1-technical
```

30 раундов =5 политик ×6; это sanity-check, не проверка преимущества. Диагностика действительна час. Запуски только последовательно, один общий порт. При ошибке остановиться; не исключать неудачные раунды и не продолжать с новой историей скрыто. Для resume/retry действует прежняя инструкция: complete проверяется по хешам; незавершённый episode повторяется только явно с сохранением attempts. Не возобновлять старые научные планы кодом новой версии.

## 5. Следующий информативный пилот, не масштабная confirmation

**Не выполнять автоматически вслед за technical.** Сначала убедиться в отсутствии регрессий и отдельно решить, оправдывает ли научная цель новый пилот. Два контрольных режима нужны вместе: stationary и switch; отрицательный stationary-эффект не исключается из отчёта.

Планы (планирование не подключается к ВМ):

```bash
.venv/bin/python "$J071_KIT/program.py" plan --base "$J071_BASE" \
  --scenario "$J071_CFG/stationary.json" --phase pilot --seeds 902711 902712 \
  --policies joint joint_discounted joint_window matched_recent article_pd \
  --policy-overrides "$J071_CFG/frozen-primary.json" --out results/v071rc1/stationary
.venv/bin/python "$J071_KIT/program.py" plan --base "$J071_BASE" \
  --scenario "$J071_CFG/switch.json" --phase pilot --seeds 902721 902722 \
  --policies joint joint_discounted joint_window matched_recent article_pd \
  --policy-overrides "$J071_CFG/frozen-primary.json" --out results/v071rc1/switch
```

Отдельные последовательные запуски с диагностикой:

```bash
.venv/bin/python deploy/timeweb/manage.py diagnose
.venv/bin/python deploy/timeweb/manage.py run --out results/v071rc1/stationary \
  2>&1 | tee -a results/v071rc1/stationary/console.log
.venv/bin/python deploy/timeweb/audit_budget.py results/v071rc1/stationary
.venv/bin/python deploy/timeweb/manage.py diagnose
.venv/bin/python deploy/timeweb/manage.py run --out results/v071rc1/switch \
  2>&1 | tee -a results/v071rc1/switch/console.log
.venv/bin/python deploy/timeweb/audit_budget.py results/v071rc1/switch
.venv/bin/python "$J071_KIT/analysis_program.py" analyze \
  --series stationary=results/v071rc1/stationary --series switch=results/v071rc1/switch \
  --out results/analysis-v071rc1-pilot
```

Гипотезы: H1 — memory уменьшает post-switch prequential path Brier; H2 — это увеличивает accepted при одинаковых квотах; H3 — выигрыш имеет цену в stationary; H4 — поздняя обратная связь/разреженность могут устранить выигрыш; H5 — quota exhaustion и цены объясняют отдельную часть потерь. Primary исход для решения о дальнейшем исследовании — seed-level accepted difference по **всему** эпизоду; post-switch принятия, Brier, weighted age/ESS, bonus, цены, feasible set, остатки M/C/байтов, полный round/cycle time — secondary mechanism metrics. Все неуспехи остаются в знаменателе. Recovery без принятия — censored, не ноль и не пропуск.

Для **изолированного механистического** сравнения (другой пилот, по необходимости) используйте неизменённые α=.15/η=.2 и `mechanisms.json` вместо `frozen-primary.json`, список `joint joint_discounted joint_discounted_legacy_bonus joint_window joint_eligible joint_discounted_eligible joint_window_eligible`, новые seeds 902731/902732 и новую output-папку. Здесь память/bonus/censoring разделены; не выбирать лучший из этих вариантов на confirmation без отдельной development-фазы. Эти семь методов при двух seeds дают 1680 раундов.

Объём: каждый основной pilot =1200 раундов, ≈38–75 минут и ≈1.9–3.8 VM-часа суммарно на трёх уже имеющихся ВМ; technical ≈2–5 минут. Основа оценки — реально измеренный средний E5 cycle 1.878 с/раунд, без гарантии сохранения скорости, плюс запас до ×2 и стартовые расходы. Mechanism pilot ≈53–106 минут, ≈2.6–5.3 VM-часа. Это не новая тарификация: денежная стоимость равна VM-часам × вашей фактической цене за ВМ-час; работающие ВМ тарифицируются также вне эксперимента. Требования те же: 3×1vCPU/2ГБ, вычисления/хранилище прежние; ориентировочно сотни МБ файлов на pilot с запасом под журналы и SQLite. Размер проверить `df -h` до старта.

Подтверждающая серия **пока не назначена**: текущие локальные интервалы включают ноль, historical anchor отрицателен. Для неё нужно заранее зафиксировать practically relevant effect/мощность, новый независимый набор time/seed-blocks 904xxx, исключить все development/pilot планы через `--exclude-plans`, определить primary family и поправку множественных сравнений. Не использовать старые E5 seeds или validation для повторной настройки.

## 6. Выгрузка

После остановки/завершения использовать штатный безопасный экспорт/HTTPS: [HTTPS_RESULTS_RU.md](../../deploy/timeweb/HTTPS_RESULTS_RU.md). Новый output-путь экспортируется отдельно; исходные data/SQLite остаются на координаторе. Не размещать state, private-stand-state или секреты в GitHub. В статье указывать exact tag/commit, frozen plan, seed-блоки, физическое совмещение ролей, controlled CPU и local EVM отдельно от реальной сетевой передачи.
