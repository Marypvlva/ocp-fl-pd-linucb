"""Data-dependent publication figures; unavailable measurements are never invented."""

from collections import Counter, defaultdict
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(rows, episodes, effects, plans, output):
    from analysis_program import table

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    folder = output / "figures"
    folder.mkdir()
    availability = []
    policies = sorted({r["policy"] for r in rows})
    series = sorted(plans)
    colors = {p: plt.get_cmap("tab20")(i % 20) for i, p in enumerate(policies)}

    def absent(name, reason):
        availability.append(dict(figure=name, created=False, reason=reason))

    def save(fig, name, question):
        fig.tight_layout()
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(folder / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        availability.append(dict(figure=name, created=True, question=question))

    def panels(columns=1):
        return plt.subplots(len(series), columns, figsize=(7.2, 3 * len(series)), squeeze=False)

    def curves(ax, subset, key, rolling=None):
        for policy in policies:
            blocks = defaultdict(list)
            for r in subset:
                if r["policy"] == policy and r.get(key) is not None:
                    blocks[r["seed"]].append(r)
            values = defaultdict(list)
            for block in blocks.values():
                block.sort(key=lambda r: r["round"])
                for j, r in enumerate(block):
                    value = (
                        r[key]
                        if rolling is None
                        else np.mean([q[key] for q in block[max(0, j - rolling + 1) : j + 1]])
                    )
                    values[r["round"]].append(value)
            if values:
                x = sorted(values)
                ax.plot(x, [np.mean(values[i]) for i in x], label=policy, color=colors[policy])
        ax.set_xlabel("Раунд")
        ax.grid(alpha=0.2)

    summary = []
    for label in series:
        for policy in policies:
            selected = [r for r in rows if r["series"] == label and r["policy"] == policy]
            if not selected:
                continue
            ep = [r for r in episodes if r["series"] == label and r["policy"] == policy]
            accepted = sum(r["accepted"] for r in selected)
            measured = sum(r["total_round_s"] for r in selected)
            summary.append(
                dict(
                    series=label,
                    policy=policy,
                    observed_rounds=len(selected),
                    completed_blocks=sum(r["complete"] for r in ep),
                    accepted=accepted,
                    success_observed=accepted / len(selected),
                    budget_bytes=selected[0]["budget_bytes"],
                    horizon=selected[0]["horizon"],
                    mean_accepted_per_planned_episode=float(
                        np.mean([r["accepted_observed"] for r in ep])
                    ),
                    mean_bytes_per_episode=float(np.mean([r["spent_bytes"] for r in ep])),
                    bytes_per_accepted=sum(r["round_bytes"] for r in selected) / accepted
                    if accepted
                    else None,
                    accepted_per_measured_minute=accepted / measured * 60 if measured else None,
                    p50_round_s=float(np.quantile([r["total_round_s"] for r in selected], 0.5)),
                    p95_round_s=float(np.quantile([r["total_round_s"] for r in selected], 0.95)),
                )
            )
    table(output / "policy_summary.csv", summary)
    same_horizon = len({r["horizon"] for r in rows}) == 1
    if len({r["budget_bytes"] for r in summary}) > 1 and same_horizon:
        fig, ax = plt.subplots(figsize=(7.2, 4))
        for policy in policies:
            subset = sorted(
                [r for r in summary if r["policy"] == policy], key=lambda r: r["budget_bytes"]
            )
            ax.plot(
                [r["budget_bytes"] / 1e6 for r in subset],
                [r["mean_accepted_per_planned_episode"] for r in subset],
                ".-",
                label=policy,
                color=colors[policy],
            )
        ax.set(xlabel="Бюджет, десятичные МБ", ylabel="Принятые обновления / эпизод")
        ax.legend(fontsize=7)
        save(
            fig,
            "01_budget",
            "Как меняется число принятых обновлений при изменении бюджета? Сопоставлять только одинаковые остальные условия.",
        )
    else:
        absent(
            "01_budget",
            "Нужны несколько бюджетов при одинаковом горизонте; H-сетку нельзя выдавать за эффект бюджета",
        )
    if rows:
        fig, axes = panels()
        for i, label in enumerate(series):
            ax = axes[i, 0]
            for policy in policies:
                group = defaultdict(list)
                for r in rows:
                    if r["series"] == label and r["policy"] == policy:
                        group[r["window_ms"]].append(r["accepted"])
                xs = sorted(group)
                if xs:
                    ax.plot(
                        xs,
                        [np.mean(group[x]) for x in xs],
                        ".-",
                        label=policy,
                        color=colors[policy],
                    )
            ax.set(
                title=label,
                xlabel="Окно доказательств, мс",
                ylabel="Доля успехов в наблюдённом классе",
            )
            ax.legend(fontsize=7, ncol=2)
        save(
            fig,
            "02_windows",
            "Есть ли потолок успешности в каждом классе окна? Описательная условная частота, не отдельный рандомизированный dose-response.",
        )
    else:
        absent("02_windows", "Нет исходных раундов")
    valid = [r for r in summary if r["bytes_per_accepted"] is not None]
    if valid:
        fig, axes_pareto = plt.subplots(1, len(series), figsize=(7.2, 4), squeeze=False)
        pareto = []
        for r in valid:
            ax = axes_pareto[0, series.index(r["series"])]
            dominated = any(
                q["bytes_per_accepted"] <= r["bytes_per_accepted"]
                and q["p95_round_s"] <= r["p95_round_s"]
                and q["success_observed"] >= r["success_observed"]
                and (
                    q["bytes_per_accepted"] < r["bytes_per_accepted"]
                    or q["p95_round_s"] < r["p95_round_s"]
                    or q["success_observed"] > r["success_observed"]
                )
                for q in valid
                if q["series"] == r["series"]
            )
            pareto.append(dict(series=r["series"], policy=r["policy"], nondominated=not dominated))
            ax.scatter(
                r["bytes_per_accepted"] / 1000,
                r["p95_round_s"],
                color=colors[r["policy"]],
                s=30 + 100 * r["success_observed"],
                marker="x" if dominated else "o",
                label=r["policy"],
            )
        for label, ax in zip(series, axes_pareto[0], strict=True):
            ax.set(
                title=label, xlabel="Прикладные кБ / принятое обновление", ylabel="P95 раунда, с"
            )
            ax.legend(fontsize=6, ncol=2)
        table(output / "pareto.csv", pareto)
        save(
            fig,
            "03_pareto",
            "Какие измеренные варианты доминируются по расходу, P95 и успешности? Описательный фронт без утверждения статистического доминирования.",
        )
    else:
        absent("03_pareto", "Нет принятых обновлений")
    for name, key, rolling, ylabel in [
        ("04_cumulative", "accepted_cumulative", None, "Накопленная награда: принято обновлений"),
        ("05_learning", "accepted", 20, "Скользящая успешность, окно 20 раундов"),
        ("resource_spending", "spent_bytes", None, "Накопленные прикладные байты"),
    ]:
        if not rows:
            absent(name, "Нет раундов")
            continue
        fig, axes = panels()
        for i, label in enumerate(series):
            curves(axes[i, 0], [r for r in rows if r["series"] == label], key, rolling)
            axes[i, 0].set(title=label, ylabel=ylabel)
            axes[i, 0].legend(fontsize=7, ncol=2)
        save(
            fig,
            name,
            "Как развивается показатель по раундам? Линии — средние блоков, отсутствие полос не означает нулевую неопределённость.",
        )
    if any(r["dual"] is not None for r in rows):
        fig, axes = panels(2)
        for i, label in enumerate(series):
            curves(axes[i, 0], [r for r in rows if r["series"] == label], "dual")
            axes[i, 0].set(title=label, ylabel="Эффективная цена дублирования")
            axes[i, 0].legend(fontsize=6, ncol=2)
            selected = [
                r
                for r in rows
                if r["series"] == label
                and r["policy"] == "joint"
                and r["gateway_duals"] is not None
            ]
            for g in range(4):
                by = defaultdict(list)
                for r in selected:
                    by[r["round"]].append(r["gateway_duals"][0][g])
                x = sorted(by)
                axes[i, 1].plot(x, [np.mean(by[t]) for t in x], label=f"g{g}")
            axes[i, 1].set(xlabel="Раунд", ylabel="Цены шлюзов Joint")
            axes[i, 1].legend(fontsize=7)
        save(
            fig,
            "06_duals",
            "Активируются ли цены именно квот бронирования? Байтовой цены в этом алгоритме нет.",
        )
    else:
        absent(
            "06_duals",
            "Исторический v0.6.1 не записывал цены; восстановление требует причинного replay, а не выдуманных значений",
        )
    if rows:
        fig, axes = panels()
        actions = sorted({r["gateway_action"] for r in rows})
        for i, label in enumerate(series):
            bottom = np.zeros(len(policies))
            for action in actions:
                values = []
                for policy in policies:
                    count = Counter(
                        r["gateway_action"]
                        for r in rows
                        if r["series"] == label and r["policy"] == policy
                    )
                    values.append(count[action] / max(1, sum(count.values())))
                axes[i, 0].bar(policies, values, bottom=bottom, label=action)
                bottom += values
            axes[i, 0].tick_params(axis="x", rotation=30)
            axes[i, 0].set(title=label, ylabel="Доля наблюдённых действий")
            axes[i, 0].legend(fontsize=7, ncol=5, title="Шлюзы")
        save(
            fig,
            "07_actions",
            "Различаются ли маршруты политик? Парное совпадение по seed/раунду отдельно в action_overlap.csv.",
        )
    else:
        absent("07_actions", "Нет действий")
    recovery = []
    aligned = []
    for label, plan in plans.items():
        events = plan.get("program", {}).get("scenario", {}).get("events", [])
        starts = sorted({event["start"] for event in events})
        for policy in policies:
            for seed in plan["seeds"]:
                block = sorted(
                    [
                        r
                        for r in rows
                        if r["series"] == label and r["policy"] == policy and r["seed"] == seed
                    ],
                    key=lambda r: r["round"],
                )
                for event in events:
                    after = [r for r in block if r["round"] > event["end"]]
                    success = next((r for r in after if r["accepted"]), None)
                    recovery.append(
                        dict(
                            series=label,
                            policy=policy,
                            seed=seed,
                            event_end=event["end"],
                            rounds_to_next_accepted=success["round"] - event["end"]
                            if success
                            else None,
                            censored=success is None,
                            followup_rounds=len(after),
                        )
                    )
                for start in starts:
                    for r in block:
                        if -20 <= r["round"] - start <= 40:
                            aligned.append(dict(r, relative_round=r["round"] - start))
    table(output / "recovery.csv", recovery)
    if aligned:
        fig, ax = plt.subplots(figsize=(7.2, 4))
        for policy in policies:
            group = defaultdict(list)
            for r in aligned:
                if r["policy"] == policy:
                    group[r["relative_round"]].append(r["accepted"])
            x = sorted(group)
            if x:
                ax.plot(x, [np.mean(group[t]) for t in x], label=policy, color=colors[policy])
        ax.axvline(0, color="black", linestyle="--")
        ax.legend(fontsize=7)
        ax.set(
            xlabel="Раунды относительно начала зарегистрированного воздействия",
            ylabel="Доля успехов",
        )
        save(
            fig,
            "08_adaptation",
            "Как меняется успех после смены режима? Объединение зарегистрированных событий описательное; recovery.csv сохраняет цензурирование.",
        )
    else:
        absent(
            "08_adaptation",
            "Нет зарегистрированных смен режима с известными границами; периодические отказы v0.6.1 сами по себе не эксперимент на скорость обучения",
        )
    if rows:
        fig, axes = panels(2)
        for i, label in enumerate(series):
            values = []
            names = []
            for policy in policies:
                x = sorted(
                    r["total_round_s"]
                    for r in rows
                    if r["series"] == label and r["policy"] == policy
                )
                if not x:
                    continue
                values.append(x)
                names.append(policy)
                axes[i, 1].plot(
                    x, np.arange(1, len(x) + 1) / len(x), label=policy, color=colors[policy]
                )
            axes[i, 0].boxplot(values, tick_labels=names, showfliers=True)
            axes[i, 0].tick_params(axis="x", rotation=30)
            axes[i, 0].set(title=label, ylabel="Полный протокольный раунд, с")
            axes[i, 1].set(xlabel="Полный протокольный раунд, с", ylabel="ECDF всех исходов")
            axes[i, 1].legend(fontsize=7, ncol=2)
        save(
            fig,
            "09_latency",
            "Где расположены хвосты задержек, включая неуспешные раунды? ECDF не трактует раунды как независимые реплики.",
        )
    else:
        absent("09_latency", "Нет измерений времени")

    def forest(subset, name):
        valid = [r for r in subset if r["ci_t"] is not None]
        if not valid:
            absent(name, "Нет хотя бы двух полных парных блоков")
            return
        fig, ax = plt.subplots(figsize=(7.2, max(3, 0.33 * len(valid))))
        for i, r in enumerate(valid):
            low, high = r["ci_t"]
            family_low, family_high = r["ci_familywise_t"]
            ax.plot([family_low, family_high], [i, i], color="gray", alpha=0.5)
            ax.errorbar(
                r["mean"],
                i,
                xerr=[[max(0, r["mean"] - low)], [max(0, high - r["mean"])]],
                fmt="o",
                color="navy",
            )
        ax.set_yticks(
            range(len(valid)), [r["series"] + "/" + r["control"] for r in valid], fontsize=7
        )
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel(
            "Полная политика − контроль, обновления / эпизод\nТёмный: точечный t-интервал; серый: Bonferroni familywise t-интервал"
        )
        save(
            fig, name, "Каковы абсолютные парные эффекты и неопределённость целых seed/time блоков?"
        )

    forest(effects, "10_forest")
    forest(
        [
            r
            for r in effects
            if r["control"]
            in (
                "no_exploration",
                "no_dual",
                "no_site_price",
                "no_roster",
                "matched_recent",
                "article_no_dual",
            )
        ],
        "12_ablations",
    )
    budgets = sorted({r["budget_bytes"] for r in rows})
    windows = sorted({r["window_ms"] for r in rows})
    if len(budgets) > 1 and len(windows) > 1 and same_horizon:
        fig, axes = plt.subplots(
            1, max(1, len(policies)), figsize=(max(7.2, len(policies) * 2), 3), squeeze=False
        )
        for j, policy in enumerate(policies):
            data = np.full((len(budgets), len(windows)), np.nan)
            for i, budget in enumerate(budgets):
                for k, window in enumerate(windows):
                    sample = [
                        r["accepted"]
                        for r in rows
                        if r["policy"] == policy
                        and r["budget_bytes"] == budget
                        and r["window_ms"] == window
                    ]
                    if sample:
                        data[i, k] = np.mean(sample)
            ax = axes[0, j]
            im = ax.imshow(np.ma.masked_invalid(data), vmin=0, vmax=1, aspect="auto")
            ax.set(title=policy)
            ax.set_xticks(range(len(windows)), windows, rotation=45)
            ax.set_yticks(range(len(budgets)), [b / 1e6 for b in budgets])
            ax.set(xlabel="Окно, мс", ylabel="Бюджет, МБ")
            fig.colorbar(im, ax=ax, fraction=0.045)
        save(
            fig,
            "11_joint_constraints",
            "Как выглядит наблюдённая совместная поверхность? Пустые клетки не интерполируются; разные воздействия нельзя смешивать в причинный вывод.",
        )
    else:
        absent(
            "11_joint_constraints", "Нужна сетка из нескольких реально измеренных бюджетов и окон"
        )
    if any(r["validation_loss"] is not None for r in rows):
        fig, axes = panels(2)
        for i, label in enumerate(series):
            subset = [r for r in rows if r["series"] == label]
            for j, (key, ylabel) in enumerate(
                [
                    ("validation_loss", "Validation loss"),
                    ("validation_accuracy", "Validation accuracy"),
                ]
            ):
                curves(axes[i, j], subset, key)
                axes[i, j].set(title=label, ylabel=ylabel)
                axes[i, j].legend(fontsize=7, ncol=2)
        save(
            fig,
            "13_model_quality",
            "Переходят ли принятые обновления в качество глобальной модели? Это фиксированный validation split; test accuracy отсутствует.",
        )
    else:
        absent("13_model_quality", "Качество модели не записано")
    (folder / "availability.json").write_text(
        json.dumps(availability, ensure_ascii=False, indent=2) + "\n"
    )
