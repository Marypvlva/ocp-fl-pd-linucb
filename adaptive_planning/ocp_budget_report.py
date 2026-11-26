"""Audit final quality and accepted updates at a precommitted budget/horizon."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from .ocp_compare_report import audit_comparison
from .ocp_experiment import configured_fault
from .tcp_closed_loop_pilot import append_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads((args.folder / "base_config.json").read_text())
    if config.get("stop_at_target") is not False:
        parser.error("Budget reports require stop_at_target=false")
    if args.output.exists():
        parser.error("Report output must be a new directory")
    cases = audit_comparison(args.folder, require_target=False)
    grouped = defaultdict(list)
    for case in cases:
        rows = json.loads((args.folder / case["policy"] / "rounds.json").read_text())
        records = [r for r in rows if r["run"] == case["run"]]
        case["round_gas_used"] = sum(r["gas_used"] for r in records)
        case["setup_gas_used"] = case["gas_used"] - case["round_gas_used"]
        case["reserve_rounds"] = sum(r["action"] == "R+P" for r in records)
        case["reserve_on_missing_participant"] = 0
        case["accepted_reserve_on_edge_fault"] = 0
        case["structural_outcome_mismatches"] = 0
        for row in records:
            if row["condition"] != "mixed_availability" or row["action"] not in ("F+P", "R+P"):
                continue
            fault = configured_fault(
                config,
                row["condition"],
                row["attempt"],
                row["certificate"]["selected"],
                row["population"],
                row["model_seed"],
            )
            reserve = row["action"] == "R+P"
            case["reserve_on_missing_participant"] += int(
                reserve and bool(fault.missing_projection)
            )
            case["accepted_reserve_on_edge_fault"] += int(
                reserve and bool(fault.failed_edges) and row["accepted"]
            )
            ideal = not fault.missing_projection and (
                not fault.failed_edges or reserve and 2 not in fault.failed_edges
            )
            case["structural_outcome_mismatches"] += int(bool(row["accepted"]) != ideal)
        grouped[case["policy"]].append(case)
    metrics = (
        "accepted_updates",
        "attempts",
        "spent_credits",
        "gas_used",
        "round_gas_used",
        "final_loss",
        "reserve_on_missing_participant",
        "accepted_reserve_on_edge_fault",
    )
    summaries = [
        dict(
            policy=policy,
            runs=len(records),
            **{f"{metric}_mean": mean(r[metric] for r in records) for metric in metrics},
        )
        for policy, records in grouped.items()
    ]
    result = dict(
        primary="accepted updates at stop",
        cases=cases,
        summaries=summaries,
        interpretation="Exploratory paired runs. Final validation loss is independent of the binary reward. Total gas includes deployment; round gas is also reported. No speed or significance claim.",
    )
    args.output.mkdir(parents=True)
    append_csv(args.output / "cases.csv", cases)
    append_csv(args.output / "summary.csv", summaries)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Обновления и качество при фиксированном бюджете",
        "",
        f"Бюджет {config['budget_credits']} кредитов, максимум {config['rounds']} раундов. "
        "Остановка по качеству отключена. Все журналы прошли аудит, списки OCP сопоставлены.",
        "",
        "| Политика | Seed | Принято / попыток | Кредиты | Validation loss | Gas, млн | Резерв при недоступном участнике |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in cases:
        lines.append(
            f"| {c['policy']} | {c['model_seed']} | {c['accepted_updates']} / {c['attempts']} | "
            f"{c['spent_credits']:.0f} | {c['final_loss']:.4f} | {c['gas_used'] / 1e6:.2f} | "
            f"{c['reserve_on_missing_participant']} |"
        )
    lines += [
        "",
        "Тип инъекции используется только при анализе, после завершения решения. "
        "Планировщик получает ошибки предыдущего раунда. Счётчик полезного резерва "
        "означает принятый раунд с инъекцией отказа основного агрегатора; это условная "
        "интерпретация данного стенда. Отдельно сохранены расхождения с идеальной моделью отказа.",
        "",
        "Gas в таблице включает развёртывание контракта, все попытки и запись модели. "
        "В CSV отдельно сохранены round_gas_used и setup_gas_used. "
        "Порядок последовательный. Эти запуски не устанавливают ранжирование по времени. "
        "Малое число seed не позволяет утверждать статистическое превосходство.",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Audited {len(cases)} fixed-budget cases: {args.output}")


if __name__ == "__main__":
    main()
