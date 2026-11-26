"""Replay both foundational algorithms and the selected-client training updates."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
import heapq
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from eth_hash.auto import keccak
import numpy as np

from .ocp_chain import verify_certificate
from .ocp_randomness import BeaconSpec, LocalTestBeacon
from .ocp_experiment import (
    make_groups,
    observation_for,
    planner_for,
    primary_edge_error,
    validate_config,
)
from .ocp_roster_planning import roster_features
from .pd_linucb_planner import make_context, outcome_reward
from .tcp_dynamic_experiment import observed_issues
from .tcp_runtime import MAX_UPDATE, flatten_model, make_model
from .training import Q_SCALE
from .protocol_feedback import feedback_from_round
from .ocp_af_audit import RewardEquationAudit
from .telemetry_alignment_analyze import close_tree
from .telemetry_robustness_analyze import read_csv, write_csv


def validate(folder):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    config = meta["config"]
    validate_config(config)
    if meta.get("family") != "ocp_pd_joint" or meta["status"] != "completed":
        raise ValueError("Expected a completed joint OCP-FL/PD-LinUCB run")
    with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt source snapshot")
        for name, digest in meta["source_sha256"].items():
            if sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Source snapshot changed: " + name)
    summary = read_csv(folder / "summary.csv")
    rows = json.loads((folder / "rounds.json").read_text())
    expected = {
        f"w{n}_s{seed}_{condition}"
        for n in config["workers"]
        for seed in config["model_seeds"]
        for condition in config["conditions"]
    }
    if (
        len(summary) != len(expected)
        or len(summary) != meta["planned_runs"]
        or {s["run"] for s in summary} != expected
    ):
        raise ValueError("Missing, duplicate or unplanned run")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run"]].append(row)
    if set(grouped) - expected:
        raise ValueError("Unplanned round")
    for s in summary:
        records = grouped[s["run"]]
        population = int(s["population"])
        seed = int(s["model_seed"])
        condition = s["condition"]
        if f"w{population}_s{seed}_{condition}" != s["run"]:
            raise ValueError("Case changed")
        journal = [
            json.loads(line)
            for line in (folder / f"journal_{s['run']}.jsonl").read_text().splitlines()
        ]
        if journal != records:
            raise ValueError("Round JSON and journal differ")
        if meta.get("bls_cache_policy") == "reset_per_run" and int(s["bls_cache_misses"]) != len(
            records
        ):
            raise ValueError("BLS verification cache was not isolated per run")
        ledger = json.loads((folder / f"chain_{s['run']}.json").read_text())
        spec = BeaconSpec(**ledger["beacon_spec"])
        if spec != LocalTestBeacon().spec:
            raise ValueError("This experiment requires its explicitly labelled local test beacon")
        receipts = ledger["receipts"]
        anchors = ledger["model_events"]
        if (
            len(anchors) != len(records)
            or any(r["status"] != 1 for r in receipts)
            or len({r["tx_hash"] for r in receipts}) != len(receipts)
            or any(a["block_number"] >= b["block_number"] for a, b in zip(receipts, receipts[1:]))
        ):
            raise ValueError("Invalid EVM transaction history")
        if (
            sum(r["gas_used"] for r in receipts) != ledger["gas_used"]
            or int(s["gas_used"]) != ledger["gas_used"]
        ):
            raise ValueError("EVM gas accounting mismatch")
        if sum(r["gas_used"] for r in records) + receipts[0]["gas_used"] != ledger["gas_used"]:
            raise ValueError("Per-cycle gas omitted or duplicated")
        groups, validation, test = make_groups(population, condition)
        model = make_model(seed, config["hidden"], config["learning_rate"])
        initial_loss = model.score(*validation)[0]
        if not np.isclose(initial_loss, float(s["initial_loss"]), rtol=0, atol=1e-12):
            raise ValueError("Model initialization changed")
        planner = planner_for(config)
        previous = None
        issues = (False, False)
        accepted = 0
        reward_sum = 0.0
        first = 0.0 if initial_loss <= config["target_loss"] else None
        attributed = "af_planner" in config
        equation_audit = (
            RewardEquationAudit(planner)
            if attributed and config["af_planner"]["kind"] == "pd"
            else None
        )
        dimension = getattr(planner.engine, "dimension", 0)
        audit_actions = planner.actions if attributed else planner.engine.actions
        A = {a: np.eye(dimension) * config["pd_ridge"] for a in audit_actions}
        b = {a: np.zeros(dimension) for a in A}
        labels = []
        dual = 0.0
        spent = Fraction(0)
        transmitted = 0
        prior_byte_spent = 0
        costs = {a: Fraction(str(v)) for a, v in config["action_costs"].items()}
        budget = Fraction(str(config["budget_credits"]))
        for attempt, row in enumerate(records, 1):
            if (
                row["attempt"] != attempt
                or row["population"] != population
                or row["condition"] != condition
                or row["model_seed"] != seed
            ):
                raise ValueError("Noncontiguous or reassigned round")
            if first is not None and config["stop_at_target"]:
                raise ValueError("Continued after target")
            ticket = row["certificate"]
            if (
                ticket["round"] != attempt
                or ticket["chain_id"] != ledger["chain_id"]
                or ticket["contract"] != ledger["contract"]
                or ticket["sample_size"] != config["sample_size"]
                or len(ticket["clients"]) != population
            ):
                raise ValueError("Foreign, replayed or resized schedule")
            selected = verify_certificate(ticket, spec)
            final = [
                r
                for r in receipts
                if r["operation"] == "finalize" and r["block_number"] == ticket["block_number"]
            ]
            if (
                len(final) != 1
                or final[0]["block_hash"] != ticket["block_hash"]
                or final[0]["timestamp"] < ticket["beacon_time"]
            ):
                raise ValueError("Finalization receipt mismatch")
            expected_missing = [ticket["clients"][0]] if condition == "non_reveal" else []
            if ticket["non_reveals"] != expected_missing:
                raise ValueError("Non-reveal scenario changed")
            obs = observation_for(config, selected, model.nparam, issues)
            if not close_tree(row["observation"], asdict(obs)):
                raise ValueError("Noncausal PD context")
            if attributed:
                feasible = planner.ledger.feasible_actions()
                if equation_audit:
                    equation_audit.release(attempt)
            else:
                while labels and labels[0][0] <= attempt:
                    _, arm, x0, reward = heapq.heappop(labels)
                    b[arm] += x0 * reward
                x = make_context(obs, previous, 0)
                if config.get("roster_context"):
                    x = np.r_[x, roster_features(selected, max(config["workers"]), obs.workers)]
                means = [float(np.linalg.solve(A[a], b[a]) @ x) for a in A]
                radii = [
                    config["pd_alpha"] * np.sqrt(max(0.0, float(x @ np.linalg.solve(A[a], x))))
                    for a in A
                ]
                feasible = [a for a in A if costs[a] <= budget - spent]
                left = config["rounds"] - attempt + 1
                if config.get("fund_horizon"):
                    feasible = [
                        a
                        for a in feasible
                        if costs[a] + (left - 1) * costs["F+P"] <= budget - spent
                    ]
                if not feasible:
                    raise ValueError("Execution after budget exhaustion")
                scores = {a: means[i] + radii[i] - dual * float(costs[a]) for i, a in enumerate(A)}
            planner.begin_round(obs, attempt, attempt)
            action = planner.choose(obs, previous, selected=selected)
            scheduled = config.get("action_schedule")
            selection_valid = (
                action in feasible
                if attributed
                else (
                    action == scheduled[attempt - 1]
                    if scheduled
                    else action in feasible
                    and scores[action] >= max(scores[a] for a in feasible) - 1e-10
                )
            )
            rule = config.get("selection_rule", "ucb")
            if rule in ("reactive_edge", "reactive_any"):
                extra = budget - spent - left * costs["F+P"]
                capacity = int(extra // (costs["R+P"] - costs["F+P"]))
                risk = obs.edge_trouble or (rule == "reactive_any" and obs.projection_missing)
                expected_action = "R+P" if capacity > 0 and (risk or capacity >= left) else "F+P"
                selection_valid = action == expected_action
            elif rule in ("risk_roster", "benefit_roster"):
                selection_valid = costs[action] + (left - 1) * costs["F+P"] <= budget - spent
            if (
                action != row["action"]
                or not selection_valid
                or not close_tree(row["decision"], planner.last_decision)
            ):
                raise ValueError("PD action/decision replay failed")
            if attributed:
                if equation_audit:
                    equation_audit.check(planner.active[3], action, row["decision"])
            else:
                d = row["decision"]
                target = float((budget - spent) / (config["rounds"] - attempt + 1))
                dual = max(0.0, dual + config["pd_eta"] * (float(costs[action]) - target))
                if (
                    not close_tree(d["predicted_reward"], means)
                    or not close_tree(d["confidence_bonus"], radii)
                    or not np.isclose(d["dual_after"], dual)
                    or not np.isclose(d["target_cost_rate"], target)
                ):
                    raise ValueError("Independent PD equation audit failed")
                A[action] += np.outer(x, x)
            spent += costs[action]
            if spent > budget:
                raise ValueError("Budget overspend")
            if row["metrics"]["selected_workers"] != list(selected) or row["metrics"][
                "selected_examples"
            ] != sum(len(groups[i][1]) for i in selected):
                raise ValueError("TCP execution roster differs from OCP selection")
            if row["accepted"] not in (0, 1):
                raise ValueError("Non-binary update outcome")
            if row["accepted"]:
                total = sum(len(groups[i][1]) for i in selected)
                updates = [
                    np.rint(
                        model.gradient(*groups[i]) * (len(groups[i][1]) / total) * Q_SCALE
                    ).astype(np.int64)
                    for i in selected
                ]
                if any(np.any(np.abs(v) > MAX_UPDATE) for v in updates):
                    raise ValueError("Out-of-bound update accepted")
                model.update(np.stack(updates).sum(axis=0, dtype=np.int64))
                accepted += 1
            loss, accuracy = model.score(*validation)
            if not np.isclose(loss, row["validation_loss"], rtol=0, atol=1e-11) or not np.isclose(
                accuracy, row["validation_accuracy"], rtol=0, atol=1e-12
            ):
                raise ValueError("Selected-client model trajectory differs")
            digest = keccak(flatten_model(model)).hex()
            if digest != row["model_hash"]:
                raise ValueError("Model hash/transactional rollback mismatch")
            anchor = anchors[attempt - 1]
            if (
                anchor["round"] != attempt
                or anchor["modelHash"] != digest
                or not any(
                    r["operation"] == "model_hash" and r["block_number"] == anchor["block_number"]
                    for r in receipts
                )
            ):
                raise ValueError("Model anchor missing")
            if first is None and loss <= config["target_loss"]:
                first = row["first_quality_wall_s"]
            if row["first_quality_wall_s"] != first:
                raise ValueError("Incorrect first-passage quality record")
            if row["forged_schedule_blocked"] != int(condition == "forged_schedule"):
                raise ValueError("Forged coordinator proposal not blocked")
            if condition == "worker_dropout" and attempt % 3 == 0 and row["accepted"]:
                raise ValueError("Dropout silently repaired by changing roster")
            reward = outcome_reward(
                bool(row["accepted"]),
                row["service_wall_s"],
                config["pd_reward"],
                config["pd_reward_reference_seconds"],
            )
            if reward != row["bandit_reward"]:
                raise ValueError("Reward changed")
            feedback = {}
            if attributed:
                if (
                    meta.get("protocol_feedback_schema") in (2, 3)
                    or config.get("protocol_feedback_schema") in (2, 3)
                    or row["metrics"].get("feedback_schema") in (2, 3)
                ):
                    feedback = feedback_from_round(
                        action, bool(row["accepted"]), row["errors"], row["metrics"]
                    ).planner_arguments()
                    if not close_tree(row.get("protocol_feedback"), feedback):
                        raise ValueError("Protocol feedback differs from executed stages")
                else:
                    # Historical experiments used a false zero for unqueried M.
                    from .ocp_experiment import backup_edge_error, projection_missing

                    feedback = dict(
                        primary_edge_error=primary_edge_error(row["errors"]),
                        backup_edge_error=backup_edge_error(row["errors"]),
                        projection_missing=projection_missing(row["errors"]),
                        full_update_missing=False,
                    )
                context = planner.active[3].copy()
            else:
                feedback = dict(primary_edge_error=primary_edge_error(row["errors"]))
            planner.observe_round(
                action,
                bool(row["accepted"]),
                row["service_wall_s"],
                row["metrics"]["switch_wall_s"],
                **feedback,
            )
            if equation_audit:
                identity = planner.ledger.steps
                observed = planner.records.get(identity)
                if observed is not None:
                    observed = observed[2]
                else:
                    # Zero-delay feedback has already left the planner's queue.
                    from .attributed_pd_linucb import observable

                    seen = observable(action)
                    raw = dict(
                        E=feedback["primary_edge_error"],
                        S=feedback["backup_edge_error"],
                        M=feedback["projection_missing"],
                        U=feedback["full_update_missing"],
                    )
                    observed = {k: v if k in seen else None for k, v in raw.items()}
                    if "stage_observations" in feedback:
                        observed["stages"] = feedback["stage_observations"]
                equation_audit.record(attempt, context, action, reward, observed)
            if not attributed:
                if config["pd_feedback_delay"] == 0:
                    b[action] += x * reward
                else:
                    heapq.heappush(
                        labels, (attempt + config["pd_feedback_delay"], action, x.copy(), reward)
                    )
            reward_sum += reward
            if not close_tree(row["planner_statistics"], planner.stats):
                raise ValueError("Delayed reward state changed")
            if not attributed:
                for i, arm in enumerate(A):
                    if not np.allclose(
                        planner.engine.A_inv[i], np.linalg.inv(A[arm]), atol=1e-11, rtol=1e-11
                    ) or not np.allclose(planner.engine.b[i], b[arm], atol=1e-12):
                        raise ValueError("Design-now/reward-later matrix mismatch")
            if attributed and planner.secondary is not None:
                if planner.secondary.spent > planner.secondary.budget:
                    raise ValueError("Secondary resource budget exceeded")
            if "app_byte_budget" in config:
                state = row["metrics"].get("app_byte_budget")
                if not isinstance(state, dict) or state["limit"] != config["app_byte_budget"]:
                    raise ValueError("Application byte budget record missing or changed")
                if not 0 <= state["spent"] <= state["spent"] + state["reserved"] <= state["limit"]:
                    raise ValueError("Application byte budget exceeded")
                transmitted += int(row["metrics"]["app_tx_bytes"])
                if transmitted > state["spent"] or state["spent"] < prior_byte_spent:
                    raise ValueError("Application byte accounting mismatch")
                prior_byte_spent = state["spent"]
            previous = action
            issues = observed_issues(SimpleNamespace(errors=row["errors"]))
        if (
            len(records) != int(s["attempts"])
            or accepted != int(s["accepted_updates"])
            or float(spent) != float(s["spent_credits"])
            or not np.isclose(reward_sum, float(s["reward_sum"]))
            or int(s["observed_rewards"]) != planner.feedback_count()
            or int(s["pending_rewards"]) != len(planner.engine.pending)
        ):
            raise ValueError("Summary/budget/future-label mismatch")
        if "app_tx_bytes" in s and int(s["app_tx_bytes"]) != sum(
            int(row["metrics"]["app_tx_bytes"]) for row in records
        ):
            raise ValueError("Application byte summary changed")
        if "app_byte_budget" in config and (
            int(s["app_byte_budget"]) != config["app_byte_budget"]
            or int(s["app_byte_spent"]) != prior_byte_spent
        ):
            raise ValueError("Application byte ledger summary changed")
        byte_exhausted = bool(
            records and records[-1]["metrics"].get("app_byte_budget", {}).get("denied_frames")
        )
        stop = (
            "app_byte_budget_exhausted"
            if byte_exhausted
            else "target_reached"
            if first is not None and config["stop_at_target"]
            else "update_out_of_range"
            if records and records[-1]["metrics"]["update_bound_exceeded"]
            else planner.ledger.stop_reason()
        )
        if (
            stop is None
            or s["stop_reason"] != stop
            or int(s["reached"]) != int(first is not None)
            or first is not None
            and float(s["t_epsilon_wall_s"]) != first
        ):
            raise ValueError("Incorrect termination/quality time")
        if not np.isclose(float(s["final_loss"]), model.score(*validation)[0]) or not np.isclose(
            float(s["test_accuracy"]), model.score(*test)[1]
        ):
            raise ValueError("Final model score changed")
    return meta, summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    meta, summary, rows = validate(args.folder)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "summary.csv", summary)
    result = dict(
        status="passed",
        runs=len(summary),
        rounds=len(rows),
        accepted=sum(r["accepted"] for r in rows),
        forged_schedules_blocked=sum(r["forged_schedule_blocked"] for r in rows),
        non_reveals=sum(len(r["certificate"]["non_reveals"]) for r in rows),
        budget_violations=0,
        model_trajectory_replayed=True,
        ocp_and_pd_independent_audits=True,
        beacon_mode="local_test_bls",
        backend="actual local EVM; no consensus network",
    )
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    lines = [
        "# OCP-FL и PD-LinUCB: совместная основа проекта",
        "",
        "OCP-FL фиксирует проверяемый случайный состав участников. PD-LinUCB выбирает размещение и способ проверки только для этого состава. При отказе клиента состав не подменяется.",
        "",
        f"Запусков: {len(summary)}. Раундов: {len(rows)}. Принято обновлений: {result['accepted']}. Заблокировано подмен расписания: {result['forged_schedules_blocked']}.",
        "",
        "Проверены реальные Solidity-транзакции в локальной EVM, commit/reveal, будущий слот beacon, BLS, Keccak, Fisher–Yates с rejection sampling, матрицы PD, delayed labels, расход кредитов и траектория MLP по выбранным клиентам.",
        "",
        "## Границы результата",
        "",
        "Это исследовательская локальная EVM EthereumTester/PyEVM и BLS-маяк с известным тестовым ключом, без независимых валидаторов или непредсказуемости настоящего drand. Реальный drand проверен отдельным подписанным тест-вектором. Контракт доверяет allowlisted oracle; независимая BLS-проверка происходит перед TCP-обучением. При неверной подписи обучение не начинается. Подписи обновлений подтверждают происхождение и сумму, но не честность градиента. Числа качества, газа в долларах и задержки из статьи не выдаются за результаты этого проекта.",
        "",
        "Кредиты PD и фактически израсходованный gas EVM считаются отдельно. Время train включает все фазы раунда, без обещания скрыть задержку цепочки параллельным выполнением.",
        "",
        "| Условие | Популяция | Seed | Принято/попыток | Итоговый loss | Кредиты | Gas EVM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['condition']} | {s['population']} | {s['model_seed']} | {s['accepted_updates']}/{s['attempts']} | {float(s['final_loss']):.5f} | {s['spent_credits']} | {s['gas_used']} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
