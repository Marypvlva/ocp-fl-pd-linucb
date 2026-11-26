"""Causal reactive controls and mixed faults must remain budgeted and auditable."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_planning.adaptive_scheduler import Observation
from adaptive_planning.ocp_analyze import validate
from adaptive_planning.ocp_budget_report import main as report_budget
from adaptive_planning.ocp_compare import main as compare
from adaptive_planning.ocp_experiment import round_fault, validate_config
from adaptive_planning.paths import ROOT
from adaptive_planning.pd_linucb_planner import PDLinUCBPlanner


class ReactiveTests(unittest.TestCase):
    def choose_all(self, rule, budget, flags):
        planner = PDLinUCBPlanner(
            budget=budget,
            horizon=len(flags),
            available_actions=["F+P", "R+P"],
            eta=0,
            selection_rule=rule,
            feedback_delay=2,
        )
        actions = []
        for attempt, (edge, participant) in enumerate(flags, 1):
            obs = Observation(3, 100, 0, 0, 0, 0, edge, participant)
            planner.begin_round(obs, attempt, attempt)
            action = planner.choose(obs, actions[-1] if actions else None)
            actions.append(action)
            planner.observe_round(action, True, 0.1, 0)
            self.assertEqual(planner.last_decision["selection_rule"], rule)
        self.assertEqual(planner.ledger.steps, len(flags))
        self.assertLessEqual(planner.ledger.spent, planner.ledger.budget)
        self.assertEqual(planner.feedback_count(), max(0, len(flags) - 2))
        return actions

    def test_error_type_changes_spending_but_never_starves_later_rounds(self):
        flags = [(False, True), (True, False), (True, False), (True, False)]
        self.assertEqual(self.choose_all("reactive_edge", 6, flags), ["F+P", "R+P", "R+P", "F+P"])
        self.assertEqual(self.choose_all("reactive_any", 6, flags), ["R+P", "R+P", "F+P", "F+P"])
        self.assertEqual(self.choose_all("reactive_edge", 4, flags), ["F+P"] * 4)

    def test_unused_quota_is_spent_at_the_end(self):
        self.assertEqual(
            self.choose_all("reactive_edge", 6, [(False, False)] * 4),
            ["F+P", "F+P", "R+P", "R+P"],
        )

    def test_reactive_policy_rejects_incompatible_configuration(self):
        base = dict(
            budget=6,
            horizon=4,
            eta=0,
            available_actions=["F+P", "R+P"],
            selection_rule="reactive_edge",
        )
        for update in (
            dict(budget=3),
            dict(eta=0.05),
            dict(available_actions=["F+P"]),
            dict(action_schedule=["F+P"] * 4),
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                PDLinUCBPlanner(**(base | update))

    def test_trace_types_and_roster_membership(self):
        config = json.loads((ROOT / "configs/tcp_ocp_paired_pilot.json").read_text())
        config.update(conditions=["mixed_availability"], rounds=3)
        config["fault_traces"] = {
            str(s): ["healthy", "edge", "participant"] for s in config["model_seeds"]
        }
        validate_config(config)
        trace = ["healthy", "edge", "participant"]
        self.assertEqual(
            round_fault("mixed_availability", 2, [0, 3], 6, outages=trace).failed_edges, (0,)
        )
        self.assertEqual(
            round_fault("mixed_availability", 2, [2, 3], 6, outages=trace).failed_edges, ()
        )
        self.assertEqual(
            round_fault("mixed_availability", 3, [2, 3], 6, outages=trace).missing_projection, (2,)
        )
        config["fault_traces"][str(config["model_seeds"][0])][0] = True
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_tcp_mixed_outcomes_replay_and_reactive_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            config = json.loads((ROOT / "configs/tcp_ocp_smoke.json").read_text())
            config.update(
                workers=[3],
                sample_size=3,
                model_seeds=[22],
                rounds=4,
                budget_credits=6,
                stop_at_target=False,
                pd_reward="success",
                conditions=["mixed_availability"],
                fault_traces={"22": ["healthy", "edge", "participant", "edge"]},
            )
            path = temp / "config.json"
            path.write_text(json.dumps(config))
            output = temp / "runs"
            args = ["--config", str(path), "--output", str(output)]
            with patch("builtins.print"):
                compare(
                    args
                    + [
                        "--reactive-controls",
                        "--policies",
                        "reactive_edge",
                        "reactive_any",
                        "R+P",
                        "F+B",
                    ]
                )
                compare(args + ["--resume"])
                report_budget([str(output), "--output", str(temp / "report")])
            report = json.loads((temp / "report/analysis.json").read_text())
            self.assertEqual(len(report["cases"]), 4)
            self.assertTrue(
                all(case["structural_outcome_mismatches"] == 0 for case in report["cases"])
            )
            for policy in ("reactive_edge", "reactive_any", "R+P"):
                _, _, rows = validate(output / policy)
                self.assertEqual(rows[2]["accepted"], 0)
                self.assertTrue(
                    rows[3]["observation"]["projection_missing"] if len(rows) > 3 else True
                )
                self.assertTrue(
                    all(set(row["certificate"]["selected"]) == {0, 1, 2} for row in rows)
                )
            _, _, edge = validate(output / "reactive_edge")
            _, _, any_error = validate(output / "reactive_any")
            self.assertEqual(edge[0]["action"], "F+P")
            self.assertEqual(edge[1]["action"], "F+P")  # current failure cannot affect decision
            self.assertEqual(edge[2]["action"], "R+P")
            self.assertEqual(any_error[2]["action"], "R+P")
            _, _, baseline = validate(output / "F+B")
            self.assertEqual(baseline[2]["accepted"], 1)
            self.assertEqual(baseline[2]["errors"], [])
