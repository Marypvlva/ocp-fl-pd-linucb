"""Verify that roster features preserve authority and only use available information."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adaptive_planning.adaptive_scheduler import Observation
from adaptive_planning.ocp_analyze import validate
from adaptive_planning.ocp_compare import main as compare
from adaptive_planning.ocp_experiment import round_fault
from adaptive_planning.ocp_roster_planning import roster_features
from adaptive_planning.paths import ROOT
from adaptive_planning.pd_linucb_planner import PDLinUCBPlanner


class OcpRosterTests(unittest.TestCase):
    def test_features_identify_membership_independently_of_order(self):
        x = roster_features((1, 3), 4, 2)
        np.testing.assert_allclose(x, [-0.5 / 2**0.5, 0.5 / 2**0.5] * 2)
        np.testing.assert_array_equal(x, roster_features((3, 1), 4, 2))
        for invalid in (None, (1, 1), (1,), (1, 4), (True, 3)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                roster_features(invalid, 4, 2)

    def test_context_and_budget_floor(self):
        planner = PDLinUCBPlanner(
            budget=3,
            horizon=3,
            available_actions=("F+P", "R+P"),
            fund_horizon=True,
            roster_population=4,
            feedback_delay=2,
        )
        obs = Observation(2, 100, 0, 0, 0, 0, False, False)
        for attempt in range(1, 4):
            planner.begin_round(obs, attempt, attempt)
            action = planner.choose(obs, None, selected=(1, 3))
            self.assertEqual(action, "F+P")
            self.assertEqual(planner.last_decision["selection_rule"], "fund_horizon")
            np.testing.assert_array_equal(
                planner.last_decision["context"][-4:], roster_features((1, 3), 4, 2)
            )
            planner.observe_round(action, True, 0.1, 0.0)
        self.assertEqual(planner.ledger.spent, 3)
        self.assertEqual(planner.feedback_count(), 1)

    def test_benefit_control_waits_for_labels_and_uses_only_observed_rescues(self):
        planner = PDLinUCBPlanner(
            budget=6,
            horizon=4,
            available_actions=("F+P", "R+P"),
            roster_population=4,
            feedback_delay=2,
            selection_rule="benefit_roster",
            eta=0,
        )
        previous = None
        outcomes = (
            (False, False, True, 0, 0),
            (True, True, True, 0, 0),
            (True, True, False, 1, 0),
            (False, True, False, 2, 1),
        )
        for step, (last_error, success, current_error, errors, rescues) in enumerate(outcomes, 1):
            obs = Observation(2, 100, 0, 0, 0, 0, last_error, False)
            planner.begin_round(obs, step, step)
            action = planner.choose(obs, previous, selected=(1, 3))
            info = planner.last_decision["benefit_estimate"]
            self.assertEqual(info["matured_primary_labels"], errors)
            self.assertEqual(info["matured_conditional_labels"], rescues)
            if step == 1:
                self.assertEqual(action, "F+P")
                with self.assertRaises(ValueError):
                    planner.observe_round(action, success, 0.1, 0.0)
            if step == 2:
                self.assertEqual(action, "R+P")
            planner.observe_round(action, success, 0.1, 0.0, primary_edge_error=current_error)
            previous = action
        self.assertEqual(planner.ledger.steps, 4)
        self.assertEqual(planner.ledger.spent, 6)
        self.assertEqual(planner.feedback_count(), 2)

    def test_shared_outage_disables_both_copies_of_affected_lane(self):
        def faults(selected, kind):
            return round_fault(
                "mixed_availability", 1, selected, 4, outages=[kind], affected_clients=[3]
            ).failed_edges

        self.assertEqual(faults([0, 3], "edge"), (0,))
        self.assertEqual(faults([0, 3], "shared_edge"), (0, 2))
        self.assertEqual(faults([0, 1], "shared_edge"), ())

    def test_tcp_comparison_replays_roster_context_and_budget(self):
        config = json.loads((ROOT / "configs/tcp_ocp_smoke.json").read_text())
        config.update(
            workers=[3],
            sample_size=2,
            model_seeds=[13],
            rounds=4,
            budget_credits=6,
            stop_at_target=False,
            pd_reward="success",
            conditions=["mixed_availability"],
            fault_traces={"13": ["edge", "participant", "shared_edge", "healthy"]},
            affected_clients={"13": [2]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config_path = path / "config.json"
            config_path.write_text(json.dumps(config))
            output = path / "comparison"
            with patch("builtins.print"):
                compare(
                    [
                        "--config",
                        str(config_path),
                        "--output",
                        str(output),
                        "--roster-controls",
                        "--policies",
                        "pd_roster",
                        "risk_roster",
                    ]
                )
            for policy in ("pd_roster", "risk_roster"):
                _, summary, rows = validate(output / policy)
                self.assertEqual(len(rows), 4)
                self.assertLessEqual(float(summary[0]["spent_credits"]), 6)
                for row in rows:
                    selected = row["certificate"]["selected"]
                    self.assertEqual(row["metrics"]["selected_workers"], selected)
                    np.testing.assert_array_equal(
                        row["decision"]["context"][-3:], roster_features(selected, 3, 2)
                    )
                self.assertEqual(rows[1]["accepted"], 0)
                if 2 in rows[2]["certificate"]["selected"]:
                    self.assertEqual(rows[2]["accepted"], 0)
