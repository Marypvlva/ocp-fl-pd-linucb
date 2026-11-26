"""Scheduled controls must retain exact charging and delayed-label safety."""

import json
import unittest

import numpy as np

from adaptive_planning.ocp_experiment import (
    planner_for,
    observation_for,
    round_fault,
    validate_config,
)
from adaptive_planning.paths import ROOT
from adaptive_planning.pd_linucb import PrimalDualLinUCB


class ScheduleControlTests(unittest.TestCase):
    def test_forced_choice_cannot_bypass_budget_or_consume_invalid_charge(self):
        engine = PrimalDualLinUCB(1, {"cheap": 1, "expensive": 2}, 1, 2)
        before = engine.state()
        with self.assertRaises(ValueError):
            engine.select(np.ones(1), forced_action="expensive")
        self.assertEqual(engine.state(), before)
        self.assertEqual(engine.select(np.ones(1), forced_action="cheap"), "cheap")
        engine.reward(1, 1)
        self.assertIsNone(engine.select(np.ones(1)))
        self.assertEqual(float(engine.ledger.spent), 1)

    def test_schedule_is_immutable_and_delayed_labels_remain_causal(self):
        config = json.loads((ROOT / "configs/tcp_ocp_paired_pilot.json").read_text())
        config.update(
            rounds=3,
            budget_credits=4,
            pd_eta=0,
            pd_feedback_delay=2,
            action_schedule=["F+P", "R+P", "F+P"],
        )
        planner = planner_for(config)
        config["action_schedule"][0] = "R+B"
        obs = observation_for(config, [0, 1, 2], 100, (False, False))
        previous = None
        for attempt, expected in enumerate(("F+P", "R+P", "F+P"), 1):
            planner.begin_round(obs, attempt, attempt)
            action = planner.choose(obs, previous)
            self.assertEqual(action, expected)
            planner.observe_round(action, True, 0.1, 0)
            previous = action
        self.assertEqual(planner.feedback_count(), 1)
        self.assertEqual(float(planner.ledger.spent), 4)
        self.assertEqual(planner.ledger.stop_reason(), "horizon_exhausted")

    def test_unaffordable_schedule_rejected_before_execution(self):
        config = json.loads((ROOT / "configs/tcp_ocp_paired_pilot.json").read_text())
        config.update(rounds=2, budget_credits=2, pd_eta=0, action_schedule=["R+P", "F+P"])
        with self.assertRaises(ValueError):
            planner_for(config)


class ActionAndFaultControlTests(unittest.TestCase):
    def test_restricted_action_set_stops_when_remaining_credit_cannot_buy_an_arm(self):
        config = json.loads((ROOT / "configs/tcp_ocp_paired_pilot.json").read_text())
        config.update(rounds=3, budget_credits=3, available_actions=["R+P"])
        planner = planner_for(config)
        obs = observation_for(config, [0, 1, 2], 100, (False, False))
        planner.begin_round(obs, 1, 1)
        self.assertEqual(planner.choose(obs, None), "R+P")
        planner.observe_round("R+P", True, 0.1, 0)
        self.assertEqual(float(planner.ledger.remaining), 1)
        self.assertEqual(planner.ledger.stop_reason(), "budget_exhausted")
        self.assertEqual(planner.engine.actions, ("R+P",))

    def test_fixed_or_scheduled_mode_cannot_escape_restricted_action_set(self):
        config = json.loads((ROOT / "configs/tcp_ocp_paired_pilot.json").read_text())
        config.update(available_actions=["F+P", "R+P"], pd_eta=0, fixed_action="F+B")
        with self.assertRaises(ValueError):
            validate_config(config)
        del config["fixed_action"]
        config["action_schedule"] = ["F+B"] * config["rounds"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_frozen_fault_trace_is_required_and_only_targets_affected_rosters(self):
        config = json.loads((ROOT / "configs/tcp_ocp_heldout_action_control.json").read_text())
        validate_config(config)
        with self.assertRaises(ValueError):
            round_fault("irregular_targeted_edge_failure", 1, [0, 1], 6)
        self.assertEqual(
            round_fault(
                "irregular_targeted_edge_failure", 1, [0, 3], 6, outages=[True]
            ).failed_edges,
            (0,),
        )
        self.assertEqual(
            round_fault(
                "irregular_targeted_edge_failure", 1, [2, 3], 6, outages=[True]
            ).failed_edges,
            (),
        )
        config["fault_traces"]["126"][0] = 1
        with self.assertRaises(ValueError):
            validate_config(config)
