from fractions import Fraction
import math
import unittest

import numpy as np

from adaptive_planning.adaptive_scheduler import Observation
from adaptive_planning.pd_linucb import CreditBudget, PrimalDualLinUCB
from adaptive_planning.pd_linucb_planner import (
    CONTEXT_NAMES,
    PDLinUCBPlanner,
    make_context,
    outcome_reward,
)


class CreditTests(unittest.TestCase):
    def test_decimal_boundary_and_no_overspend(self):
        budget = CreditBudget({"a": 0.1}, 0.3, 10)
        for _ in range(3):
            budget.commit("a")
        self.assertEqual(budget.spent, Fraction(3, 10))
        self.assertEqual(budget.stop_reason(), "budget_exhausted")
        with self.assertRaises(ValueError):
            budget.commit("a")
        self.assertEqual(budget.steps, 3)

    def test_small_charge_cannot_disappear_next_to_huge_budget(self):
        budget = CreditBudget({"tiny": 0.1, "big": 1e16}, 1e16, 3)
        budget.commit("tiny")
        self.assertNotIn("big", budget.feasible_actions())

    def test_zero_cost_remains_feasible_until_horizon(self):
        budget = CreditBudget({"free": 0.0, "paid": 1.0}, 1.0, 3)
        budget.commit("paid")
        self.assertEqual(budget.feasible_actions(), ("free",))
        budget.commit("free")
        budget.commit("free")
        self.assertEqual(budget.stop_reason(), "horizon_exhausted")

    def test_invalid_inputs(self):
        for costs, budget, horizon in (
            ({"a": -1}, 1, 2),
            ({"a": math.nan}, 1, 2),
            ({"a": 1}, math.inf, 2),
            ({"a": 1}, 0, 2),
            ({"a": 1}, 1, True),
            ({}, 1, 2),
        ):
            with (
                self.subTest(costs=costs, budget=budget, horizon=horizon),
                self.assertRaises(ValueError),
            ):
                CreditBudget(costs, budget, horizon)


class EquationTests(unittest.TestCase):
    def test_design_now_reward_later_and_dynamic_dual_target(self):
        model = PrimalDualLinUCB(2, {"a": 2.0, "b": 1.0}, 3.0, 3, eta=0.05)
        self.assertEqual(model.select([1.0, 0.0]), "a")
        np.testing.assert_allclose(model.A_inv[0], np.diag([0.5, 1.0]))
        np.testing.assert_array_equal(model.b, np.zeros((2, 2)))
        self.assertAlmostEqual(model.dual, 0.05)
        self.assertEqual(model.last_decision["target_cost_rate"], 1.0)
        self.assertEqual(model.select([1.0, 0.0]), "b")
        self.assertEqual(model.last_decision["feasible_actions"], ["b"])
        self.assertAlmostEqual(model.last_decision["target_cost_rate"], 0.5)
        self.assertAlmostEqual(model.dual, 0.075)
        model.reward(1, 1.0)
        np.testing.assert_array_equal(model.b[0], [1.0, 0.0])
        self.assertEqual(model.select([1.0, 0.0]), None)
        self.assertEqual(model.ledger.steps, 2)

    def test_independent_inverse_and_reward_reference(self):
        rng = np.random.default_rng(463)
        costs = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 2.0}
        model = PrimalDualLinUCB(5, costs, 50.0, 30, alpha=0.7, ridge=1.3, eta=0.08)
        matrices = {a: 1.3 * np.eye(5) for a in costs}
        vectors = {a: np.zeros(5) for a in costs}
        pending = {}
        for step in range(1, 31):
            if step - 3 in pending:
                action, x, reward = pending[step - 3]
                model.reward(step - 3, reward)
                vectors[action] += x * reward
            x = rng.normal(size=5)
            feasible = model.ledger.feasible_actions()
            if not feasible:
                break
            scores = {
                a: float(np.linalg.solve(matrices[a], vectors[a]) @ x)
                + 0.7 * math.sqrt(float(x @ np.linalg.solve(matrices[a], x)))
                - model.dual * costs[a]
                for a in feasible
            }
            expected = max(feasible, key=scores.get)
            chosen = model.select(x)
            self.assertEqual(chosen, expected)
            matrices[chosen] += np.outer(x, x)
            pending[step] = (chosen, x.copy(), float(step % 3 != 0))
            for i, action in enumerate(costs):
                np.testing.assert_allclose(
                    model.A_inv[i], np.linalg.inv(matrices[action]), atol=1e-12
                )
                np.testing.assert_allclose(model.b[i], vectors[action], atol=1e-12)

    def test_zero_labels_are_observed_and_rewards_are_not_duplicated(self):
        model = PrimalDualLinUCB(1, {"a": 1}, 10, 10)
        model.select([1.0])
        model.reward(1, 0.0)
        self.assertEqual(model.observed.tolist(), [1])
        self.assertEqual(model.pending, {})
        for identity, reward in ((1, 1.0), (99, 0.0), (True, 0.5)):
            with self.assertRaises(ValueError):
                model.reward(identity, reward)
        model.select([1.0])
        before = model.b.copy()
        for reward in (math.nan, math.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                model.reward(2, reward)
            np.testing.assert_array_equal(before, model.b)

    def test_context_is_owned_and_invalid_selection_does_not_charge(self):
        model = PrimalDualLinUCB(2, {"a": 1}, 10, 10)
        for x in ([math.nan, 0], [1.0], [math.inf, 0]):
            with self.assertRaises(ValueError):
                model.select(x)
        self.assertEqual(model.ledger.steps, 0)
        x = np.array([1.0, 2.0])
        model.select(x)
        x[:] = 0
        model.reward(1, 1.0)
        np.testing.assert_array_equal(model.b[0], [1.0, 2.0])

    def test_eta_zero_is_exact_linucb_ablation(self):
        model = PrimalDualLinUCB(1, {"a": 2.0, "b": 1.0}, 20, 10, eta=0.0)
        for step in range(1, 6):
            model.select([1.0])
            model.reward(step, float(step % 2))
            self.assertEqual(model.dual, 0.0)


class AdapterTests(unittest.TestCase):
    obs = Observation(12, 50260, 0.0, 40.0, 0.0, 2.0)

    def test_causal_normalized_context(self):
        context = make_context(self.obs, "R+P", 2)
        self.assertEqual(context.shape, (len(CONTEXT_NAMES),))
        self.assertTrue(np.all((0 <= context) & (context <= 1)))
        self.assertEqual(context[-4:].tolist(), [0.0, 0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            make_context(self.obs, None, -1)

    def test_hidden_pending_labels_do_not_influence_decisions(self):
        planners = [PDLinUCBPlanner(budget=100, horizon=20, feedback_delay=5) for _ in range(2)]
        previous = [None, None]
        for step in range(1, 6):
            for i, planner in enumerate(planners):
                planner.begin_round(self.obs, max(1, step - 2), step)
                action = planner.choose(self.obs, previous[i])
                planner.observe_round(action, bool(i), 0.2, 0.0)
                previous[i] = action
            self.assertEqual(previous[0], previous[1])
            self.assertEqual(planners[0].stats, planners[1].stats)
            self.assertEqual(planners[0].feedback_count(), 0)
        for planner in planners:
            planner.begin_round(self.obs, 4, 6)
        self.assertEqual(planners[0].feedback_count(), 1)
        self.assertFalse(np.array_equal(planners[0].engine.b, planners[1].engine.b))

    def test_due_labels_before_next_choice_and_no_final_flush(self):
        planner = PDLinUCBPlanner(budget=100, horizon=3, feedback_delay=2)
        previous = None
        for step in range(1, 4):
            planner.begin_round(self.obs, step, step)
            action = planner.choose(self.obs, previous)
            planner.observe_round(action, True, 0.2, 0.0)
            previous = action
        self.assertEqual(planner.feedback_count(), 1)
        self.assertEqual(sorted(planner.engine.pending), [2, 3])
        with self.assertRaises(ValueError):
            planner.begin_round(self.obs, 4, 4)

    def test_immediate_feedback_and_invalid_repeated_calls(self):
        planner = PDLinUCBPlanner(budget=10, horizon=10)
        with self.assertRaises(ValueError):
            planner.choose(self.obs, None)
        planner.begin_round(self.obs, 1, 1)
        with self.assertRaises(ValueError):
            planner.begin_round(self.obs, 1, 1)
        action = planner.choose(self.obs, None)
        self.assertEqual(planner.ledger.steps, 1)
        planner.observe_round(action, False, 0.2, 0.0)
        self.assertEqual(planner.feedback_count(), 1)
        self.assertEqual(planner.last_reward, 0.0)
        with self.assertRaises(ValueError):
            planner.observe_round(action, True, 0.2, 0.0)

    def test_optional_throughput_mapping_is_bounded(self):
        self.assertEqual(outcome_reward(True, 0.2, "throughput", 0.1), 0.5)
        self.assertEqual(outcome_reward(True, 0.01, "throughput", 0.1), 1.0)
        self.assertEqual(outcome_reward(False, 0.2, "throughput", 0.1), 0.0)
        self.assertEqual(outcome_reward(True, 0.2, "success", 0.1), 1.0)


if __name__ == "__main__":
    unittest.main()
