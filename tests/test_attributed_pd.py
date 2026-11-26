"""Checks for certificate-attributed, protocol-factored PD-LinUCB and its world."""

import itertools
import unittest

import numpy as np

from adaptive_planning.attributed_pd_linucb import (
    AttributedPDLinUCBPlanner,
    CertificateEvidence,
    CertificateFilter,
    FundedBudget,
    accepted_under,
    implied_labels,
    observable,
)
from adaptive_planning.adaptive_scheduler import Observation
from adaptive_planning.ocp_experiment import remedy_fault
from adaptive_planning.remedy_controls import PlugInRemedyControl, ReactiveRemedyControl
from adaptive_planning.remedy_world import ACTION_COSTS, ACTION_TRAFFIC, WorldSpec, make_world

ALL = ("F+P", "F+B", "R+P", "R+B")


def play(planner, world, delay_check=None):
    previous, action, accepted = (False, False), None, 0
    for t in range(world.spec.horizon):
        members = world.rosters[t]
        obs = Observation(len(members), 4810, 0.0, 0.0, 0.0, 0.0, *previous)
        planner.begin_round(obs, t + 1, t + 1)
        action = planner.choose(obs, action, selected=members)
        E, S, M, U = world.indicators[t]
        ok = world.accepted(t, action)
        planner.observe_round(
            action,
            bool(ok),
            0.0,
            0.0,
            primary_edge_error=E,
            backup_edge_error=S,
            projection_missing=M,
            full_update_missing=U,
        )
        previous = (E, M if action[-1] == "P" else False)
        accepted += ok
        if delay_check:
            delay_check(t, planner)
    return accepted


class ProtocolRuleTests(unittest.TestCase):
    def test_acceptance_rule_truth_table(self):
        for E, S, M, U in itertools.product((False, True), repeat=4):
            self.assertEqual(accepted_under("F+P", E, S, M, U), not E and not M)
            self.assertEqual(accepted_under("F+B", E, S, M, U), not E and not U)
            self.assertEqual(accepted_under("R+P", E, S, M, U), (not E or not S) and not M)
            self.assertEqual(accepted_under("R+B", E, S, M, U), (not E or not S) and not U)

    def test_implied_labels_only_when_all_completions_agree(self):
        labels = implied_labels(ALL, {"E": False, "S": None, "M": False, "U": None})
        self.assertEqual(labels, {"F+P": 1.0, "R+P": 1.0})
        labels = implied_labels(ALL, {"E": True, "S": False, "M": False, "U": None})
        self.assertEqual(labels, {"F+P": 0.0, "F+B": 0.0, "R+P": 1.0})
        labels = implied_labels(ALL, {"E": True, "S": None, "M": None, "U": False})
        self.assertEqual(labels, {"F+P": 0.0, "F+B": 0.0})
        for action in ALL:
            self.assertIn("E", observable(action))

    def test_implied_labels_agree_with_truth_for_every_observation(self):
        for E, S, M, U in itertools.product((False, True), repeat=4):
            for chosen in ALL:
                seen = observable(chosen)
                observed = {k: v for k, v in zip("ESMU", (E, S, M, U))}
                observed = {k: (v if k in seen else None) for k, v in observed.items()}
                for action, label in implied_labels(ALL, observed).items():
                    self.assertEqual(label, float(accepted_under(action, E, S, M, U)))
                self.assertIn(chosen, implied_labels(ALL, observed))


class AttributionTests(unittest.TestCase):
    def test_evidence_is_masked_by_the_current_roster(self):
        evidence = CertificateEvidence(6, decays=(0.5,))
        evidence.round_finished((1, 2), {"E": True, "S": None, "M": False, "U": None})
        features = evidence.features((2, 4)).reshape(4, 1, 6)
        self.assertEqual(features[0, 0, 2], 1.0)
        self.assertEqual(features[0, 0, 1], 0.0)
        self.assertEqual(features[0, 0, 4], 0.0)
        evidence.round_finished((3, 5), {"E": False, "S": None, "M": False, "U": None})
        self.assertAlmostEqual(evidence.features((2,)).reshape(4, 1, 6)[0, 0, 2], 0.5)

    def test_filter_clears_quiet_members_and_raises_failing_ones(self):
        flt = CertificateFilter(4, persistences=(0.9,))
        before = flt.predicted()[0, 0].copy()
        flt.round_finished((0, 1), {"E": True, "S": None, "M": False, "U": None})
        after = flt.belief[0, 0]
        self.assertGreater(after[0], before[0])
        self.assertEqual(after[2], before[2])
        flt.round_finished((0, 3), {"E": False, "S": None, "M": False, "U": None})
        self.assertEqual(flt.belief[0, 0, 0], 0.0)
        self.assertTrue(np.all(np.isfinite(flt.features((0, 2)))))


class BudgetTests(unittest.TestCase):
    def test_funded_budget_always_leaves_cheapest_action_for_the_horizon(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            ledger = FundedBudget(ACTION_COSTS, 25, 20)
            while ledger.steps < ledger.horizon:
                options = ledger.feasible_actions()
                self.assertIn("F+P", options)
                ledger.commit(options[int(rng.integers(len(options)))])
            self.assertLessEqual(ledger.spent, ledger.budget)

    def test_two_budget_planner_respects_both_resources(self):
        world = make_world(WorldSpec(horizon=120, traffic_ratio=1.1, budget_ratio=1.5), 5)
        planner = AttributedPDLinUCBPlanner(
            budget=world.budget,
            horizon=120,
            population=6,
            actions=ALL,
            alpha=0.3,
            feedback_delay=2,
            reward_model="factored",
            base_context=False,
            secondary_costs=ACTION_TRAFFIC,
            secondary_budget=world.traffic_budget,
        )
        accepted = play(planner, world)
        self.assertLessEqual(float(planner.ledger.spent), world.budget)
        self.assertLessEqual(float(planner.secondary.spent), world.traffic_budget + 1e-9)
        self.assertEqual(planner.ledger.steps, 120)
        self.assertLessEqual(accepted, world.oracle2())


class PlannerTests(unittest.TestCase):
    def test_disabled_extensions_reproduce_original_pd_linucb(self):
        from adaptive_planning.pd_linucb_planner import PDLinUCBPlanner

        world = make_world(WorldSpec(horizon=150), 11)
        original = PDLinUCBPlanner(
            budget=world.budget,
            horizon=150,
            costs=ACTION_COSTS,
            alpha=1.0,
            eta=0.05,
            feedback_delay=2,
            available_actions=["F+P", "R+P"],
            roster_population=6,
            fund_horizon=True,
        )
        extended = AttributedPDLinUCBPlanner(
            budget=world.budget,
            horizon=150,
            population=6,
            costs=ACTION_COSTS,
            actions=("F+P", "R+P"),
            alpha=1.0,
            eta=0.05,
            feedback_delay=2,
            attributed=False,
            implied_feedback=False,
        )
        previous, a1, a2 = (False, False), None, None
        for t in range(150):
            members = world.rosters[t]
            obs = Observation(2, 4810, 0.0, 0.0, 0.0, 0.0, *previous)
            for planner in (original, extended):
                planner.begin_round(obs, t + 1, t + 1)
            a1 = original.choose(obs, a1, selected=members)
            a2 = extended.choose(obs, a2, selected=members)
            self.assertEqual(a1, a2)
            E, S, M, U = world.indicators[t]
            ok = bool(world.accepted(t, a1))
            original.observe_round(a1, ok, 0.0, 0.0, primary_edge_error=E)
            extended.observe_round(
                a2, ok, 0.0, 0.0, primary_edge_error=E, backup_edge_error=S, projection_missing=M
            )
            previous = (E, M)
        self.assertAlmostEqual(original.engine.dual, extended.engine.dual)

    def test_factored_engine_learns_only_revealed_indicators_after_delay(self):
        world = make_world(WorldSpec(horizon=60, repairable=(0.4, 0.8)), 3)
        planner = AttributedPDLinUCBPlanner(
            budget=world.budget,
            horizon=60,
            population=6,
            actions=ALL,
            alpha=0.3,
            feedback_delay=3,
            reward_model="factored",
            base_context=False,
        )

        def check(t, p):
            labels = p.engine.labels
            self.assertLessEqual(labels["E"], max(0, t + 1 - 3))
            self.assertLessEqual(labels["S"], labels["E"])

        play(planner, world, check)
        matured_b = int(planner.engine.observed[1] + planner.engine.observed[3])
        self.assertLessEqual(planner.engine.labels["U"], matured_b)
        self.assertEqual(planner.engine.labels["E"], int(planner.engine.observed.sum()))

    def test_controls_complete_horizon_within_budget(self):
        world = make_world(WorldSpec(horizon=100, traffic_ratio=1.2), 9)
        for cls, kw in (
            (PlugInRemedyControl, dict(pacing="quantile")),
            (PlugInRemedyControl, dict(pacing="dual")),
            (ReactiveRemedyControl, {}),
        ):
            control = cls(
                budget=world.budget,
                horizon=100,
                population=6,
                costs=ACTION_COSTS,
                actions=ALL,
                feedback_delay=2,
                secondary_costs=ACTION_TRAFFIC,
                secondary_budget=world.traffic_budget,
                **kw,
            )
            play(control, world)
            self.assertEqual(control.ledger.steps, 100)
            self.assertLessEqual(float(control.ledger.spent), world.budget)
            self.assertLessEqual(float(control.secondary.spent), world.traffic_budget + 1e-9)


class WorldTests(unittest.TestCase):
    def test_live_fault_function_reproduces_the_simulated_world(self):
        spec = WorldSpec(horizon=200, backup_failure=0.3, reassign_every=50, switch_at=120)
        world = make_world(spec, 21)
        config = {"remedy_traces": {"21": world.trace_config()}}
        for t in range(200):
            self.assertEqual(remedy_fault(config, t + 1, world.rosters[t], 21), world.faults[t])

    def test_rosters_are_independent_of_fault_parameters(self):
        a = make_world(WorldSpec(horizon=50), 4)
        b = make_world(WorldSpec(horizon=50, repairable=(0.5, 0.5), proof=(0.3, 0.9)), 4)
        self.assertEqual(a.rosters, b.rosters)

    def test_oracle_bounds_fixed_and_greedy_counts(self):
        world = make_world(WorldSpec(horizon=200), 8)
        base = sum(world.accepted(t, "F+P") for t in range(200))
        self.assertGreaterEqual(world.oracle(), base)
        self.assertEqual(world.oracle2(), world.oracle())


if __name__ == "__main__":
    unittest.main()


class LatentAndForgettingTests(unittest.TestCase):
    def test_latent_filter_clears_quiet_rosters_and_raises_failing_members(self):
        from adaptive_planning.attributed_pd_linucb import LatentFaultFilter

        flt = LatentFaultFilter(6)
        before = flt.probability("E", (0, 1))
        flt.round_finished((0, 1), {"E": True, "S": None, "M": False, "U": None})
        self.assertGreater(flt.probability("E", (0, 2)), before)
        flt.round_finished((0, 3), {"E": False, "S": None, "M": False, "U": None})
        self.assertLess(flt.probability("E", (0, 4)), flt.probability("E", (1, 4)))
        self.assertEqual(flt.features((0, 1)).shape, (flt.dimension,))
        resp = flt.responsibility((1, 5))
        self.assertAlmostEqual(float(resp.sum()), 1.0)

    def test_unobserved_indicator_only_advances_prediction(self):
        from adaptive_planning.attributed_pd_linucb import LatentFaultFilter

        a, b = LatentFaultFilter(4), LatentFaultFilter(4)
        a.round_finished((0, 1), {"E": None, "S": None, "M": None, "U": None})
        for k in ("E", "M"):
            np.testing.assert_allclose(a.state[k]["exposure"], b.state[k]["exposure"])

    def test_latent_planner_completes_within_both_budgets(self):
        world = make_world(WorldSpec(horizon=150, traffic_ratio=1.15, budget_ratio=1.5), 13)
        planner = AttributedPDLinUCBPlanner(
            budget=world.budget, horizon=150, population=6, actions=ALL, alpha=0.05,
            ridge=10.0, feedback_delay=2, reward_model="factored", base_context=False,
            attribution="latent", dual="resolve", secondary_costs=ACTION_TRAFFIC,
            secondary_budget=world.traffic_budget,
        )
        accepted = play(planner, world)
        self.assertEqual(planner.ledger.steps, 150)
        self.assertLessEqual(float(planner.ledger.spent), world.budget)
        self.assertLessEqual(float(planner.secondary.spent), world.traffic_budget + 1e-9)
        self.assertLessEqual(accepted, world.oracle2())
        self.assertEqual(planner.engine.offsets, {"E": 1, "S": 2, "M": 3, "U": 4})

    def test_forgetting_one_reproduces_undiscounted_engine(self):
        world = make_world(WorldSpec(horizon=120), 17)
        runs = []
        for forget in (1.0, 1.0 - 1e-15):
            planner = AttributedPDLinUCBPlanner(
                budget=world.budget, horizon=120, population=6, actions=ALL, alpha=0.3,
                feedback_delay=2, reward_model="factored", base_context=False, forget=forget,
            )
            runs.append(play(planner, world))
        self.assertEqual(runs[0], runs[1])

    def test_forgetting_returns_unrevealed_estimates_to_prior(self):
        from adaptive_planning.attributed_pd_linucb import FactoredPDLinUCB

        engine = FactoredPDLinUCB(3, {"F+P": 1.0, "R+P": 2.0}, 300, 100, forget=0.9)
        x = np.array([1.0, 0.0, 0.0])
        engine.select(x)
        engine.learn(1, {"E": True, "S": True, "M": False, "U": None})
        high = engine.estimate(x)[0]["S"]
        for _ in range(200):
            engine.decay()
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual(engine.estimate(x)[0]["S"], 0.5, places=3)

    def test_drifting_regimes_leave_default_worlds_unchanged(self):
        a = make_world(WorldSpec(horizon=80), 3)
        b = make_world(WorldSpec(horizon=80, backup_regime=(0.5, 0.95, 0.9)), 3)
        self.assertEqual(a.rosters, b.rosters)
        self.assertEqual([i[0] for i in a.indicators], [i[0] for i in b.indicators])
