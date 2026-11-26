"""Regressions for causal references, recovery topology and frozen live plans."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adaptive_planning.protocol_feedback import feedback_from_round
from adaptive_planning.remedy_world import WorldSpec, make_world
from adaptive_planning.attributed_pd_linucb import AttributedPDLinUCBPlanner
from adaptive_planning.adaptive_scheduler import Observation
from deploy.live_study import config_for, require_complete, check_plan

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "causal_bayes", ROOT / "research/study4/bayes_reference.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Study4Tests(unittest.TestCase):
    def test_unknown_projection_does_not_condition_bayes(self):
        ref = module.BayesReference(make_world(WorldSpec(), 410000))
        ref.choose((0, 1))
        before = ref.prior.copy()
        ref.observe((0, 1), "F+P", None, None, None, None)
        np.testing.assert_allclose(ref.belief, before)

    def test_backup_recovery_can_mix_fragments(self):
        m = dict(
            aggregate_started=True,
            aggregate_completed=True,
            compute_started=True,
            compute_completed=True,
            feedback_schema=3,
            verification_attempts=1,
        )
        self.assertFalse(feedback_from_round("R+P", True, ["e0: x", "e3: x"], m).backup_edge_error)
        m["feedback_schema"] = 2
        self.assertTrue(feedback_from_round("R+P", True, ["e0: x", "e3: x"], m).backup_edge_error)

    def test_filter_only_residuals_stay_zero(self):
        p = AttributedPDLinUCBPlanner(
            budget=10,
            horizon=5,
            population=2,
            actions=("F+P", "R+P"),
            attribution="latent",
            reward_model="factored",
            base_context=False,
            residual_learning=False,
            alpha=0.0,
            feedback_delay=0,
        )
        o = Observation(2, 10, 0, 0, 0, 0, False, False)
        for t in range(1, 6):
            p.begin_round(o, t, t)
            a = p.choose(o, None, selected=(0, 1))
            p.observe_round(a, False, 0, 0, primary_edge_error=True, projection_missing=None)
        self.assertTrue(all(np.all(b == 0) for b in p.engine.b.values()))
        self.assertGreater(p.engine.labels["E"], 0)

    def test_empty_study_and_changed_plan_are_rejected(self):
        plan = dict(cells={"S1": {}}, seeds=[1], policies={"AF_L": {}}, horizon=1, order_rng=1)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                require_complete(Path(folder), plan)
        with self.assertRaises(ValueError):
            check_plan(plan)

    def test_world_comes_from_frozen_parameters(self):
        plan = dict(
            cells={"custom": dict(population=6, budget_ratio=1.1)},
            horizon=10,
            policies={"AF_L": dict(kind="pd")},
            delay=2,
            deadline_s=3,
            rpc_timeout_s=2,
        )
        cfg, w = config_for(plan, "custom", 1, "AF_L")
        self.assertEqual(w.budget, 11)
        self.assertEqual(cfg["budget_credits"], 11)


if __name__ == "__main__":
    unittest.main()
