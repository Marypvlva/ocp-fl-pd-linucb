import ast
from dataclasses import replace
import importlib.util
import unittest

import numpy as np

from adaptive_planning.adaptive_scheduler import Observation
from adaptive_planning.availability_analyze import best_fixed, contrasts
from adaptive_planning.availability_experiment import make_trace
from adaptive_planning.availability_scheduler import AvailabilityPlanner
from adaptive_planning.paths import ROOT
from adaptive_planning.tcp_runtime import make_model
from adaptive_planning import training


class AvailabilityTests(unittest.TestCase):
    def test_memory_then_probe_and_only_one_feedback(self):
        planner = AvailabilityPlanner(1, 0.025)
        base = Observation(12, 50260, 0.0, 0.0, 40.0, 2.0)
        actions = []
        for attempt in range(1, 7):
            obs = replace(base, edge_trouble=attempt == 2, projection_missing=attempt == 2)
            action = planner.choose(obs, actions[-1] if actions else None)
            actions.append(action)
            if attempt == 1:
                with self.assertRaises(ValueError):
                    planner.choose(obs, action)
            planner.observe(obs, action, True, 0.2, 0.0)
            with self.assertRaises(ValueError):
                planner.observe(obs, action, True, 0.2, 0.0)
        self.assertEqual(actions, ["F+P", "R+B", "R+B", "R+B", "R+B", "F+P"])
        self.assertEqual(planner.feedback_count(), 6)

    def test_trace_prefix_healthy_control_and_four_regimes(self):
        short = make_trace(42, 8, 12, "persistent", 2, 2)
        long = make_trace(42, 80, 12, "persistent", 2, 2)
        self.assertEqual(short, long[:8])
        self.assertEqual(len({s.regime for s in short}), 4)
        for step in short:
            self.assertEqual(
                bool(step.fault.failed_edges), step.regime in ("edge_outage", "combined_outage")
            )
            self.assertEqual(
                bool(step.fault.missing_projection),
                step.regime in ("projection_outage", "combined_outage"),
            )
        self.assertTrue(
            all(
                not s.fault.failed_edges and not s.fault.missing_projection
                for s in make_trace(42, 20, 12, "healthy", 2, 2)
            )
        )

    def test_best_fixed_requires_all_four_and_honest_censoring(self):
        rows = [
            dict(policy=p, reached="1", t_epsilon_wall_s=str(t), observed_wall_s=str(t + 0.1))
            for p, t in zip(("F+B", "F+P", "R+B", "R+P"), (4, 2, 3, 5))
        ]
        self.assertEqual(best_fixed(rows)["policy"], "F+P")
        rows[0].update(reached="0", t_epsilon_wall_s="", observed_wall_s="1")
        self.assertIsNone(best_fixed(rows))
        rows[0]["observed_wall_s"] = "3"
        self.assertEqual(best_fixed(rows)["policy"], "F+P")
        with self.assertRaises(ValueError):
            best_fixed(rows[:-1])

    def test_primary_really_uses_fastest_fixed_in_each_pair(self):
        records = []
        for repeat in range(2):
            for policy, time in (
                ("F+B", 5),
                ("F+P", 8),
                ("R+B", 3),
                ("R+P", 6),
                ("availability", 2),
                ("network_rule", 4),
                ("network_freshness", 4),
            ):
                records.append(
                    dict(
                        workers="12",
                        repeat=str(repeat),
                        condition="persistent_delayed",
                        policy=policy,
                        reached="1",
                        t_epsilon_wall_s=str(time),
                        observed_wall_s=str(time + 0.1),
                        model_seed=str(16 + repeat),
                        learning_rate=".5",
                    )
                )
            for policy in ("availability", "F+P"):
                records.append(
                    dict(
                        workers="12",
                        repeat=str(repeat),
                        condition="healthy",
                        policy=policy,
                        reached="1",
                        t_epsilon_wall_s="2",
                        observed_wall_s="2.1",
                        model_seed=str(16 + repeat),
                        learning_rate=".5",
                    )
                )
        results, pairs = contrasts(records)
        primary = next(r for r in results if r["primary"])
        self.assertEqual(primary["mean_delta_wall_s"], -1.0)
        self.assertEqual(primary["bootstrap95_high_s"], -1.0)
        self.assertTrue(
            all(r["winning_fixed"] == "R+B" for r in pairs if r["reference"] == "best_fixed")
        )


class MigrationParityTests(unittest.TestCase):
    def test_original_mlp_math_and_dataset_are_preserved(self):
        path = ROOT / "legacy/prototypes/training_experiment.py"
        spec = importlib.util.spec_from_file_location("historical_training_reference", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        old_class = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "MLP"
        )
        new_class = next(
            n
            for n in ast.parse((ROOT / "adaptive_planning/training.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "MLP"
        )
        self.assertEqual(ast.dump(old_class), ast.dump(new_class))
        old_groups, old_val, _ = reference.get_data()
        groups, val, _ = training.get_data()
        for a, b in zip(old_groups, groups):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        for x, y in zip(old_val, val):
            np.testing.assert_array_equal(x, y)
        model = make_model(16, hidden=8, learning_rate=0.5)
        old = reference.MLP.__new__(reference.MLP)
        old.params = [p.copy() for p in model.params]
        old.nparam = model.nparam
        old.learning_rate = 0.5
        np.testing.assert_array_equal(old.gradient(*groups[0]), model.gradient(*groups[0]))
        vector = np.full(model.nparam, 1000, dtype=np.int64)
        old.update(vector)
        model.update(vector)
        for a, b in zip(old.params, model.params):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(old.score(*val), model.score(*val))


if __name__ == "__main__":
    unittest.main()
