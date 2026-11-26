"""Quality failures must never appear as inexpensive successes in resource comparisons."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_planning import ocp_compare_report


class ResourceReportTests(unittest.TestCase):
    def test_frontier_excludes_failures_and_dominated_cases_and_keeps_ties(self):
        cases = [
            dict(policy="cheap", reached=True, spent_credits=10, gas_used=20),
            dict(policy="gas_efficient", reached=True, spent_credits=20, gas_used=10),
            dict(policy="tied", reached=True, spent_credits=10, gas_used=20),
            dict(policy="dominated", reached=True, spent_credits=20, gas_used=20),
            dict(policy="failed", reached=False, spent_credits=1, gas_used=1),
        ]
        self.assertEqual(
            ocp_compare_report.pareto_policies(cases), ["cheap", "gas_efficient", "tied"]
        )
        self.assertEqual(ocp_compare_report.pareto_policies([cases[-1]]), [])

    def test_continued_training_is_rejected_as_resource_to_first_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "base_config.json").write_text(json.dumps({"stop_at_target": False}))
            with self.assertRaisesRegex(ValueError, "stop_at_target"):
                ocp_compare_report.audit_comparison(folder)

    def test_report_marks_incomplete_policy_means_undefined(self):
        cases = [
            dict(
                policy="complete",
                run="a",
                reached=True,
                spent_credits=20,
                gas_used=100,
                attempts=10,
            ),
            dict(
                policy="complete",
                run="b",
                reached=True,
                spent_credits=30,
                gas_used=100,
                attempts=10,
            ),
            dict(
                policy="incomplete",
                run="a",
                reached=False,
                spent_credits=1,
                gas_used=1,
                attempts=1,
                stop_reason="budget_exhausted",
                final_loss=0.23,
            ),
            dict(
                policy="incomplete",
                run="b",
                reached=True,
                spent_credits=30,
                gas_used=100,
                attempts=10,
            ),
        ]
        cases = [
            dict(target_loss=0.22, model_seed={"a": 1, "b": 2}[case["run"]], **case)
            for case in cases
        ]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report"
            with (
                patch.object(ocp_compare_report, "audit_comparison", return_value=cases),
                patch("builtins.print"),
            ):
                ocp_compare_report.main([temp, "--output", str(output)])
            result = json.loads((output / "analysis.json").read_text())
            summaries = {row["policy"]: row for row in result["summaries"]}
            self.assertEqual(summaries["complete"]["credits_at_target_mean"], 25)
            self.assertEqual(summaries["incomplete"]["reached"], 1)
            self.assertIsNone(summaries["incomplete"]["credits_at_target_mean"])
            self.assertIsNone(summaries["incomplete"]["gas_at_target_mean"])
            self.assertEqual(result["pareto_by_run"]["a"], ["complete"])
            self.assertIn("budget_exhausted", (output / "report.md").read_text())
