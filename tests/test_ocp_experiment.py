import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_planning import ocp_experiment
from adaptive_planning.paths import ROOT


class JointFaultTests(unittest.TestCase):
    def test_intermittent_failure_requires_outage_and_targeted_invitation(self):
        condition = "intermittent_targeted_edge_failure"
        targeted = (0, 2, 3)
        other = (2, 3, 4)
        for attempt in range(1, 17):
            expected = (0,) if 5 <= attempt <= 8 or 13 <= attempt <= 16 else ()
            with self.subTest(attempt=attempt):
                self.assertEqual(
                    ocp_experiment.round_fault(condition, attempt, targeted, 6).failed_edges,
                    expected,
                )
                self.assertEqual(
                    ocp_experiment.round_fault(condition, attempt, other, 6).failed_edges,
                    (),
                )


class JointJournalTests(unittest.TestCase):
    def test_multiple_completed_cases_keep_all_summaries(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            config = json.loads((ROOT / "configs/tcp_ocp_smoke.json").read_text())
            config.update(workers=[2], sample_size=1, conditions=["healthy", "non_reveal"])
            (temp / "config.json").write_text(json.dumps(config))

            def run(config, population, seed, condition, folder):
                label = f"w{population}_s{seed}_{condition}"
                return [dict(run=label)], dict(
                    run=label,
                    condition=condition,
                    accepted_updates=0,
                    attempts=0,
                    spent_credits=0,
                    stop_reason="test",
                )

            with (
                patch.object(ocp_experiment, "run_joint", side_effect=run),
                patch.object(ocp_experiment, "capture_sources"),
                patch.object(ocp_experiment, "source_hashes", return_value={}),
            ):
                # The writer must preserve both cases instead of trying to
                # exclusively create summary.csv for every successive case.
                with patch("builtins.print"):
                    ocp_experiment.main(
                        ["--config", str(temp / "config.json"), "--output", str(temp / "result")]
                    )
            with (temp / "result/summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["condition"] for r in rows], ["healthy", "non_reveal"])
            self.assertEqual(
                json.loads((temp / "result/metadata.json").read_text())["status"], "completed"
            )
            self.assertEqual(len(json.loads((temp / "result/rounds.json").read_text())), 2)
