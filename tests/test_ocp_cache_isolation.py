"""Independent runs must not inherit the previous run's verified beacon cache."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_planning.ocp_analyze import validate
from adaptive_planning.ocp_experiment import main
from adaptive_planning.paths import ROOT


class CacheIsolationTests(unittest.TestCase):
    def test_two_real_runs_of_the_same_beacon_prefix_each_verify_cold(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            config = json.loads((ROOT / "configs/tcp_ocp_smoke.json").read_text())
            config.update(
                workers=[2],
                sample_size=1,
                model_seeds=[16, 17],
                rounds=1,
                budget_credits=2,
                pd_eta=0,
                fixed_action="R+P",
                trace_genesis=2524608000,
            )
            config_path = temp / "config.json"
            config_path.write_text(json.dumps(config))
            folder = temp / "runs"
            with patch("builtins.print"):
                main(["--config", str(config_path), "--output", str(folder)])
            metadata, summaries, rows = validate(folder)
            self.assertEqual(metadata["bls_cache_policy"], "reset_per_run")
            self.assertEqual(len(rows), 2)
            self.assertEqual([int(s["bls_cache_misses"]) for s in summaries], [1, 1])
            self.assertEqual(summaries[0]["bls_cache_hits"], summaries[1]["bls_cache_hits"])
            # Both runs used exactly the same beacon proof, with different training seeds.
            self.assertEqual(
                rows[0]["certificate"]["beacon_proof"], rows[1]["certificate"]["beacon_proof"]
            )
