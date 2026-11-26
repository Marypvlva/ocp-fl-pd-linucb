"""A resumed long experiment must retain its selected policy set and order."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_planning import ocp_compare
from adaptive_planning.paths import ROOT


class ComparisonResumeTests(unittest.TestCase):
    def test_resume_keeps_selected_policies_and_rejects_order_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            config = json.loads((ROOT / "configs/tcp_ocp_smoke.json").read_text())
            config.update(workers=[4])
            config_path = temp / "config.json"
            config_path.write_text(json.dumps(config))
            output = temp / "comparison"
            launched = []

            def run(argv):
                config_file = Path(argv[argv.index("--config") + 1])
                folder = Path(argv[argv.index("--output") + 1])
                folder.mkdir()
                launched.append(folder.name)
                (folder / "metadata.json").write_text(
                    json.dumps({"config": json.loads(config_file.read_text())})
                )

            def validate(folder):
                metadata = json.loads((folder / "metadata.json").read_text())
                return metadata, [{"run": "initial_target_reached", "attempts": 0}], []

            args = ["--config", str(config_path), "--output", str(output)]
            with (
                patch.object(ocp_compare, "run_experiment", side_effect=run),
                patch.object(ocp_compare, "validate", side_effect=validate),
                patch("builtins.print"),
            ):
                ocp_compare.main(
                    args + ["--restricted-bandits", "--policies", "pd_projection", "R+P"]
                )
                ocp_compare.main(args + ["--resume"])
                self.assertEqual(launched, ["pd_projection", "R+P"])
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    ocp_compare.main(args + ["--resume", "--policies", "R+P", "pd_projection"])
                self.assertEqual(launched, ["pd_projection", "R+P"])
