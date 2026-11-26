import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from adaptive_planning.cli import catalog, freeze, source_bundle


class LifecycleTests(unittest.TestCase):
    def test_bundle_has_manifest_and_excludes_results_secrets_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            (Path(outside) / "private.py").write_text("private")
            (root / "adaptive_planning").symlink_to(outside, target_is_directory=True)
            (root / "main.py").write_text("print('ok')\n")
            (root / ".env").write_text("private")
            (root / "results").mkdir()
            (root / "results/raw.csv").write_text("x\n1\n")
            (root / "linked.py").symlink_to(root / ".env")
            output = root / "sources.zip"
            self.assertEqual(source_bundle(output, root), 1)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {"main.py", "SOURCE_MANIFEST.json"})
                hashes = json.loads(archive.read("SOURCE_MANIFEST.json"))
                self.assertEqual(
                    hashes["main.py"], hashlib.sha256(archive.read("main.py")).hexdigest()
                )
            with self.assertRaises(FileExistsError):
                source_bundle(output, root)

    def test_freeze_is_immutable_and_includes_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            config = root / "configs/test.json"
            config.write_text('{"seed": 1}')
            protocol = root / "PROTOCOL.md"
            protocol.write_text("fixed before the study")
            output = root / "work/frozen.json"
            result = freeze(config, protocol, output, root)
            self.assertEqual(result["config"], {"seed": 1})
            self.assertEqual(set(result["source_sha256"]), {"configs/test.json", "PROTOCOL.md"})
            with self.assertRaises(FileExistsError):
                freeze(config, protocol, output, root)
            outside = root / "work/unlisted.json"
            outside.write_text("{}")
            with self.assertRaises(ValueError):
                freeze(outside, protocol, root / "work/other.json", root)

    def test_catalog_keeps_failed_and_incomplete_separate_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, status in (
                ("done", "completed"),
                ("failed", "failed"),
                ("unfinished", "running"),
            ):
                folder = root / "results" / name
                folder.mkdir(parents=True)
                (folder / "metadata.json").write_text(
                    json.dumps(dict(status=status, family="test", config=dict(seed=1)))
                )
                (folder / "summary.csv").write_text("reached\n1\n0\n")
            bad = root / "results/bad"
            bad.mkdir()
            (bad / "metadata.json").write_text("{bad")
            before = {p: p.read_bytes() for p in (root / "results").rglob("*") if p.is_file()}
            result = catalog(root)
            statuses = {row["path"].split("/")[-1]: row["status"] for row in result["experiments"]}
            self.assertEqual(
                statuses,
                dict(
                    done="completed", failed="failed", unfinished="running", bad="invalid_metadata"
                ),
            )
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertTrue(
                all(
                    row["recorded_runs"] == 2
                    for row in result["experiments"]
                    if row["status"] != "invalid_metadata"
                )
            )


if __name__ == "__main__":
    unittest.main()
