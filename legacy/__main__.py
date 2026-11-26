"""Run an archived prototype in a new isolated result directory."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import uuid

from adaptive_planning.cli import source_bundle
from adaptive_planning.paths import ROOT


def main():
    os.environ.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(ROOT / "work/mplconfig"),
        XDG_CACHE_HOME=str(ROOT / "work/cache"),
    )
    sources = Path(__file__).parent / "prototypes"
    names = sorted(p.stem for p in sources.glob("*.py"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", nargs="?", choices=names)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list or not args.script:
        print("\n".join(names))
        return
    folder = (
        ROOT
        / "results"
        / (
            "legacy_"
            + args.script
            + "_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
            + uuid.uuid4().hex[:8]
        )
    )
    folder.mkdir(parents=True, exist_ok=False)
    copied = folder / "sources"
    shutil.copytree(sources, copied, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("crypto_baseline", "randomized_check"):
        (copied / f"{name}.py").write_text(
            f"import sys\nfrom adaptive_planning import {name} as implementation\nsys.modules[__name__] = implementation\n"
        )
    source_bundle(folder / "source_snapshot.zip")
    metadata = dict(
        status="running",
        family="legacy_isolated",
        script=args.script,
        definition="historical simulation/benchmark, not the active real-TCP study",
    )
    sys.path.insert(0, str(copied))
    try:
        runpy.run_path(str(copied / f"{args.script}.py"), run_name="__main__")
        metadata["status"] = "completed"
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        sys.path.remove(str(copied))
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"Archived run saved separately: {folder}")


if __name__ == "__main__":
    main()
