"""Local project entry point. Help, indexing and source snapshots use stdlib only."""

from .paths import ROOT
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import zipfile

SOURCE_NAMES = (
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-dev.txt",
    "Makefile",
    ".gitignore",
    "pyproject.toml",
)


def source_files(root=ROOT):
    root = root.resolve()
    paths = [
        *root.glob("*.py"),
        *root.glob("*.md"),
        *root.glob("tests/*.py"),
        *root.glob("tests/fixtures/*.json"),
        *root.glob("contracts/*.sol"),
        *root.glob("contracts/*.mjs"),
        *root.glob("contracts/*.json"),
        *root.glob("contracts/artifacts/*.json"),
        *root.glob("configs/*.json"),
        *root.glob(".github/workflows/*.yml"),
        *root.glob("adaptive_planning/**/*.py"),
        *root.glob("legacy/**/*.py"),
        *root.glob("legacy/*.md"),
        *root.glob("docs/**/*.md"),
    ]
    paths += [root / name for name in SOURCE_NAMES]
    return sorted(
        {
            p
            for p in paths
            if p.is_file()
            and not p.is_symlink()
            and p.resolve().is_relative_to(root)
            and "__pycache__" not in p.parts
            and "results" not in p.relative_to(root).parts
        }
    )


def source_hashes(root=ROOT):
    root = root.resolve()
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files(root)
    }


def freeze(config, protocol, output, root=ROOT):
    root = root.resolve()
    config, protocol = config.resolve(), protocol.resolve()
    if not config.is_relative_to(root.resolve()) or not protocol.is_relative_to(root.resolve()):
        raise ValueError("Config and protocol must belong to the project")
    payload = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        config_path=str(config.relative_to(root)),
        config=json.loads(config.read_text()),
        protocol_path=str(protocol.relative_to(root)),
        python=platform.python_version(),
        source_sha256=source_hashes(root),
    )
    if not all(str(p.relative_to(root)) in payload["source_sha256"] for p in (config, protocol)):
        raise ValueError("Config and protocol were not included in the source snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    return payload


def source_bundle(output, root=ROOT):
    root = root.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(root)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(root / name, name)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or any(
            hashlib.sha256(archive.read(n)).hexdigest() != h for n, h in hashes.items()
        ):
            raise ValueError("Source archive verification failed")
    return len(hashes)


def catalog(root=ROOT):
    results = root / "results"
    entries = []
    for folder in sorted(p for p in results.iterdir() if p.is_dir()) if results.exists() else []:
        meta = folder / "metadata.json"
        entry = dict(path=str(folder.relative_to(root)), status="legacy", family="legacy")
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
                entry.update(
                    status=data.get("status", "unknown"),
                    family=data.get("family", "historical_tcp"),
                    stage=data.get("config", {}).get("stage"),
                    seed=data.get("config", {}).get("seed"),
                    planned_runs=data.get("planned_runs"),
                    primary=data.get("primary_contrast"),
                    error=data.get("error"),
                )
            except (ValueError, TypeError, AttributeError) as error:
                entry.update(status="invalid_metadata", error=str(error))
        if (folder / "summary.csv").exists():
            import csv

            try:
                with (folder / "summary.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                entry["recorded_runs"] = len(rows)
                entry["reached_runs"] = sum(row.get("reached") == "1" for row in rows)
            except (OSError, ValueError) as error:
                entry.update(status="invalid_summary", error=str(error))
        entries.append(entry)
    return dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        definition="Inventory only; legacy and incomplete runs are not pooled or reclassified as completed",
        legacy_root_csv_files=len(list(results.glob("*.csv"))) if results.exists() else 0,
        experiments=entries,
    )


def doctor(root=ROOT, bind=True):
    checks = dict(
        python=platform.python_version(),
        supported_python=sys.version_info[:2] == (3, 13),
        platform=platform.platform(),
        project=str(root),
        git_repository=(root / ".git").exists(),
    )
    dependencies = {}
    for name in (
        "numpy",
        "matplotlib",
        "scikit-learn",
        "cryptography",
        "web3",
        "eth-tester",
        "py-ecc",
        "py-evm",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    checks["dependencies"] = dependencies
    checks["tcp_loopback"] = None
    if bind:
        try:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
            checks["tcp_loopback"] = True
        except OSError as error:
            checks.update(tcp_loopback=False, tcp_error=str(error))
    try:
        (root / "work").mkdir(exist_ok=True)
        with tempfile.TemporaryFile(dir=root / "work"):
            pass
        checks["writable_work"] = True
    except OSError as error:
        checks.update(writable_work=False, write_error=str(error))
    try:
        artifact = json.loads((root / "contracts/artifacts/OcpfScheduler.json").read_text())
        checks["contract_artifact_current"] = (
            artifact["source_sha256"]
            == hashlib.sha256((root / "contracts/OcpfScheduler.sol").read_bytes()).hexdigest()
        )
    except (OSError, ValueError, KeyError):
        checks["contract_artifact_current"] = False
    checks["ok"] = (
        checks["supported_python"]
        and all(dependencies.values())
        and checks["contract_artifact_current"]
        and checks["writable_work"]
        and checks["tcp_loopback"] is not False
    )
    return checks


def run(script, arguments):
    env = os.environ.copy()
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        env[name] = "1"
    env.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(ROOT / "work/mplconfig"),
        XDG_CACHE_HOME=str(ROOT / "work/cache"),
    )
    if len(script) == 1 and script[0].endswith(".py"):
        script = ["-m", "adaptive_planning." + Path(script[0]).stem]
    return subprocess.call([sys.executable, *script, *arguments], cwd=ROOT, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check Python, dependencies, work directory and loopback TCP")
    sub.add_parser("test", help="Run all tests, including real TCP and fault-injection checks")
    smoke = sub.add_parser("smoke", help="Short integration check; not a T_epsilon benchmark")
    smoke.add_argument("--workers", nargs="+", type=int, choices=range(2, 17), default=[4, 8, 12])
    smoke.add_argument(
        "--study", choices=("ocp", "pd-linucb", "availability", "freshness"), default="ocp"
    )
    study = sub.add_parser("study", help="Run an explicit predeclared study")
    study.add_argument(
        "experiment",
        choices=("ocp", "pd-linucb", "availability", "freshness"),
        nargs="?",
        default="ocp",
    )
    study.add_argument("--config", type=Path)
    study.add_argument("--output", type=Path)
    train = sub.add_parser(
        "train", help="OCP-FL client scheduling plus PD-LinUCB execution choices"
    )
    train.add_argument("--engine", choices=("joint", "pd-only"), default="joint")
    train.add_argument("--config", type=Path)
    train.add_argument("--output", type=Path)
    index = sub.add_parser(
        "index", help="Inventory completed, failed, incomplete and legacy experiments"
    )
    index.add_argument("--output", type=Path)
    frozen = sub.add_parser(
        "freeze", help="Create an immutable protocol/source manifest before a study"
    )
    frozen.add_argument("--config", type=Path, required=True)
    frozen.add_argument("--protocol", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    bundle = sub.add_parser(
        "bundle", help="Create a verified source-only ZIP without datasets or credentials"
    )
    bundle.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="Audit a completed telemetry study")
    verify.add_argument("folder", type=Path)
    report = sub.add_parser("report", help="Analyze a completed freshness study")
    report.add_argument("folder", type=Path)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ok"] else 1
        if args.command == "test":
            return run(["-m", "unittest", "discover", "-s", "tests", "-v"], [])
        if args.command == "train":
            config = args.config or ROOT / (
                "configs/tcp_ocp_train.json"
                if args.engine == "joint"
                else "configs/tcp_pd_linucb_train.json"
            )
            command = ([] if args.engine == "joint" else ["--planner-only"]) + [
                "--config",
                str(config),
            ]
            if args.output:
                command += ["--output", str(args.output)]
            return run(
                ["ocp_experiment.py" if args.engine == "joint" else "pd_linucb_experiment.py"],
                command,
            )
        if args.command in ("smoke", "study"):
            family = args.study if args.command == "smoke" else args.experiment
            config_name = family.replace("-", "_")
            config = (
                ROOT / f"configs/tcp_{config_name}_smoke.json"
                if args.command == "smoke"
                else args.config or ROOT / f"configs/tcp_{config_name}_eval.json"
            )
            command = (
                ["--study", "freshness", "--modes", "exact", "stale_noise"]
                if family == "freshness"
                else []
            )
            command += ["--config", str(config)]
            if args.command == "smoke":
                command += ["--workers", *map(str, args.workers)]
            elif args.output:
                command += ["--output", str(args.output)]
            return run(
                [
                    "telemetry_robustness_experiment.py"
                    if family == "freshness"
                    else "pd_linucb_experiment.py"
                    if family == "pd-linucb"
                    else "ocp_experiment.py"
                    if family == "ocp"
                    else "availability_experiment.py"
                ],
                command,
            )
        if args.command == "index":
            result = catalog()
            if args.output:
                args.output.mkdir(parents=True, exist_ok=False)
                (args.output / "experiments.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False)
                )
                lines = [
                    "# Реестр экспериментов",
                    "",
                    "Незавершённые и старые серии не объединяются с завершёнными.",
                    "",
                    "| Каталог | Статус | Семейство | Запусков | Достигли |",
                    "|---|---|---|---:|---:|",
                ]
                for e in result["experiments"]:
                    lines.append(
                        f"| {e['path']} | {e['status']} | {e['family']} | {e.get('recorded_runs', '—')} | {e.get('reached_runs', '—')} |"
                    )
                (args.output / "experiments.md").write_text("\n".join(lines) + "\n")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "freeze":
            result = freeze(args.config, args.protocol, args.output)
            print(f"Frozen {len(result['source_sha256'])} files: {args.output}")
            return 0
        if args.command == "bundle":
            print(f"Verified {source_bundle(args.output)} source files: {args.output}")
            return 0
        if args.command == "verify":
            meta = json.loads((args.folder / "metadata.json").read_text())
            family = meta.get("family")
            if family == "ocp_pd_joint":
                from .ocp_analyze import validate
            elif family == "pd_linucb_budget":
                from .pd_linucb_analyze import validate
            elif family == "persistent_availability":
                from .availability_analyze import validate
            elif family == "telemetry_freshness":
                from .freshness_analyze import validate
            elif family == "telemetry_alignment":
                from .telemetry_alignment_analyze import validate
            elif family == "telemetry_robustness":
                from .telemetry_robustness_analyze import validate
            else:
                raise ValueError("Use the original analyzer for this historical experiment family")
            _, summary, rows = validate(args.folder)
            print(
                json.dumps(
                    dict(status="passed", family=family, runs=len(summary), rounds=len(rows))
                )
            )
            return 0
        if args.command == "report":
            family = json.loads((args.folder / "metadata.json").read_text()).get("family")
            analyzer = {
                "ocp_pd_joint": "ocp_analyze.py",
                "pd_linucb_budget": "pd_linucb_analyze.py",
                "persistent_availability": "availability_analyze.py",
                "telemetry_freshness": "freshness_analyze.py",
                "telemetry_alignment": "telemetry_alignment_analyze.py",
                "telemetry_robustness": "telemetry_robustness_analyze.py",
            }.get(family)
            if analyzer is None:
                raise ValueError("Use the archived analyzer for this experiment family")
            return run([analyzer], [str(args.folder), "--output", str(args.output)])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
