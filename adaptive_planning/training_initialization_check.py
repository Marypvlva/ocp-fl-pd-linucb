"""Training-only reference diagnostic. No TCP timings or protocol claims."""

from .paths import ROOT
from .cli import source_hashes
import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np

from .tcp_closed_loop_pilot import append_csv, capture_sources
from .tcp_runtime import MAX_UPDATE, make_model


def run_case(groups, validation, seed, learning_rate, max_updates, target):
    model = make_model(seed, learning_rate=learning_rate)
    total = sum(len(y) for _, y in groups)
    loss, _ = model.score(*validation)
    rows = [dict(step=0, validation_loss=loss, max_abs_update=0, gradient_wall_s=0.0, accepted=0)]
    reached = loss <= target
    reason = "target_reached" if reached else "max_updates"
    for step in range(1, max_updates + 1):
        if reached:
            break
        start = perf_counter()
        updates = [np.rint(model.gradient(X, y) * len(y) / total * 1e6) for X, y in groups]
        largest = max(float(np.max(np.abs(v))) for v in updates)
        finite = all(np.all(np.isfinite(v)) for v in updates)
        elapsed = perf_counter() - start
        accepted = finite and largest <= MAX_UPDATE
        if accepted:
            # A central reference for training diagnosis only, never TCP verification.
            gradient = np.stack(updates).astype(np.int64).sum(axis=0, dtype=np.int64)
            model.update(gradient)
            loss, _ = model.score(*validation)
        rows.append(
            dict(
                step=step,
                validation_loss=loss,
                max_abs_update=largest if finite else None,
                gradient_wall_s=elapsed,
                accepted=int(accepted),
            )
        )
        if not accepted:
            reason = "update_out_of_range" if finite else "nonfinite_gradient"
            break
        if loss <= target:
            reached, reason = True, "target_reached"
    return rows, dict(
        model_seed=seed,
        learning_rate=learning_rate,
        reached=int(reached),
        successful_updates=sum(r["accepted"] for r in rows),
        final_loss=loss,
        stop_reason=reason,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--development-seeds", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=list(range(16, 24)))
    parser.add_argument("--rates", nargs="+", type=float, default=[1.0, 0.5])
    parser.add_argument("--max-updates", type=int, default=120)
    parser.add_argument("--target-loss", type=float, default=0.22)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        not 2 <= args.workers <= 16
        or args.max_updates < 1
        or not 0 < args.target_loss < 10
        or any(not np.isfinite(r) or r <= 0 for r in args.rates)
        or set(args.development_seeds) & set(args.validation_seeds)
    ):
        parser.error("Invalid or overlapping diagnostic settings")
    from .training import make_workers

    groups, validation, _ = make_workers(args.workers)
    folder = args.output or ROOT / "results" / (
        "training_init_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    meta = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        purpose="training-only central reference, not a TCP or cryptographic benchmark",
        predetermined_next_learning_rate=0.5,
        source_sha256=source_hashes(),
        status="running",
    )
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
    capture_sources(folder)
    all_rows, summaries = [], []
    try:
        for seed_set, seeds in (
            ("development", args.development_seeds),
            ("validation", args.validation_seeds),
        ):
            for seed in seeds:
                for rate in args.rates:
                    rows, summary = run_case(
                        groups, validation, seed, rate, args.max_updates, args.target_loss
                    )
                    summary.update(seed_set=seed_set, workers=args.workers)
                    for r in rows:
                        r.update(
                            seed_set=seed_set,
                            workers=args.workers,
                            model_seed=seed,
                            learning_rate=rate,
                        )
                    all_rows.extend(rows)
                    summaries.append(summary)
                    append_csv(folder / f"steps_{seed_set}_s{seed}_lr{rate}.csv", rows)
                    print(summary, flush=True)
        meta["status"] = "completed"
    except BaseException as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        append_csv(folder / "steps.csv", all_rows)
        append_csv(folder / "summary.csv", summaries)
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved: {folder}", flush=True)
    return folder


if __name__ == "__main__":
    main()
