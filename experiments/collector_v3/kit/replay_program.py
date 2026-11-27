"""Causal controller reconstruction from one logged episode; never an offline policy evaluation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from model import ARMS, SLA, Policy, Settings
from gateway_study import caps
from program import read_plan


def reconstruct(directory, seed, mode):
    directory = Path(directory)
    config, digest = read_plan(directory / "plan.json")
    SLA[:] = config["windows_ms"]
    h, n, k = config["horizon"], config["population"], config["k"]
    index = config["seeds"].index(seed)
    settings = Settings(
        **(
            config["policy_specs"][mode]
            if "policy_specs" in config
            else config["calibration"]["records"][mode]["chosen"]["settings"]
        )
    )
    from article_policy import ArticlePolicy

    policy_class = ArticlePolicy if settings.mode == "article_pd" else Policy
    policy = policy_class(
        settings,
        1,
        n,
        k,
        h,
        config.get("mirror_limit", h // 4),
        config.get("policy_tie_seed", 53),
        caps(index, h, config.get("site_fraction", 0.2))[None, :],
        np.array([config["calibration"]["prices"][str(index % 4)]]),
    )
    rows = []
    for file in sorted((directory / f"{seed}_{mode}").glob("round-*.json")):
        row = json.loads(file.read_text())
        rd = row["round"]
        c = config["windows_ms"].index(row["window_ms"])
        forced = config.get("forced_arms", {}).get(str(seed))
        arm = int(
            policy.decide(
                c,
                np.array([row["selected"]]),
                forced_arm=forced[rd - 1] if forced is not None else None,
            )[0]
        )
        if list(ARMS[arm]) != row["gateways"]:
            raise ValueError(
                f"Causal replay action mismatch at {seed}/{mode}/{rd}; no reconstruction certified"
            )
        samples = np.full((1, k, 2), -1.0)
        for j, g in enumerate(row["gateways"]):
            for i, member in enumerate(row["selected"]):
                samples[0, i, j] = (
                    row["metrics"].get("collector_ms", {}).get(str(g), {}).get(str(member), -1.0)
                )
        mask = (
            np.array(row["path_observed"], dtype=bool)
            if config.get("feedback_schema", 1) >= 2
            else None
        )
        policy.observe(samples, np.array([row["accepted"]]), mask)
        rows.append(
            dict(
                round=rd,
                seed=seed,
                policy=mode,
                provenance="causal_reconstruction_not_direct_measurement",
                round_sha256=sha256(file.read_bytes()).hexdigest(),
                plan_sha256=digest,
                **policy.last_decision,
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.resolve() == a.directory.resolve() or a.directory.resolve() in a.out.resolve().parents:
        raise ValueError("Do not modify the original series")
    a.out.mkdir(parents=True, exist_ok=False)
    config, _ = read_plan(a.directory / "plan.json")
    for seed in config["seeds"]:
        for mode in config["modes"]:
            rows = reconstruct(a.directory, seed, mode)
            with (a.out / f"{seed}_{mode}.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
    (a.out / "SCOPE.txt").write_text(
        "Controller state reconstructed from past executed observations, with exact logged-action agreement. Not directly measured prices; not counterfactual replay, regret or a claim about unchosen paths. Historical v0.6.1 censoring semantics deliberately retained.\n"
    )


if __name__ == "__main__":
    main()
