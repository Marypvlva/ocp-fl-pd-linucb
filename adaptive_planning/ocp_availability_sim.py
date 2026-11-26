"""Exploratory paired availability simulation; no TCP, EVM or model training.

The current hidden failure is never an input to a policy. Only previous protocol
errors and selected-action binary rewards (delayed two rounds) are observed.
"""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import heapq
import json
from pathlib import Path

import numpy as np

from .adaptive_scheduler import Observation
from .pd_linucb import PrimalDualLinUCB
from .pd_linucb_planner import make_context
from .pd_linucb_planner import DEFAULT_COSTS
from .structured_pd_linucb import SharedPrimalDualLinUCB

POLICIES = (
    "pd",
    "pd_no_type",
    "pd_type_onehot",
    "pd_type_guard",
    "linucb",
    "reactive_edge",
    "reactive_any",
    "spread",
)
JOINT_POLICIES = ("pd_joint", "reactive_joint")
STRUCTURED_POLICIES = ("pd_joint_compact", "pd_shared")


def generate_trace(seed, horizon, persistence, participant_probability):
    rng = np.random.default_rng(seed)
    # The edge fault rate remains fixed when adding unrepairable participant faults.
    draws = rng.choice(
        3, horizon, p=[0.65 - participant_probability, 0.35, participant_probability]
    )
    refresh = rng.random(horizon) >= persistence
    states = draws.copy()
    for t in range(1, horizon):
        if not refresh[t]:
            states[t] = states[t - 1]
    # Uniformly sample 3 of 6 clients with exactly 2 targeted clients.
    # P(at least one targeted) = 1 - C(4,3)/C(6,3) = 0.8.
    targeted = rng.random(horizon) < 0.8
    effective = np.where((states == 1) & ~targeted, 0, states)
    return effective


def simulate(policy, failures, budget, delay=2):
    if policy not in (*POLICIES, *JOINT_POLICIES, *STRUCTURED_POLICIES):
        raise ValueError("Unknown simulation policy")
    failures = np.asarray(failures)
    if failures.ndim != 1 or not len(failures) or not np.isin(failures, [0, 1, 2]).all():
        raise ValueError("Expected a nonempty trace of healthy/edge/participant outcomes")
    horizon = len(failures)
    if type(budget) is not int or not horizon <= budget <= 2 * horizon:
        raise ValueError("Budget must fund the cheap horizon and at most the reserve horizon")
    if type(delay) is not int or delay < 0:
        raise ValueError("Invalid feedback delay")
    compact = policy in ("pd_type_onehot", "pd_type_guard")
    structured = policy in STRUCTURED_POLICIES
    engine_type = SharedPrimalDualLinUCB if policy == "pd_shared" else PrimalDualLinUCB
    engine = engine_type(
        5 if structured else 3 if compact else 17,
        DEFAULT_COSTS
        if policy in (*JOINT_POLICIES, *STRUCTURED_POLICIES)
        else {"F+P": 1, "R+P": 2},
        budget,
        horizon,
        eta=0
        if policy in ("linucb", "reactive_edge", "reactive_any", "reactive_joint", "spread")
        else 0.05,
    )
    labels = []
    previous_action = None
    previous_issue = 0
    accepted = reserves = saved = wasted = projection_recoveries = 0
    actions = []
    for step, hidden in enumerate(failures, 1):
        if engine.ledger.stop_reason():
            break
        while labels and labels[0][0] <= step:
            _, identity, reward = heapq.heappop(labels)
            engine.reward(identity, reward)
        obs = Observation(
            3, 4810, 0, 0, 0, 0, previous_issue == 1, previous_issue == 2 and policy != "pd_no_type"
        )
        context = np.eye(3)[previous_issue] if compact else make_context(obs, previous_action, 0)
        if structured:
            context = np.array(
                [
                    1,
                    obs.edge_trouble,
                    obs.projection_missing,
                    previous_action is not None and previous_action.startswith("R"),
                    previous_action is not None and previous_action.endswith("B"),
                ],
                dtype=float,
            )
        left = horizon - step + 1
        slack = engine.ledger.remaining - left
        forced = None
        if policy == "reactive_joint":
            forced = "F+P"
            if slack > 0:
                if obs.projection_missing:
                    forced = "F+B"
                elif obs.edge_trouble or slack >= left:
                    forced = "R+P"
        elif policy in ("reactive_edge", "reactive_any"):
            risk = obs.edge_trouble or (policy == "reactive_any" and obs.projection_missing)
            forced = "R+P" if slack > 0 and (risk or slack >= left) else "F+P"
        elif policy == "spread":
            quota = budget - horizon
            forced = "R+P" if step * quota // horizon > (step - 1) * quota // horizon else "F+P"
        elif policy == "pd_type_guard" and slack == 0:
            forced = "F+P"
        action = engine.select(context, forced_action=forced)
        # This line is the first policy-dependent use of the current hidden state.
        reserve = action.startswith("R")
        baseline = action.endswith("B")
        success = hidden == 0 or (hidden == 1 and reserve) or (hidden == 2 and baseline)
        heapq.heappush(labels, (step + delay, step, float(success)))
        while labels and labels[0][0] <= step:
            _, identity, reward = heapq.heappop(labels)
            engine.reward(identity, reward)
        accepted += int(success)
        reserves += int(reserve)
        saved += int(reserve and hidden == 1)
        wasted += int(reserve and not baseline and hidden == 2)
        projection_recoveries += int(baseline and hidden == 2)
        previous_action = action
        # B does not request projection proofs, so it cannot observe their absence.
        previous_issue = 0 if hidden == 2 and baseline else int(hidden)
        actions.append(action)
    assert engine.ledger.spent <= budget
    if policy in ("reactive_edge", "reactive_any", "reactive_joint", "spread", "pd_type_guard"):
        assert len(actions) == horizon
    return dict(
        accepted=accepted,
        attempted=len(actions),
        spent=float(engine.ledger.spent),
        reserves=reserves,
        rescued=saved,
        reserve_on_unrepairable=wasted,
        projection_recoveries=projection_recoveries,
        labels_observed=int(engine.observed.sum()),
        actions=actions,
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries(rows, control="reactive_edge"):
    groups = defaultdict(dict)
    for row in rows:
        key = (row["participant_probability"], row["persistence"], row["budget"], row["seed"])
        groups[key][row["policy"]] = row
    cells = defaultdict(list)
    for key, policies in groups.items():
        for policy, row in policies.items():
            cells[(*key[:3], policy)].append((row, policies[control]))
    output = []
    for key, pairs in sorted(cells.items()):
        deltas = np.array([row["accepted"] - control["accepted"] for row, control in pairs])
        # Resample independent paired seeds, never individual rounds.
        rng = np.random.default_rng(56123)
        means = rng.choice(deltas, size=(5000, len(deltas)), replace=True).mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        output.append(
            dict(
                participant_probability=key[0],
                persistence=key[1],
                budget=key[2],
                policy=key[3],
                seeds=len(pairs),
                accepted_mean=float(np.mean([row["accepted"] for row, _ in pairs])),
                rescued_mean=float(np.mean([row["rescued"] for row, _ in pairs])),
                wasted_mean=float(np.mean([row["reserve_on_unrepairable"] for row, _ in pairs])),
                **{f"delta_vs_{control}": float(deltas.mean())},
                delta_bootstrap_low=float(lo),
                delta_bootstrap_high=float(hi),
                wins=int((deltas > 0).sum()),
                ties=int((deltas == 0).sum()),
                losses=int((deltas < 0).sum()),
            )
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--joint-controls", action="store_true", help="Also compare all four actions"
    )
    parser.add_argument("--seed-start", type=int, default=8000)
    parser.add_argument(
        "--shared-model", action="store_true", help="Also compare shared additive effects"
    )
    args = parser.parse_args(argv)
    policies = list(POLICIES)
    if args.joint_controls or args.shared_model:
        policies.extend(JOINT_POLICIES)
    if args.shared_model:
        policies.extend(STRUCTURED_POLICIES)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        horizon=120,
        seeds=list(range(args.seed_start, args.seed_start + 48)),
        budgets=[138, 174, 216],
        persistence=[0.0, 0.5, 0.85],
        participant_probability=[0.0, 0.25],
        policies=policies,
        alpha=1.0,
        ridge=1.0,
        eta=0.05,
        delay=2,
        primary_control="reactive_joint"
        if args.joint_controls or args.shared_model
        else "reactive_edge",
        inference="exploratory paired bootstrap; no multiplicity correction",
        feedback="selected action only",
        reward="binary accepted update",
        environment="synthetic structural outcomes; no model training, TCP, EVM or actual rosters",
    )
    # Save protocol and the exact three implementation files before generating outcomes.
    sources = args.output / "sources"
    sources.mkdir()
    hashes = {}
    for name in (
        "structured_pd_linucb.py",
        "ocp_availability_sim.py",
        "pd_linucb.py",
        "pd_linucb_planner.py",
        "adaptive_scheduler.py",
    ):
        content = Path(__file__).with_name(name).read_bytes()
        (sources / name).write_bytes(content)
        hashes[name] = sha256(content).hexdigest()
    manifest["source_sha256"] = hashes
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    traces = []
    for participant in manifest["participant_probability"]:
        for rho in manifest["persistence"]:
            for seed in manifest["seeds"]:
                failures = generate_trace(seed, manifest["horizon"], rho, participant)
                traces.append(
                    dict(
                        participant_probability=participant,
                        persistence=rho,
                        seed=seed,
                        failures=failures.tolist(),
                    )
                )
                for budget in manifest["budgets"]:
                    for policy in manifest["policies"]:
                        result = simulate(policy, failures, budget)
                        result.pop("actions")
                        rows.append(
                            dict(
                                participant_probability=participant,
                                persistence=rho,
                                seed=seed,
                                budget=budget,
                                policy=policy,
                                **result,
                            )
                        )
            print(f"Finished participant={participant}, persistence={rho}", flush=True)
    (args.output / "traces.json").write_text(json.dumps(traces) + "\n")
    write_csv(args.output / "cases.csv", rows)
    write_csv(args.output / "summary.csv", summaries(rows))
    if "reactive_joint" in policies:
        write_csv(args.output / "joint_contrasts.csv", summaries(rows, "reactive_joint"))
    if "pd_joint_compact" in policies:
        write_csv(args.output / "sharing_contrasts.csv", summaries(rows, "pd_joint_compact"))
    print(f"Saved {len(rows)} policy runs across {len(traces)} paired traces", flush=True)


if __name__ == "__main__":
    main()
