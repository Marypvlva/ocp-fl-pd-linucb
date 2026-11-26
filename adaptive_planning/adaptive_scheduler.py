"""Network-aware baselines using only current advertised rates and past feedback."""

from dataclasses import dataclass
import numpy as np

ACTIONS = ("F+B", "F+P", "R+B", "R+P")


@dataclass(frozen=True)
class Observation:
    workers: int
    dimension: int
    model_mbps: float
    edge_mbps: float
    verifier_mbps: float
    latency_ms: float
    edge_trouble: bool = False
    projection_missing: bool = False


def transfer_prior(action, observation):
    """Coarse prior, NOT an elapsed-time measurement or a protocol decision.

    Includes all collected replicas, so R returns two full sums to the verifier.
    Payload-only serialization terms and critical-path latency are approximate;
    framing, overlap, CPU, localhost and pacing overhead are learned separately.
    """
    payload = observation.dimension * 8
    n = observation.workers
    copies = 2 if action.startswith("R") else 1

    def transfer(size, rate):
        return size / (rate * 1e6) if rate else 0.0

    verify_bytes = copies * payload
    if action.endswith("B"):
        verify_bytes += n * (payload + 64)
    else:
        verify_bytes += n * (64 + 8)
    return (
        transfer(n * payload, observation.model_mbps)
        + transfer(copies * n * payload, observation.edge_mbps)
        + transfer(verify_bytes, observation.verifier_mbps)
        + (8 if action.endswith("P") else 6) * observation.latency_ms / 1000.0
    )


def overhead_prior(action):
    # Fixed before dynamic trials; coarse scale from the earlier localhost stage.
    return 0.030 + 0.005 * action.endswith("P") + 0.006 * action.startswith("R")


def predicted_seconds(action, observation, previous, switch_seconds):
    return (
        transfer_prior(action, observation)
        + overhead_prior(action)
        + switch_seconds * (previous is not None and action != previous)
    )


def network_rule(observation, previous, switch_seconds):
    candidates = [
        a
        for a in ACTIONS
        if (not observation.edge_trouble or a.startswith("R"))
        and (not observation.projection_missing or a.endswith("B"))
    ]
    return min(
        candidates, key=lambda a: predicted_seconds(a, observation, previous, switch_seconds)
    )


class NetworkOnlinePlanner:
    """Small model-based bandit; no training on outcomes of unchosen actions.

    Known transmission budgets supply a structural prior. Only the chosen action
    updates success probability and mean residual time. Exploration is 5%.
    It has no access to a trace, next profile, fault injection, or phase duration.
    """

    def __init__(self, seed, switch_seconds, exploration=0.05):
        self.rng = np.random.default_rng(seed)
        self.switch_seconds = switch_seconds
        self.exploration = exploration
        self.stats = {a: dict(n=0, success=0, residual_sum=2 * overhead_prior(a)) for a in ACTIONS}
        self.last_decision = {}

    def estimate(self, action, observation, previous):
        stat = self.stats[action]
        probability = (stat["success"] + 4) / (stat["n"] + 5)
        overhead = stat["residual_sum"] / (stat["n"] + 2)
        duration = (
            transfer_prior(action, observation)
            + overhead
            + self.switch_seconds * (previous is not None and previous != action)
        )
        return probability, max(0.001, duration)

    def choose(self, observation, previous):
        explore = bool(self.rng.random() < self.exploration)
        if explore:
            action = str(self.rng.choice(ACTIONS))
        else:
            action = max(
                ACTIONS,
                key=lambda a: (
                    self.estimate(a, observation, previous)[0]
                    / self.estimate(a, observation, previous)[1]
                ),
            )
        probability, duration = self.estimate(action, observation, previous)
        self.last_decision = dict(
            exploration=explore,
            estimated_success=probability,
            predicted_seconds=duration,
            predicted_updates_per_second=probability / duration,
        )
        return action

    def observe(self, observation, action, success, cycle_seconds, switch_wall_seconds):
        residual = max(
            0.001, cycle_seconds - switch_wall_seconds - transfer_prior(action, observation)
        )
        stat = self.stats[action]
        stat["n"] += 1
        stat["success"] += int(success)
        stat["residual_sum"] += residual

    def feedback_count(self):
        return sum(stat["n"] for stat in self.stats.values())


class FrozenNetworkPlanner(NetworkOnlinePlanner):
    """Ablation: identical decisions/priors/RNG, with selected-action learning off."""

    def observe(self, observation, action, success, cycle_seconds, switch_wall_seconds):
        return None
