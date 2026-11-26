"""Age-weighted transmission prior built only from delivered sensor reports."""

from collections import deque
from statistics import fmean

from .adaptive_scheduler import transfer_prior
from .aligned_scheduler import AlignedNetworkPlanner


class FreshnessMixturePlanner(AlignedNetworkPlanner):
    """Shrink an old report toward a bounded empirical history of reports.

    For age d, latest-report weight is 1/(1+d). The remaining weight belongs
    to the mean transfer budget over the last 16 DISTINCT capture IDs. This is
    a heuristic, not a calibrated posterior over network states. It never sees
    a profile catalogue, phase lengths, the true current profile or the future.
    Matched cost feedback, success learning and 5% exploration stay unchanged.
    """

    HISTORY_SIZE = 16

    def __init__(self, seed, switch_seconds, exploration=0.05):
        super().__init__(seed, switch_seconds, exploration)
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.age = 0

    def begin_round(self, observation, capture_attempt, attempt):
        if self.history and capture_attempt == self.history[-1][0]:
            names = ("model_mbps", "edge_mbps", "verifier_mbps", "latency_ms")
            if any(
                getattr(observation, name) != getattr(self.history[-1][1], name) for name in names
            ):
                raise ValueError("Repeated capture ID changed its network report")
        super().begin_round(observation, capture_attempt, attempt)
        if not self.history or capture_attempt > self.history[-1][0]:
            self.history.append((capture_attempt, observation))
        self.age = attempt - capture_attempt

    def transmission_budget(self, action, observation):
        latest = transfer_prior(action, observation)
        if not self.age:
            return latest
        history = fmean(transfer_prior(action, obs) for _, obs in self.history)
        weight = 1.0 / (1.0 + self.age)
        return weight * latest + (1.0 - weight) * history

    def estimate(self, action, observation, previous):
        probability, duration = super().estimate(action, observation, previous)
        if not self.age:
            return probability, duration
        return probability, max(
            0.001,
            duration
            - transfer_prior(action, observation)
            + self.transmission_budget(action, observation),
        )

    def choose(self, observation, previous):
        if self.active is None:
            raise RuntimeError("begin_round must precede a decision")
        action = super().choose(observation, previous)
        self.last_decision.update(
            telemetry_age_rounds=self.age,
            latest_report_weight=1.0 / (1.0 + self.age),
            history_capture_ids=[capture for capture, _ in self.history],
            transmission_budget_s=self.transmission_budget(action, observation),
        )
        return action
