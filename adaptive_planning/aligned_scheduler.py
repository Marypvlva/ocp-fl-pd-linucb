"""Match delayed sensor reports to the cost of their capture round, causally."""

import math

from .adaptive_scheduler import NetworkOnlinePlanner, transfer_prior


class AlignedNetworkPlanner(NetworkOnlinePlanner):
    """Same action scoring and exploration; delayed, matched cost feedback.

    Protocol success is observed immediately. Cost feedback waits for the report
    captured in that outcome's round. A report is used once, even when startup
    repeats it. The final undelivered outcomes are not trained retrospectively.
    No true current profile, phase duration or future report enters this class.
    """

    def __init__(self, seed, switch_seconds, exploration=0.05):
        super().__init__(seed, switch_seconds, exploration)
        for stat in self.stats.values():
            stat["cost_n"] = 0
        self.pending = {}
        self.attempt = 0
        self.last_capture = 0
        self.last_cost_capture = 0
        self.active = None

    def estimate(self, action, observation, previous):
        stat = self.stats[action]
        probability = (stat["success"] + 4) / (stat["n"] + 5)
        overhead = stat["residual_sum"] / (stat["cost_n"] + 2)
        duration = (
            transfer_prior(action, observation)
            + overhead
            + self.switch_seconds * (previous is not None and previous != action)
        )
        return probability, max(0.001, duration)

    def _cost_feedback(self, observation, capture, outcome):
        action, cycle, switch = outcome
        residual = max(0.001, cycle - switch - transfer_prior(action, observation))
        stat = self.stats[action]
        stat["residual_sum"] += residual
        stat["cost_n"] += 1
        self.last_cost_capture = capture

    def begin_round(self, observation, capture_attempt, attempt):
        if (
            type(attempt) is not int
            or attempt != self.attempt + 1
            or self.active is not None
            or type(capture_attempt) is not int
            or not self.last_capture <= capture_attempt <= attempt
            or capture_attempt < 1
            or attempt - capture_attempt > 20
        ):
            raise ValueError("Reports must follow the causal, bounded attempt prefix")
        if capture_attempt < attempt and capture_attempt > self.last_cost_capture:
            if capture_attempt not in self.pending:
                raise ValueError("No chosen-action outcome for this capture")
            self._cost_feedback(observation, capture_attempt, self.pending.pop(capture_attempt))
        # A skipped report is discarded; no current report is substituted for it.
        self.pending = {k: v for k, v in self.pending.items() if k > capture_attempt}
        self.attempt, self.last_capture = attempt, capture_attempt
        self.active = (observation, capture_attempt)

    def observe_round(self, action, success, cycle_seconds, switch_wall_seconds):
        if (
            self.active is None
            or action not in self.stats
            or not all(math.isfinite(x) and x >= 0 for x in (cycle_seconds, switch_wall_seconds))
            or switch_wall_seconds > cycle_seconds
        ):
            raise ValueError("Invalid or duplicate chosen-action outcome")
        observation, capture = self.active
        stat = self.stats[action]
        stat["n"] += 1
        stat["success"] += int(success)
        outcome = (action, cycle_seconds, switch_wall_seconds)
        if capture == self.attempt:
            self._cost_feedback(observation, capture, outcome)
        else:
            self.pending[self.attempt] = outcome
        self.active = None

    def observe(self, *args, **kwargs):
        raise RuntimeError("Use begin_round/observe_round to match report timestamps")

    def cost_feedback_count(self):
        return sum(stat["cost_n"] for stat in self.stats.values())
