"""A small causal circuit breaker for persistent protocol availability failures."""

from .adaptive_scheduler import predicted_seconds


class AvailabilityPlanner:
    """Remember an observed edge/P failure for four decisions, then probe again.

    The preferred healthy configuration is F+P. This deliberately simple rule
    tests temporal availability, not a learned universal cost model. It receives
    only the preceding round's observable errors, never the fault injector or
    the phase label. The cooldown is fixed before evaluation.
    """

    COOLDOWN = 4

    def __init__(self, seed, switch_seconds):
        self.switch_seconds = switch_seconds
        self.stats = dict(attempt=0, feedback=0, edge_block_until=0, projection_block_until=0)
        self.last_decision = {}
        self.action = None

    def choose(self, observation, previous):
        if self.stats["attempt"] != self.stats["feedback"]:
            raise ValueError("Missing chosen-action feedback")
        attempt = self.stats["attempt"] + 1
        self.stats["attempt"] = attempt
        if observation.edge_trouble:
            self.stats["edge_block_until"] = attempt + self.COOLDOWN - 1
        if observation.projection_missing:
            self.stats["projection_block_until"] = attempt + self.COOLDOWN - 1
        placement = "R" if attempt <= self.stats["edge_block_until"] else "F"
        method = "B" if attempt <= self.stats["projection_block_until"] else "P"
        self.action = placement + "+" + method
        self.last_decision = dict(
            attempt=attempt,
            edge_block_until=self.stats["edge_block_until"],
            projection_block_until=self.stats["projection_block_until"],
            predicted_seconds=predicted_seconds(
                self.action, observation, previous, self.switch_seconds
            ),
        )
        return self.action

    def observe(self, observation, action, success, cycle_seconds, switch_wall_seconds):
        if self.stats["attempt"] != self.stats["feedback"] + 1 or action != self.action:
            raise ValueError("Invalid or duplicate chosen-action feedback")
        self.stats["feedback"] += 1

    def feedback_count(self):
        return self.stats["feedback"]
