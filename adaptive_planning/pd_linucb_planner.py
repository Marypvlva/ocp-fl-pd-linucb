"""Map real verified-training decisions onto the article's bounded-reward bandit."""

import heapq
import math

import numpy as np

from .adaptive_scheduler import ACTIONS
from .ocp_roster_planning import RosterBenefitControl, RosterRiskControl, roster_features
from .pd_linucb import PrimalDualLinUCB

DEFAULT_COSTS = {"F+B": 2.0, "F+P": 1.0, "R+B": 3.0, "R+P": 2.0}
CONTEXT_NAMES = (
    "intercept",
    "workers_fraction",
    "dimension_fraction",
    "model_rate_log",
    "model_unlimited",
    "edge_rate_log",
    "edge_unlimited",
    "verifier_rate_log",
    "verifier_unlimited",
    "latency_log",
    "previous_edge_error",
    "previous_projection_missing",
    "telemetry_age_fraction",
    "previous_FB",
    "previous_FP",
    "previous_RB",
    "previous_RP",
)


def outcome_reward(success, cycle_seconds, mode="success", reference=0.1):
    if type(success) not in (bool, int) or success not in (0, 1):
        raise ValueError("Success must be a binary protocol outcome")
    if (
        mode not in ("success", "throughput")
        or not math.isfinite(cycle_seconds)
        or cycle_seconds < 0
    ):
        raise ValueError("Invalid reward outcome")
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Invalid reward reference")
    return float(bool(success)) * (
        1.0 if mode == "success" else min(1.0, reference / max(cycle_seconds, reference))
    )


def make_context(observation, previous, age):
    if previous is not None and previous not in ACTIONS:
        raise ValueError("Unknown previous action")
    if type(age) is not int or not 0 <= age <= 20:
        raise ValueError("Invalid sensor age")
    rates = (observation.model_mbps, observation.edge_mbps, observation.verifier_mbps)
    values = (observation.workers, observation.dimension, *rates, observation.latency_ms)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Invalid observable context")
    x = [1.0, min(1.0, observation.workers / 16), min(1.0, observation.dimension / 150010)]
    for rate in rates:
        x.extend((math.log1p(min(10000.0, rate)) / math.log1p(10000.0), float(rate == 0)))
    x += [
        math.log1p(min(10000.0, observation.latency_ms)) / math.log1p(10000.0),
        float(observation.edge_trouble),
        float(observation.projection_missing),
        age / 20,
    ]
    x += [float(previous == a) for a in ACTIONS]
    return np.array(x, dtype=np.float64)


class PDLinUCBPlanner:
    """TCP adapter with explicit artificial label delay; sensor delay is separate.

    Success reward is exactly 1 for an accepted update, otherwise 0. The optional
    throughput reward is accepted * min(1, reference_seconds / observed_cycle).
    Both remain in [0,1]; only the former is the article's binary reward mapping.
    """

    def __init__(
        self,
        *,
        budget,
        horizon,
        costs=None,
        alpha=1.0,
        ridge=1.0,
        eta=0.05,
        feedback_delay=0,
        reward_mode="success",
        reward_reference_seconds=0.1,
        action_schedule=None,
        available_actions=None,
        selection_rule="ucb",
        roster_population=None,
        fund_horizon=False,
    ):
        if type(feedback_delay) is not int or not 0 <= feedback_delay <= 1000:
            raise ValueError("Invalid reward delay")
        if (
            reward_mode not in ("success", "throughput")
            or not math.isfinite(reward_reference_seconds)
            or reward_reference_seconds <= 0
        ):
            raise ValueError("Invalid reward mapping")
        costs = DEFAULT_COSTS if costs is None else costs
        if set(costs) != set(ACTIONS):
            raise ValueError("All four action costs are required")
        actions = ACTIONS if available_actions is None else available_actions
        if (
            not isinstance(actions, (list, tuple))
            or not actions
            or any(type(a) is not str or a not in ACTIONS for a in actions)
            or len(set(actions)) != len(actions)
        ):
            raise ValueError(
                "Available actions must be a nonempty distinct subset of the four modes"
            )
        if roster_population is not None and (
            type(roster_population) is not int or not 2 <= roster_population <= 16
        ):
            raise ValueError("Invalid roster context population")
        self.roster_population = roster_population
        dimension = len(CONTEXT_NAMES) + (roster_population or 0)
        self.engine = PrimalDualLinUCB(
            dimension, {a: costs[a] for a in actions}, budget, horizon, alpha, ridge, eta
        )
        self.ledger = self.engine.ledger
        if selection_rule not in (
            "ucb",
            "reactive_edge",
            "reactive_any",
            "risk_roster",
            "benefit_roster",
        ):
            raise ValueError("Unknown selection rule")
        if type(fund_horizon) is not bool or fund_horizon and selection_rule != "ucb":
            raise ValueError("The PD horizon guard is only available for UCB")
        if (selection_rule != "ucb" or fund_horizon) and (
            set(actions) != {"F+P", "R+P"}
            or selection_rule != "ucb"
            and eta != 0
            or action_schedule is not None
            or self.ledger.costs["R+P"] <= self.ledger.costs["F+P"]
            or self.ledger.budget < horizon * self.ledger.costs["F+P"]
        ):
            raise ValueError(
                "Budgeted reserve policies require two projection arms and a funded horizon"
            )
        if selection_rule in ("risk_roster", "benefit_roster") and roster_population is None:
            raise ValueError("The roster controls require roster observations")
        if selection_rule == "benefit_roster" and reward_mode != "success":
            raise ValueError("The reserve benefit control requires binary feedback")
        self.fund_horizon = fund_horizon
        self.risk_control = RosterRiskControl() if selection_rule == "risk_roster" else None
        self.benefit_control = (
            RosterBenefitControl() if selection_rule == "benefit_roster" else None
        )
        self.benefit_context = None
        self.selection_rule = selection_rule
        self.schedule = None
        if action_schedule is not None:
            if (
                not isinstance(action_schedule, (list, tuple))
                or len(action_schedule) != horizon
                or any(type(a) is not str or a not in self.ledger.costs for a in action_schedule)
                or sum(self.ledger.costs[a] for a in action_schedule) > self.ledger.budget
                or eta != 0
            ):
                raise ValueError("Schedule must cover the horizon within budget with eta=0")
            self.schedule = tuple(action_schedule)
        self.delay = feedback_delay
        self.reward_mode = reward_mode
        self.reference = reward_reference_seconds
        self.queue = []
        self.age = 0
        self.attempt = 0
        self.active = None
        self.ready = False
        self.last_decision = {}
        self.last_reward = None
        self.released_ids = []

    def _release(self, step):
        while self.queue and self.queue[0][0] <= step:
            _, identity, reward = heapq.heappop(self.queue)
            self.engine.reward(identity, reward)
            if self.benefit_control is not None:
                self.benefit_control.release(identity)
            self.released_ids.append(identity)

    def begin_round(self, observation, capture_attempt, attempt):
        if (
            type(attempt) is not int
            or attempt != self.ledger.steps + 1
            or self.active is not None
            or self.ready
            or type(capture_attempt) is not int
            or not 1 <= capture_attempt <= attempt
        ):
            raise ValueError("Noncausal or unfinished decision")
        self.age = attempt - capture_attempt
        if self.age > 20 or self.ledger.stop_reason():
            raise ValueError("Cannot begin an unavailable decision")
        self.attempt = attempt
        self.ready = True
        self.released_ids = []
        self._release(attempt)

    def choose(self, observation, previous, *, selected=None):
        if self.active is not None or not self.ready or self.attempt != self.ledger.steps + 1:
            raise ValueError("begin_round must precede one selection")
        context = make_context(observation, previous, self.age)
        membership = None
        if self.roster_population is not None:
            membership = roster_features(selected, self.roster_population, observation.workers)
            context = np.concatenate((context, membership))
        forced_action = None if self.schedule is None else self.schedule[self.attempt - 1]
        rule = self.selection_rule
        benefit_decision = None
        if rule != "ucb" or self.fund_horizon:
            left = self.ledger.horizon - self.ledger.steps
            cheap, reserve = (self.ledger.costs[a] for a in ("F+P", "R+P"))
            slots = (self.ledger.remaining - left * cheap) // (reserve - cheap)
            if self.fund_horizon:
                if slots == 0:
                    forced_action, rule = "F+P", "fund_horizon"
            else:
                if self.risk_control is not None:
                    trouble = self.risk_control.prefer_reserve(
                        observation, membership, slots / left
                    )
                elif self.benefit_control is not None:
                    trouble, self.benefit_context, benefit_decision = self.benefit_control.score(
                        observation, membership, slots / left
                    )
                else:
                    trouble = observation.edge_trouble or (
                        rule == "reactive_any" and observation.projection_missing
                    )
                forced_action = "R+P" if slots > 0 and (trouble or slots >= left) else "F+P"
        action = self.engine.select(context, forced_action=forced_action)
        if action is not None:
            if rule != "ucb":
                self.engine.last_decision["selection_rule"] = rule
            if benefit_decision is not None:
                self.engine.last_decision["benefit_estimate"] = benefit_decision
            self.ready = False
            self.active = (self.ledger.steps, action)
            self.last_decision = self.engine.last_decision.copy()
        return action

    def observe_round(
        self, action, success, cycle_seconds, switch_wall_seconds, *, primary_edge_error=None
    ):
        if (
            self.active is None
            or action != self.active[1]
            or not math.isfinite(cycle_seconds)
            or cycle_seconds < 0
            or not math.isfinite(switch_wall_seconds)
            or not 0 <= switch_wall_seconds <= cycle_seconds
        ):
            raise ValueError("Invalid or duplicate selected-action outcome")
        identity, _ = self.active
        reward = outcome_reward(success, cycle_seconds, self.reward_mode, self.reference)
        if self.benefit_control is not None:
            if type(primary_edge_error) is not bool:
                raise ValueError("Reserve benefit feedback requires an observed primary error flag")
            self.benefit_control.pending[identity] = (
                self.benefit_context,
                action,
                primary_edge_error,
                bool(success),
            )
            self.benefit_context = None
        self.last_reward = reward
        heapq.heappush(self.queue, (identity + self.delay, identity, reward))
        self._release(identity)
        self.active = None

    def feedback_count(self):
        return int(self.engine.observed.sum())

    @property
    def stats(self):
        return dict(
            **self.engine.state(),
            released_reward_ids=list(self.released_ids),
            queued_rewards=len(self.queue),
        )
