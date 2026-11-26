"""Bursty, client-attributed fault traces for verified rounds with several remedies.

Three fault processes act on a certified OCP-FL roster:

* repairable lane outage: the primary aggregation lane fails whenever a client
  of group A_r is in the roster; the backup lane still works (rescued by R);
* common-cause outage: both lanes fail whenever a client of group A_c is in the
  roster (no remedy);
* proof outage: a client of group A_m cannot answer the projection challenge
  (rescued by full recomputation B).

Each process is a two-state Markov chain with stationary on-probability pi and
persistence phi (probability to copy the previous state; otherwise the state is
redrawn from the stationary law). Rosters come from the OCP-FL sampler on
pseudorandom inputs, so they are independent of the fault processes.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .ocp_randomness import select_clients
from .tcp_runtime import Fault

ACTION_COSTS = {"F+P": 1.0, "F+B": 2.0, "R+P": 2.0, "R+B": 3.0}
# Application traffic per round relative to F+P, measured on the TCP stand
# (four certified workers, journals of 3 Nov 2026): 1.00, 1.42, 1.55, 1.97.
ACTION_TRAFFIC = {"F+P": 1.0, "F+B": 1.4, "R+P": 1.55, "R+B": 1.95}


@dataclass(frozen=True)
class WorldSpec:
    population: int = 6
    sample_size: int = 2
    horizon: int = 480
    budget_ratio: float = 1.25
    repairable: tuple = (0.10, 0.85)
    common: tuple = (0.10, 0.85)
    proof: tuple = (0.10, 0.85)
    group_sizes: tuple = (1, 1, 1)
    switch_at: int | None = None
    backup_failure: float = 0.0
    full_update_failure: float = 0.0
    # Drifting remedy reliability: (stationary probability of a degraded regime,
    # persistence, failure probability while degraded). None keeps i.i.d. noise.
    backup_regime: tuple | None = None
    upload_regime: tuple | None = None
    reassign_every: int | None = None
    traffic_ratio: float | None = None
    extra: dict = field(default_factory=dict)


def markov_trace(rng, horizon, pi, phi):
    state = rng.random() < pi
    states = []
    for _ in range(horizon):
        if rng.random() >= phi:
            state = rng.random() < pi
        states.append(bool(state))
    return np.array(states)


@dataclass
class World:
    spec: WorldSpec
    seed: int
    rosters: list
    faults: list
    indicators: list
    groups: dict
    latent: dict = field(default_factory=dict)

    def trace_config(self):
        """Frozen latent trace for the live TCP stand (roster-independent)."""
        return dict(
            repairable=[bool(v) for v in self.latent["repairable"]],
            common=[bool(v) for v in self.latent["common"]],
            proof=[bool(v) for v in self.latent["proof"]],
            backup_noise=[bool(v) for v in self.latent["backup_noise"]],
            upload_noise=[bool(v) for v in self.latent["upload_noise"]],
            schedule=[[list(map(int, g)) for g in groups] for groups in self.groups["schedule"]],
        )

    @property
    def budget(self):
        return int(math.floor(self.spec.budget_ratio * self.spec.horizon + 1e-9))

    def accepted(self, t, action):
        E, S, M, U = self.indicators[t]
        lane_ok = (not E) or (action[0] == "R" and not S)
        proof_ok = (not U) if action[-1] == "B" else (not M)
        return lane_ok and proof_ok

    @property
    def traffic_budget(self):
        if self.spec.traffic_ratio is None:
            return None
        return self.spec.traffic_ratio * self.spec.horizon

    def oracle2(self, costs=ACTION_COSTS, traffic=ACTION_TRAFFIC):
        """Exact clairvoyant optimum with credit and traffic budgets (MILP)."""
        if self.traffic_budget is None:
            return self.oracle(costs)
        from scipy.optimize import Bounds, LinearConstraint, milp

        actions = list(costs)
        T, K = self.spec.horizon, len(actions)
        value = np.array([[float(self.accepted(t, a)) for a in actions] for t in range(T)])
        c = -value.ravel()
        rows = np.zeros((T, T * K))
        for t in range(T):
            rows[t, t * K : (t + 1) * K] = 1
        credit = np.tile([costs[a] for a in actions], T)
        load = np.tile([traffic[a] for a in actions], T)
        constraints = [
            LinearConstraint(rows, 1, 1),
            LinearConstraint(credit[None, :], -np.inf, self.budget),
            LinearConstraint(load[None, :], -np.inf, self.traffic_budget),
        ]
        result = milp(c, constraints=constraints, integrality=np.ones(T * K), bounds=Bounds(0, 1))
        return int(round(-result.fun))

    def oracle(self, costs=ACTION_COSTS):
        """Clairvoyant maximum of accepted rounds on the realized trace."""
        base = 0
        extras = []
        cheapest = min(costs.values())
        for t in range(self.spec.horizon):
            options = [c for a, c in costs.items() if self.accepted(t, a)]
            if not options:
                continue
            extra = min(options) - cheapest
            if extra == 0:
                base += 1
            else:
                extras.append(extra)
        spare = self.budget - self.spec.horizon * cheapest
        rescued = 0
        for extra in sorted(extras):
            if extra <= spare:
                spare -= extra
                rescued += 1
        return base + rescued


def make_world(spec, seed):
    rng = np.random.default_rng(seed)
    n, k, T = spec.population, spec.sample_size, spec.horizon
    order = [int(i) for i in rng.permutation(n)]
    sizes = spec.group_sizes
    if sum(sizes) > n:
        raise ValueError("Groups exceed population")
    A_r = order[: sizes[0]]
    A_c = order[sizes[0] : sizes[0] + sizes[1]]
    A_m = order[sizes[0] + sizes[1] : sum(sizes)]
    rep = markov_trace(rng, T, *spec.repairable)
    com = markov_trace(rng, T, *spec.common)
    prf = markov_trace(rng, T, *spec.proof)
    backup_noise = rng.random(T) < spec.backup_failure
    upload_noise = rng.random(T) < spec.full_update_failure
    schedule = []
    current = (A_r, A_c, A_m)
    for t in range(T):
        if spec.reassign_every and t > 0 and t % spec.reassign_every == 0:
            order = [int(i) for i in rng.permutation(n)]
            current = (
                order[: sizes[0]],
                order[sizes[0] : sizes[0] + sizes[1]],
                order[sizes[0] + sizes[1] : sum(sizes)],
            )
        groups = current
        if spec.switch_at is not None and t >= spec.switch_at:
            groups = (current[1], current[0], current[2])
        schedule.append(groups)
    if spec.backup_regime is not None:
        pi_bad, phi, p_bad = spec.backup_regime
        degraded = markov_trace(rng, T, pi_bad, phi)
        backup_noise = np.where(degraded, rng.random(T) < p_bad, backup_noise)
    if spec.upload_regime is not None:
        pi_bad, phi, p_bad = spec.upload_regime
        degraded_upload = markov_trace(rng, T, pi_bad, phi)
        upload_noise = np.where(degraded_upload, rng.random(T) < p_bad, upload_noise)
    roster_rng = np.random.default_rng(seed + 40100)
    rosters = [select_clients(roster_rng.bytes(32), n, k) for _ in range(T)]
    faults, indicators = [], []
    for t, members in enumerate(rosters):
        groups_r, groups_c, groups_m = schedule[t]
        hit_r = rep[t] and not set(members).isdisjoint(groups_r)
        hit_c = com[t] and not set(members).isdisjoint(groups_c)
        missing = tuple(i for i in members if prf[t] and i in groups_m)
        E = bool(hit_r or hit_c)
        S = bool(hit_c or (hit_r and backup_noise[t]))
        M = bool(missing)
        U = bool(missing and upload_noise[t])
        edges = (0, 2) if S else (0,) if E else ()
        faults.append(
            Fault(
                failed_edges=edges,
                missing_projection=missing,
                missing_full_update=missing if U else (),
            )
        )
        indicators.append((E, S, M, U))
    return World(
        spec,
        seed,
        rosters,
        faults,
        indicators,
        dict(repairable=A_r, common=A_c, proof=A_m, schedule=schedule),
        dict(
            repairable=rep,
            common=com,
            proof=prf,
            backup_noise=backup_noise,
            upload_noise=upload_noise,
        ),
    )
