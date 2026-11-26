"""Application-level pacing shared by all localhost processes.

Rates describe aggregate directed bottlenecks, not per-connection bandwidth.
This controls actual frame transmission; no delay is added to measured rounds later.
"""

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import math
from time import sleep
from .clock import now as monotonic

from .byte_budget import SharedByteBudget

CHUNK_BYTES = 16_384
MAX_DYNAMIC_ROUNDS = 65_536


@dataclass(frozen=True)
class NetworkProfile:
    name: str = "unlimited"
    model_mbps: float = 0.0
    edge_mbps: float = 0.0
    verifier_mbps: float = 0.0
    latency_ms: float = 0.0

    def __post_init__(self):
        for name in ("model_mbps", "edge_mbps", "verifier_mbps", "latency_ms"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid network setting: {name}")

    def to_dict(self):
        return asdict(self)


# MB/s means 10**6 bytes/s, not megabits/s. These settings are hypotheses to test.
PROFILES = {
    "unlimited": NetworkProfile(),
    "edge_limited": NetworkProfile("edge_limited", edge_mbps=40.0, latency_ms=2.0),
    "verifier_limited": NetworkProfile("verifier_limited", verifier_mbps=40.0, latency_ms=2.0),
}


def wait_until(when, deadline):
    now = monotonic()
    if now >= deadline:
        raise TimeoutError("Pacing deadline exceeded")
    if when > deadline:
        sleep(max(0.0, deadline - now))
        raise TimeoutError("Pacing deadline exceeded")
    sleep(max(0.0, when - now))


class SharedPacer:
    """Reserve serialization slots across spawned processes and their threads.

    Transmission follows each reserved slot. Scheduler jitter can bunch deliveries;
    this is a paced byte budget, not a packet-level WAN emulator. Failed writes can
    consume an already reserved slot, as explicitly reported in the documentation.
    """

    def __init__(self, context, bytes_per_second):
        self.rate = bytes_per_second
        self.next_free = context.Value("d", 0.0, lock=False)
        self.lock = context.Lock()

    def wait(self, size, deadline, *, rate=None):
        rate = self.rate if rate is None else rate
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid pacing rate")
        left = deadline - monotonic()
        if left <= 0 or not self.lock.acquire(timeout=max(0.0, left)):
            raise TimeoutError("Pacing queue deadline exceeded")
        try:
            end = max(monotonic(), self.next_free.value) + size / rate
            # Never reserve work beyond the request's deadline.
            if end <= deadline:
                self.next_free.value = end
        finally:
            self.lock.release()
        wait_until(end, deadline)


class NetworkShaper:
    def __init__(self, context, profile, *, dynamic=False, byte_limit=None):
        self.profile = profile
        self.byte_budget = SharedByteBudget(context, byte_limit) if byte_limit is not None else None
        self.dynamic = dynamic
        self.profile_names = tuple(PROFILES)
        self.cache = {}
        if dynamic:
            self.round_profiles = context.Array("b", [-1] * MAX_DYNAMIC_ROUNDS, lock=False)
            self.last_configured = context.Value("i", 0, lock=False)
            self.config_lock = context.Lock()
        self.pacers = {}
        for direction, setting in (
            ("model", profile.model_mbps),
            ("edge", profile.edge_mbps),
            ("verifier", profile.verifier_mbps),
        ):
            if setting or dynamic:
                self.pacers[direction] = SharedPacer(context, setting * 1e6 if setting else 1.0)

    @contextmanager
    def config_guard(self, deadline=None):
        # multiprocessing locks are not released when their owner is terminated.
        until = min(deadline if deadline is not None else float("inf"), monotonic() + 0.5)
        if not self.config_lock.acquire(timeout=max(0.0, until - monotonic())):
            raise TimeoutError("Network profile lock deadline exceeded")
        try:
            yield
        finally:
            self.config_lock.release()

    def configure_round(self, round_id, profile, *, deadline=None):
        """Launcher-only binding, immutable for this round including delayed RPCs."""
        if not self.dynamic or type(round_id) is not int or not 0 < round_id < MAX_DYNAMIC_ROUNDS:
            raise ValueError("Invalid dynamic network round")
        if profile.name not in PROFILES or profile != PROFILES[profile.name]:
            raise ValueError("Dynamic rounds require a registered profile")
        with self.config_guard(deadline):
            if round_id <= self.last_configured.value:
                raise ValueError("Network round already bound or stale")
            self.round_profiles[round_id] = self.profile_names.index(profile.name)
            self.last_configured.value = round_id

    def profile_for_round(self, round_id, *, deadline=None):
        if not self.dynamic or round_id == 0:
            return self.profile
        if round_id in self.cache:
            return self.cache[round_id]
        if type(round_id) is not int or not 0 < round_id < MAX_DYNAMIC_ROUNDS:
            raise ValueError("Invalid dynamic network round")
        with self.config_guard(deadline):
            code = self.round_profiles[round_id]
        if code < 0:
            raise ValueError("Network round is not bound")
        profile = PROFILES[self.profile_names[code]]
        self.cache[round_id] = profile
        return profile

    def rate_for(self, header):
        direction = self.direction(header)
        if direction is None:
            return 0.0
        profile = self.profile_for_round(
            header["round"], deadline=header.get("deadline_ns", 0) / 1e9 or None
        )
        return (
            getattr(
                profile,
                {"model": "model_mbps", "edge": "edge_mbps", "verifier": "verifier_mbps"}[
                    direction
                ],
            )
            * 1e6
        )

    def direction(self, header):
        if not header["round"]:
            return None  # management traffic does not contend with data-plane traffic
        if not header["response"] and header["kind"] == "compute":
            return "model"
        if header["response"] and header["kind"] == "fragment":
            return "edge"
        if header["response"] and header["kind"] in ("compute", "aggregate", "projection", "signed_recovery", "open_committed"):
            return "verifier"
        return None

    def latency(self, header, deadline):
        if header["round"]:
            latency = self.profile_for_round(header["round"], deadline=deadline).latency_ms
            if latency:
                wait_until(monotonic() + latency / 1000.0, deadline)

    def pace(self, header, size, deadline):
        rate = self.rate_for(header)
        if rate:
            self.pacers[self.direction(header)].wait(size, deadline, rate=rate)

    def limited(self, header):
        return self.rate_for(header) > 0
