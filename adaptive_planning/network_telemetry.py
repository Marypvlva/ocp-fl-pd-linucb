"""Causal, synthetic telemetry perturbations; physical pacing remains separate."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TelemetryMode:
    name: str
    delay_rounds: int = 0
    log_sigma: float = 0.0

    def __post_init__(self):
        if (
            type(self.delay_rounds) is not int
            or not 0 <= self.delay_rounds <= 20
            or type(self.log_sigma) not in (int, float)
            or not math.isfinite(self.log_sigma)
            or not 0 <= self.log_sigma <= 2
        ):
            raise ValueError("Invalid telemetry settings")


TELEMETRY_MODES = {
    "exact": TelemetryMode("exact"),
    "noise": TelemetryMode("noise", log_sigma=0.35),
    "stale": TelemetryMode("stale", delay_rounds=2),
    "stale_noise": TelemetryMode("stale_noise", delay_rounds=2, log_sigma=0.35),
}


@dataclass(frozen=True)
class RateReport:
    capture_attempt: int
    source_profile: str
    model_mbps: float
    edge_mbps: float
    verifier_mbps: float
    latency_ms: float


class NetworkTelemetry:
    """One capture per attempt, then delivery from the already captured prefix.

    Zero retains its protocol meaning 'unlimited configured flow'. Noise perturbs
    positive magnitudes, not that categorical flag. Initial delivery uses the
    first startup observation until the delay pipeline is full.
    """

    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.reports = []

    def observe(self, actual_profile, attempt):
        if type(attempt) is not int or attempt != len(self.reports) + 1:
            raise ValueError("Telemetry must consume the causal attempt prefix")
        z = self.rng.normal(size=4)  # common random numbers across all sensor modes
        factors = np.exp(np.clip(self.mode.log_sigma * z - 0.5 * self.mode.log_sigma**2, -20, 20))
        values = [
            actual_profile.model_mbps,
            actual_profile.edge_mbps,
            actual_profile.verifier_mbps,
            actual_profile.latency_ms,
        ]
        observed = []
        for i, (value, factor) in enumerate(zip(values, factors)):
            if not value or not self.mode.log_sigma:
                observed.append(float(value))
            else:
                observed.append(float(np.clip(value * factor, 0.001 if i == 3 else 1.0, 10000.0)))
        self.reports.append(RateReport(attempt, actual_profile.name, *observed))
        source = max(0, attempt - 1 - self.mode.delay_rounds)
        return self.reports[source]
