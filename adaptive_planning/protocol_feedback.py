"""Causal feedback from stages actually executed by the verification protocol."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProtocolFeedback:
    primary_edge_error: bool | None = None
    backup_edge_error: bool | None = None
    projection_missing: bool | None = None
    full_update_missing: bool | None = None
    compute_ok: bool | None = None
    lane_ok: bool | None = None
    verification_ok: bool | None = None

    def planner_arguments(self):
        fields = asdict(self)
        stages = {name: fields.pop(name) for name in ("compute_ok", "lane_ok", "verification_ok")}
        return fields | {"stage_observations": stages}


def feedback_from_round(action, accepted, errors, metrics):
    """An unexecuted stage supplies no label, including after a preceding failure.

    New journals explicitly record stage entry and completion. Historical
    journals can infer an aggregate attempt from its measured RPC duration,
    but cannot infer whether worker computation completed before a deadline.
    """
    aggregate_started = metrics.get("aggregate_started")
    if aggregate_started is None:
        aggregate_started = metrics.get("edge_rpc_wall_s", 0.0) > 0.0
    primary = backup = None
    if aggregate_started:
        primary = any(error.startswith(("e0:", "e1:")) for error in errors)
        if action[0] == "R":
            backup = any(error.startswith(("e2:", "e3:")) for error in errors)
            if metrics.get("feedback_schema", 2) >= 3:
                # A surviving copy of each fragment can mix primary and reserve replicas.
                # S means failed recovery conditional on primary failure.
                backup = bool(primary and not metrics["aggregate_completed"])
    verification_started = metrics.get("verification_attempts", 0) > 0
    projection = None
    projection_started = metrics.get("projection_started", verification_started)
    if action[-1] == "P" and projection_started:
        projection = any("Missing projection participant" in error for error in errors)
    compute = metrics.get("compute_completed")
    if not metrics.get("compute_started", aggregate_started):
        compute = None
    elif compute is None and aggregate_started:
        compute = True
    uploaded = None
    if action[-1] == "B" and compute is not None:
        uploaded = not bool(compute)
    lane = metrics.get("aggregate_completed") if aggregate_started else None
    if lane is None and aggregate_started:
        lane = verification_started
    verified = metrics.get("verification_completed") if verification_started else None
    if verified is None and verification_started:
        verified = bool(accepted)
    return ProtocolFeedback(primary, backup, projection, uploaded, compute, lane, verified)
