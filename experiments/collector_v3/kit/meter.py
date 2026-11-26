"""Reconcile asynchronous sender telemetry with local or distributed durable credit."""

from time import sleep, perf_counter


def settled_metrics(cluster, round_id, before):
    budget = cluster.application_budget
    deadline = perf_counter() + getattr(budget, "settlement_grace_s", 1.0)
    differences = []
    while True:
        result = cluster.collect_metrics(round_id)
        state = result["app_byte_budget"]
        expected = state["spent"] - before
        actual = result["app_tx_bytes"]
        if not state["reserved"] and not result["telemetry_missing_nodes"] and actual == expected:
            return result, state
        differences.append((expected, actual, state["reserved"], result["telemetry_missing_nodes"]))
        if perf_counter() >= deadline:
            raise ValueError(
                f"Byte counters did not reconcile for round {round_id}: {differences[-8:]}"
            )
        sleep(0.05)
