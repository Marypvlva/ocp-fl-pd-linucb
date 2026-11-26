"""Wait for the asynchronous sender counters and reconcile them with the hard byte ledger."""
from time import sleep


def settled_metrics(cluster, round_id, before):
    differences=[]
    for delay in (0,.01,.03,.06,.15,.3):
        if delay:
            sleep(delay)
        result=cluster.collect_metrics(round_id)
        state=cluster.shaper.byte_budget.state()
        expected=state['spent']-before
        actual=result['app_tx_bytes']
        if not state['reserved'] and not result['telemetry_missing_nodes'] and actual==expected:
            return result,state
        differences.append((expected,actual,state['reserved'],result['telemetry_missing_nodes']))
    raise ValueError(f'Byte counters did not reconcile for round {round_id}: {differences}')
