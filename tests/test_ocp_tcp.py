import copy
import unittest
import numpy as np

from adaptive_planning.ocp_chain import OcpChain
from adaptive_planning.tcp_runtime import ACTIONS, Cluster, Fault, flatten_model, make_model
from adaptive_planning.training import Q_SCALE, make_workers


class SelectedTCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, _, _ = make_workers(4)
        cls.chain = OcpChain.local(4, 2)
        cls.cluster = Cluster(
            cls.groups, hidden=8, rpc_timeout=1.0, selection_authority=cls.chain
        ).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.cluster.close()

    def test_01_guard_blocks_missing_and_forged_schedule_before_tcp(self):
        c = self.cluster
        t = self.chain.run_local_ceremony(c.round_id + 1)
        m = make_model(16, 8, 0.5)
        initial = flatten_model(m)
        before = c.round_id
        with self.assertRaises(ValueError):
            c.run_round(m, "F+B")
        bad = copy.deepcopy(t)
        bad["selected"] = list(reversed(bad["selected"]))
        with self.assertRaises(ValueError):
            c.run_round(m, "F+B", selection=bad)
        self.assertEqual(c.round_id, before)
        self.assertEqual(flatten_model(m), initial)
        self.assertTrue(c.run_round(m, "F+B", selection=t).accepted)
        with self.assertRaises(ValueError):
            c.run_round(m, "F+B", selection=t)

    def test_02_all_modes_use_exact_selected_weighted_sum(self):
        c = self.cluster
        for action in ACTIONS:
            ticket = self.chain.run_local_ceremony(c.round_id + 1)
            selected = ticket["selected"]
            model = make_model(17, 8, 0.5)
            reference = make_model(17, 8, 0.5)
            total = sum(len(self.groups[i][1]) for i in selected)
            updates = [
                np.rint(
                    reference.gradient(*self.groups[i]) * (len(self.groups[i][1]) / total) * Q_SCALE
                ).astype(np.int64)
                for i in selected
            ]
            expected = np.stack(updates).sum(axis=0, dtype=np.int64)
            excluded = tuple(i for i in range(4) if i not in selected)
            result = c.run_round(
                model,
                action,
                selection=ticket,
                fault=Fault(missing_compute=excluded, missing_projection=excluded),
            )
            self.assertTrue(result.accepted, result.errors)
            np.testing.assert_array_equal(result.vector, expected)
            reference.update(expected)
            self.assertEqual(flatten_model(model), flatten_model(reference))
            self.assertEqual(result.metrics["selected_workers"], selected)
            self.assertEqual(result.metrics["selected_examples"], total)

    def test_03_missing_selected_worker_does_not_trigger_resampling(self):
        c = self.cluster
        ticket = self.chain.run_local_ceremony(c.round_id + 1)
        model = make_model(18, 8, 0.5)
        before = flatten_model(model)
        result = c.run_round(
            model, "R+B", selection=ticket, fault=Fault(missing_compute=(ticket["selected"][0],))
        )
        self.assertFalse(result.accepted)
        self.assertEqual(flatten_model(model), before)
        self.assertEqual(result.metrics["selected_workers"], ticket["selected"])


if __name__ == "__main__":
    unittest.main()
