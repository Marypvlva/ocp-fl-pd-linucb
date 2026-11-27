"""Timestamp, censoring, geometry and immutable-resource regression checks."""
from dataclasses import replace
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/collector_v3/kit'))
from model import Policy, Settings
from memory_policy import MemoryPolicy
from policy_factory import make_policy


class MemoryTests(unittest.TestCase):
    def make(self, **settings):
        return MemoryPolicy(Settings(**dict(dict(delay=2, memory='discount'),**settings)),
                            1,4,2,12,6,53,np.full((1,4),12))

    def send(self,p,c=0,arm=0,success=True,mask=True,horizon=None):
        p.decide(c,np.array([[0,1]]),forced_arm=arm)
        p.observe(np.full((1,2,2),50. if success else -1.),np.array([success]),
                  np.full((1,2,2),mask),observation_horizon_ms=horizon)

    def test_generation_age_not_arrival_age_and_squared_moments(self):
        p=self.make(discount=.5)
        self.send(p)
        self.assertEqual(p.done.sum(),0)
        self.send(p)
        self.assertEqual(p.done.sum(),0)
        p.decide(0,np.array([[0,1]]),forced_arm=0)
        self.assertEqual(p.done[0,0,0,0],.25)
        self.assertEqual(p.done2[0,0,0,0],.0625)
        self.assertEqual(p.age_mass[0,0,0,0],.5)
        self.assertEqual(p.design[0,0,0,0],1.75)
        d=p.last_decision['memory']
        self.assertAlmostEqual(d['path_bonus'][0][0][0],p.s.alpha*np.sqrt(1+.3125)/1.75)

    def test_gamma_one_and_long_window_reproduce_legacy_exactly(self):
        for memory in ('discount','window'):
            s=Settings(alpha=.15,eta=.2,delay=2)
            old=Policy(s,1,4,2,12,3,53,np.array([[12,12,4,4]]))
            new=MemoryPolicy(replace(s,memory=memory,discount=1.,window=100),
                              1,4,2,12,3,53,np.array([[12,12,4,4]]))
            rng=np.random.default_rng(513)
            for t in range(12):
                roster=np.array([rng.choice(4,2,replace=False)])
                c=t%3
                a,b=old.decide(c,roster),new.decide(c,roster)
                np.testing.assert_array_equal(a,b)
                for attr in ('done','design','wins','price','gprice','spent','gateway_spent'):
                    np.testing.assert_allclose(getattr(old,attr),getattr(new,attr),atol=1e-12)
                obs=np.where(rng.random((1,2,2))<.5,50.,-1.)
                mask=rng.random((1,2,2))<.8
                old.observe(obs,np.array([False]),mask)
                new.observe(obs,np.array([False]),mask)

    def test_expired_late_feedback_never_resurrects_window(self):
        p=self.make(memory='window',window=1,delay=3)
        for _ in range(4):self.send(p)
        self.assertEqual(p.done.sum(),0)
        self.assertEqual(p.expired_feedback,1)
        self.assertEqual(len(p.events),2)  # previous slot plus current booked action
        self.assertEqual(p.gateway_spent[0,0],4)

    def test_eligibility_does_not_pool_positive_only_longer_horizons(self):
        p=self.make(memory='none',feedback_rule='eligible',delay=0)
        self.send(p,c=0)
        p.decide(1,np.array([[0,1]]),forced_arm=0)
        self.assertEqual(p.done[0,0,0].tolist(),[1,0,0])
        self.assertEqual(p.wins[0,0,0].tolist(),[1,0,0])

    def test_truncated_horizon_and_unexecuted_paths_are_unknown(self):
        for mask,cutoff in ((False,None),(True,70.)):
            p=self.make(feedback_rule='eligible',delay=0)
            self.send(p,mask=mask,horizon=cutoff)
            p.decide(0,np.array([[0,1]]),forced_arm=0)
            self.assertEqual(p.done.sum(),0)
            self.assertEqual(p.design.sum(),2)  # only new pending design
            self.assertEqual(p.gateway_spent[0,0],2)

    def test_forgetting_never_refunds_mirror_or_site_quotas(self):
        p=self.make(memory='window',window=1,delay=0)
        for _ in range(6):self.send(p,arm=4)
        self.assertEqual(p.spent[0],6)
        with self.assertRaises(ValueError):p.decide(0,np.array([[0,1]]),forced_arm=4)
        self.assertEqual(p.spent[0],6)

    def test_no_future_label_leak(self):
        p,q=self.make(),self.make()
        for i in range(2):
            a=p.decide(0,np.array([[0,1]]));b=q.decide(0,np.array([[0,1]]))
            np.testing.assert_array_equal(a,b)
            self.assertEqual(p.last_decision,q.last_decision)
            p.observe(np.full((1,2,2),50.),np.array([True]))
            q.observe(np.full((1,2,2),-1.),np.array([False]))

    def test_window_statistics_equal_explicit_timestamp_sums(self):
        for memory in ('window','discount'):
            p=self.make(memory=memory,window=3,discount=.8,delay=2)
            labels=[]
            for t in range(10):
                p.decide(0,np.array([[0,1]]),forced_arm=0)
                eligible=[s for s in labels if s+2<=t and (memory!='window' or s>=t-3)]
                ws=[.8**(t-s) if memory=='discount' else 1 for s in eligible]
                self.assertAlmostEqual(p.done[0,0,0,0],sum(ws))
                self.assertAlmostEqual(p.done2[0,0,0,0],sum(w*w for w in ws))
                self.assertAlmostEqual(p.wins[0,0,0,0],sum(w for w,s in zip(ws,eligible) if s%2==0))
                p.observe(np.full((1,2,2),50. if t%2==0 else -1.),np.array([t%2==0]))
                labels.append(t)

    def test_validation_and_feedback_exactly_once(self):
        for options in ({'discount':0},{'discount':float('nan')},{'window':0},{'memory_bonus':'bad'}):
            with self.assertRaises(ValueError):self.make(**options)
        p=self.make()
        with self.assertRaises(ValueError):p.observe(np.zeros((1,2,2)),np.array([True]))
        p.decide(0,np.array([[0,1]]))
        with self.assertRaises(ValueError):p.decide(0,np.array([[0,1]]))
        with self.assertRaises(ValueError):p.observe(np.zeros((1,3,2)),np.array([True]))

    def test_factory_keeps_legacy_policy_untouched(self):
        self.assertIs(type(make_policy(Settings(),1,4,2,12,3)),Policy)
        self.assertIs(type(make_policy(Settings(memory='discount'),1,4,2,12,3)),MemoryPolicy)

    def test_caller_context_mutation_cannot_rewrite_delayed_geometry(self):
        p=self.make(discount=.5,delay=0)
        roster=np.array([[0,1]])
        p.decide(0,roster,forced_arm=0)
        roster[:]=[2,3]
        p.observe(np.full((1,2,2),50.),np.array([True]))
        p.decide(0,np.array([[0,1]]),forced_arm=0)
        np.testing.assert_allclose(p.done[0,:,0,0],[.5,.5,0,0])
        self.assertFalse(p.design[0,2:].any())

    def test_dual_update_uses_remaining_budget_before_charge(self):
        p=self.make(eta=.2,delay=0)
        p.caps=np.array([[12,12,3,3]])
        p.decide(0,np.array([[0,1]]),forced_arm=5)  # gateways 0,2
        self.assertAlmostEqual(p.price[0],.25+.2*(1-6/12))
        np.testing.assert_allclose(p.gprice,[[0,0,.2*(1-3/12),0]])
        p.observe(np.full((1,2,2),-1.),np.array([False]))
        price=p.price.copy();gp=p.gprice.copy()
        p.decide(0,np.array([[0,1]]),forced_arm=1)
        self.assertAlmostEqual(p.price[0],max(0,price[0]-.2*5/11))
        np.testing.assert_allclose(p.gprice,np.maximum(0,gp+.2*(np.array([[0,1,0,0]])-np.array([[11,12,2,3]])/11)))

    def test_memory_variants_live_loopback_and_exact_causal_replay(self):
        code='''
import json,sys
from pathlib import Path
from distributed_live import plan,run
from program import freeze,verify_complete
from replay_program import reconstruct
root=Path.cwd();tmp=Path(sys.argv[1]);placement=tmp/'placement.json'
placement.write_text(json.dumps(dict(workers=['local'],edges=['local']*4)))
plan(root,placement,tmp/'base','pilot',rounds=4,seeds=[81101],modes=['joint'],windows=[1000,1500,2000])
config=freeze(tmp/'base/plan.json',dict(id='memory_technical_loopback',horizon=4,network_mode='natural'),
    tmp/'run','pilot',[902701],['joint_discounted','joint_window','joint_discounted_eligible'],
    policy_overrides={'joint_discounted':{'discount':.9},'joint_window':{'window':1}})
run(root,tmp/'run',dict(url='loopback',bind='127.0.0.1',port=0,network_mode='natural'))
for mode in config['modes']:
    episode=tmp/'run'/f'902701_{mode}'
    verify_complete(episode,config)
    assert len(reconstruct(tmp/'run',902701,mode))==4
    rows=[json.loads(p.read_text()) for p in sorted(episode.glob('round-*.json'))]
    assert all(r['policy_decision']['memory'] for r in rows)
    assert all(r['emulated_link_ms'] is None for r in rows)
    assert all(r['metrics']['collector_phase_budget_ms']==r['window_ms'] for r in rows if r['metrics']['projection_started'])
'''
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-c',code,tmp],cwd=ROOT,
                env=dict(os.environ,ADAPTIVE_CLOCK='wall',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                    PYTHONPATH=str(ROOT/'experiments/collector_v3/kit')+os.pathsep+str(ROOT)),
                capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout[-1000:]+result.stderr[-4000:])

if __name__=='__main__':unittest.main()
