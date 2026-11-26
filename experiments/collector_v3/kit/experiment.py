"""Measured trace replay, paired OCP sampler and information bounds."""
import argparse
from hashlib import sha256
from itertools import combinations
from math import comb
import json
from pathlib import Path
import sys

import numpy as np
from model import Policy, Settings, SLA, LEFT, RIGHT, ARMS, EXTRA, CHANNELS

def panels(directory, split):
    prov=json.loads((directory/'provenance.json').read_text())
    days=prov['dates'][:1] if split=='dev' else prov['dates'][1:]
    blocks=[]
    for day in days:
        series=[]
        for mid in prov['measurements']:
            with np.load(directory/f'{day}_{mid}.npz') as data:
                expected=prov['probe_ids'][:32] if split=='dev' else prov['probe_ids'][32:]
                assert list(data['ids'])==expected
                series.append(data['rtt'])
        blocks.append(np.stack(series,axis=-1))
    allrtt=np.concatenate(blocks,axis=0)
    return np.stack(np.split(allrtt,allrtt.shape[1]//16,axis=1))


def contexts(tensors,split,k,repeats,tag=0):
    from adaptive_planning.ocp_randomness import select_clients
    panels,h,n,g=tensors.shape
    assert g==4
    # Independent Monte Carlo draws from the actual project's rejection shuffle.
    arr=np.empty((panels,repeats,h,k),np.int16)
    for p in range(panels):
        for r in range(repeats):
            for t in range(h):
                key=sha256(f'proof-collector/{split}/{p}/{r}/{t}/{tag}'.encode()).digest()
                arr[p,r,t]=select_clients(key,n,k)
    classes=np.resize(np.arange(len(SLA)),h)
    np.random.default_rng(804193 if split=='dev' else 181143).shuffle(classes)
    return arr,classes


def observed(values,chosen,members,c):
    runs=len(chosen)
    row=np.arange(runs)[:,None]
    a=values[row,members,LEFT[chosen,None]]
    b=values[row,members,RIGHT[chosen,None]]
    ok1=(a>0)&(a<=SLA[c])
    ok2=(b>0)&(b<=SLA[c])&EXTRA[chosen,None]
    x=np.stack((np.where(ok1,a,-1.),np.where(ok2,b,-1.)),axis=-1)
    return x,(ok1|ok2).all(axis=1)


def run(tensors,rosters,classes,budget,settings,seed=53,shuffle=False,return_trace=False, gateway_caps=None,fixed_prices=None,policy_class=Policy,policy_options=None):
    p,r,h,k=rosters.shape
    series=rosters.reshape(p*r,h,k)
    ix=np.repeat(np.arange(p),r)
    policy=policy_class(settings,p*r,tensors.shape[2],k,h,budget,seed,gateway_caps,fixed_prices,**(policy_options or {}))
    ys=[]; aa=[]
    for t,c in enumerate(classes):
        members=series[:,t]
        seen=np.roll(series,1,axis=1)[:,t] if shuffle else members
        # In sham, only current decision is blind; all updates use the authentic past identities.
        if shuffle and t==0:
            seen=np.tile(np.arange(k),(p*r,1))
        a=policy.decide(c,seen)
        if shuffle:
            # Remap immediately booked designs to the true cohort before learning (next round).
            for branch,g in enumerate((LEFT[a],RIGHT[a])):
                used=np.ones(p*r,bool) if branch==0 else EXTRA[a]
                policy.design[policy._index(seen,g,c)]-=used[:,None]
                policy.design[policy._index(members,g,c)]+=used[:,None]
            policy.last=(c,members.copy(),a.copy(),None)
        samples,y=observed(tensors[ix,t],a,members,c)
        policy.observe(samples,y)
        ys.append(y);aa.append(a)
    outcome=np.stack(ys,axis=-1).reshape(p,r,h)
    actions=np.stack(aa,axis=-1).reshape(p,r,h)
    if return_trace:
        return outcome,actions
    return outcome.sum(axis=-1),EXTRA[actions].sum(axis=-1)


def bounds(tensors,rosters,classes,budget):
    p,r,h,k=rosters.shape
    all_sets=np.array(list(combinations(range(tensors.shape[2]),k)))
    tally={k:[] for k in ('blind','separated','singles','full','base_realized','full_realized')}
    for g in range(p):
        timely=(tensors[g]>0)&(tensors[g]<=SLA[classes,None,None])
        by_arm=(timely[:,:,LEFT]|timely[:,:,RIGHT])
        # Every possible certified roster, exactly once, for every slot.
        mask=by_arm[:,all_sets,:].all(axis=2)
        single=mask[:,:,:CHANNELS]
        pair=mask[:,:,CHANNELS:]
        e0=single.any(axis=2).mean(axis=1);e1=pair.any(axis=2).mean(axis=1)
        blind0=single.mean(axis=1).max(axis=1);blind1=pair.mean(axis=1).max(axis=1)
        def top(a,b):
            delta=np.maximum(0,b-a)
            return a.sum()+(np.sort(delta)[-budget:].sum() if budget else 0.)
        tally['blind'].append(top(blind0,blind1))
        tally['separated'].append(top(e0,e1))
        tally['singles'].append(e0.sum())
        # This is an upper bound on perfect joint decisions in expectation.
        tally['full'].append(e0.sum()+min(budget,(e1-e0).sum()))
        rows=by_arm[np.arange(h)[None,:,None],rosters[g]].all(axis=2)
        base=rows[:,:,:CHANNELS].any(axis=2).sum(axis=1)
        added=(rows.any(axis=2)&~rows[:,:,:CHANNELS].any(axis=2)).sum(axis=1)
        tally['base_realized'].append(base.tolist())
        tally['full_realized'].append((base+np.minimum(budget,added)).tolist())
    return tally
