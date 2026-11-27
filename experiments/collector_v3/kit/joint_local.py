"""Finite local development/validation benchmark. No cloud or counterfactual replay.

Synthetic complete latency tensors plus an unchanged public RIPE Atlas anchor.
Three candidates per family, fixed before validation; no validation re-tuning.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from model import Settings, SLA, ARMS
from policy_factory import make_policy
from analysis_program import table, write_json
from experiment import panels

FAMILIES=('joint','joint_discounted','joint_window','matched_recent','article_pd')
BASE=Settings(alpha=.15,eta=.2,initial_price=.25,ridge=1.,delay=2)


def grid():
    return dict(
        joint=[replace(BASE,alpha=a) for a in (.05,.15,.3)],
        joint_discounted=[replace(BASE,memory='discount',discount=g) for g in (.9,.95,.98)],
        joint_window=[replace(BASE,memory='window',window=w) for w in (16,32,64)],
        matched_recent=[replace(BASE,mode='recent',eta=e) for e in (.05,.1,.2)],
        article_pd=[replace(BASE,mode='article_pd',alpha=a,eta=.05,initial_price=0.) for a in (.5,1.,2.)])


def prepare(output):
    output=Path(output)
    if output.exists():raise ValueError('Do not overwrite a frozen manifest')
    spec=dict(version=1,horizon=120,population=4,k=2,windows_ms=[27.,35.,46.],
        cases=['stationary','abrupt','gradual','shared_host'],quota_fractions=[.1,.5],
        mirror_fraction=.1,dev_seeds=[171001,171002,171003,171004],
        validation_seeds=list(range(172001,172009)),
        candidate_budget=3,candidates={k:[asdict(s) for s in v] for k,v in grid().items()},
        primary='mean accepted fraction equally weighted over 8 predeclared synthetic cases',
        atlas='Secondary unchanged trace anchor; historical validation, not new confirmation',
        source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            Path(__file__).with_name('model.py'),Path(__file__).with_name('memory_policy.py'),
            Path(__file__).with_name('article_policy.py')]},
        scope='Local latency/coverage proxy, not real TCP, application bytes, EVM or scientific confirmation.')
    write_json(output,spec)
    return spec


def read_manifest(path):
    path=Path(path);spec=json.loads(path.read_text())
    for name,digest in spec['source_sha256'].items():
        if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source differs from prevalidation manifest: '+name)
    return spec,sha256(path.read_bytes()).hexdigest()


def world(spec,seed,case):
    """Exogenous full tensors generated before ANY learner decision."""
    h,n=spec['horizon'],spec['population']
    rng=np.random.default_rng(seed)
    roster=np.stack([rng.choice(n,spec['k'],replace=False) for _ in range(h)])
    classes=np.resize(np.arange(3),h);rng.shuffle(classes)
    # Fixed heterogeneous scales, randomly relabelled for every independent seed.
    scales=rng.uniform(12,55,(n,4))
    scales=scales[:,rng.permutation(4)]
    alternative=np.roll(scales,1,axis=1)
    z=np.zeros(h)
    if case=='abrupt':z[h//3:2*h//3]=1
    if case=='gradual':z=(1-np.cos(np.linspace(0,2*np.pi,h)))/2
    if case=='shared_host':z[h//3:2*h//3]=1
    varying=(1-z[:,None,None])*scales+z[:,None,None]*alternative
    values=3+rng.exponential(varying)
    # Context-dependent failure; the same worker appears in different cohorts.
    values[rng.random(values.shape)<.03]=-1.
    if case=='shared_host':
        busy=rng.random((h,2))<.25
        for g in range(4):values[:,:,g]+=busy[:,g//2,None]*40
    return values,roster,classes


def episode(settings,values,roster,classes,capfraction=.1,mirror=.1):
    h,n,_=values.shape;k=roster.shape[1]
    cap=np.full((1,4),int(h*capfraction));cap[0,:2]=h
    policy=make_policy(settings,1,n,k,h,int(h*mirror),53,cap)
    accepted=[];actions=[];predictions=[]
    for t,c in enumerate(classes):
        a=int(policy.decide(int(c),roster[t][None,:])[0])
        x=np.full((1,k,2),-1.)
        for j,g in enumerate(ARMS[a]):
            val=values[t,roster[t],g]
            x[0,:,j]=np.where((val>0)&(val<=SLA[c]),val,-1.)
        y=bool((x[0]>0).any(axis=1).all())
        policy.observe(x,np.array([y]))
        accepted.append(y);actions.append(a)
        predictions.append(policy.last_decision['predictions'][0][a])
    return dict(accepted=int(sum(accepted)),fraction=float(np.mean(accepted)),
        phase_accepted=[int(sum(accepted[i:i+h//3])) for i in range(0,h,h//3)],
        recovery_offset=next((i+1 for i,y in enumerate(accepted[h//3:2*h//3]) if y),None),
        mirror=int(policy.spent[0]),site_spent=policy.gateway_spent[0].tolist(),
        site_caps=cap[0].tolist(), actions=actions, outcome=list(map(int,accepted)))


def evaluate(spec,settings,seeds):
    rows=[]
    for seed in seeds:
        for case in spec['cases']:
            values,roster,classes=world(spec,seed,case)
            for fraction in spec['quota_fractions']:
                result=episode(settings,values,roster,classes,fraction,spec['mirror_fraction'])
                rows.append(dict(seed=seed,case=case,quota_fraction=fraction,**result))
    return rows


def tune(manifest,output):
    spec,digest=read_manifest(manifest);SLA[:]=spec['windows_ms']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    chosen={};scores=[]
    for family,items in spec['candidates'].items():
        best=None
        for index,raw in enumerate(items):
            rows=evaluate(spec,Settings(**raw),spec['dev_seeds'])
            score=float(np.mean([r['fraction'] for r in rows]))
            scores.append(dict(family=family,candidate=index,score=score,episodes=len(rows)))
            table(output/f'{family}-{index}.csv',rows,latex=False)
            if best is None or score>best[0]:best=(score,index,raw)
        chosen[family]=dict(index=best[1],settings=best[2],development_score=best[0])
    write_json(output/'selection.json',dict(manifest_sha256=digest,chosen=chosen,
        criterion=spec['primary'],candidates_per_family=3,locked_before_validation=True))
    table(output/'scores.csv',scores)


def validate(manifest,selection,output):
    spec,digest=read_manifest(manifest);SLA[:]=spec['windows_ms']
    selection=Path(selection);choice=json.loads(selection.read_text())
    if choice['manifest_sha256']!=digest:raise ValueError('Selection/manifest mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();rows=[]
    for family,item in choice['chosen'].items():
        for r in evaluate(spec,Settings(**item['settings']),spec['validation_seeds']):
            rows.append(dict(family=family,**r))
    # Untuned mechanism controls, never compete for best configuration selection.
    controls=dict(discount_legacy_bonus=replace(Settings(**choice['chosen']['joint_discounted']['settings']),memory_bonus='legacy'),
        joint_eligible=replace(Settings(**choice['chosen']['joint']['settings']),feedback_rule='eligible'),
        discounted_eligible=replace(Settings(**choice['chosen']['joint_discounted']['settings']),feedback_rule='eligible'),
        window_eligible=replace(Settings(**choice['chosen']['joint_window']['settings']),feedback_rule='eligible'))
    for family,s in controls.items():
        for r in evaluate(spec,s,spec['validation_seeds']):rows.append(dict(family=family,**r))
    table(output/'episodes.csv',rows,latex=False)
    summary=[]
    for case in spec['cases']:
        for q in spec['quota_fractions']:
            subset=[r for r in rows if r['case']==case and r['quota_fraction']==q]
            ref={r['seed']:r['fraction'] for r in subset if r['family']=='joint'}
            for f in list(FAMILIES)+list(controls):
                seq=[r for r in subset if r['family']==f]
                delta=np.array([r['fraction']-ref[r['seed']] for r in seq])
                summary.append(dict(case=case,quota_fraction=q,family=f,blocks=len(seq),
                    accepted_pct=100*float(np.mean([r['fraction'] for r in seq])),
                    versus_joint_pp=100*float(delta.mean()),
                    delta_by_seed_pp=(100*delta).tolist(),
                    paired_seed_sd_pp=100*float(delta.std(ddof=1))))
    table(output/'summary.csv',summary)
    # Secondary trace anchor: unchanged April development/May-Aug validation tensors.
    # No second tuning on these traces and no transformations to favor a method.
    atlas=Path(__file__).parents[1]/'data/atlas'
    anchor=[]
    old=SLA.copy();SLA[:]=[80,130,200]
    values=panels(atlas,'test')
    for panel,trace in enumerate(values):
        rng=np.random.default_rng(173000+panel)
        roster=np.stack([rng.choice(trace.shape[1],2,replace=False) for _ in trace])
        classes=np.resize(np.arange(3),len(trace));rng.shuffle(classes)
        for family,item in choice['chosen'].items():
            r=episode(Settings(**item['settings']),trace,roster,classes,.2,.25)
            anchor.append(dict(panel=panel,month_block=panel//2,family=family,**r))
    SLA[:]=old
    table(output/'atlas_anchor.csv',anchor,latex=False)
    write_json(output/'provenance.json',dict(manifest_sha256=digest,
        selection_sha256=sha256(selection.read_bytes()).hexdigest(),elapsed_s=time.perf_counter()-start,
        caution='Synthetic validation and historical ICMP proxy only; no real distributed measurements, no application-byte modeling. Seed is paired unit; Atlas panels within month are correlated. No claim of confirmation.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--out',type=Path,required=True)
    for name in ('tune','validate'):
        q=sub.add_parser(name);q.add_argument('manifest',type=Path);q.add_argument('--out',type=Path,required=True)
        if name=='validate':q.add_argument('--selection',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='prepare':prepare(a.out)
    elif a.cmd=='tune':tune(a.manifest,a.out)
    else:validate(a.manifest,a.selection,a.out)

if __name__=='__main__':main()
