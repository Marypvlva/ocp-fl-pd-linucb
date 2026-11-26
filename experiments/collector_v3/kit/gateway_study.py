"""Finite per-collector allocations and exact pre-reveal upper bounds."""
import argparse
import csv
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
from experiment import panels,contexts,run
from model import Settings, SLA, USE, ARMS, EXTRA, LEFT, RIGHT
from treasury import COST,dual,shadow


def caps(panel,h,fraction=.20):
    cap=np.full(4,int(fraction*h))
    cap[[panel%4,(panel+1)%4]]=h
    return cap


def tabulate(tensor,classes,k,club):
    h,n,g=tensor.shape
    combos=np.array(list(combinations(range(n),k)))
    timely=(tensor>0)&(tensor<=SLA[classes,None,None])
    timely=timely[:,:,LEFT]|timely[:,:,RIGHT]
    # Filled in chunks to limit RAM. Every possible roster has equal probability.
    averages=[];separate=[]
    for t in range(0,h,24):
        masks=timely[t:t+24,combos,:].all(axis=2)
        averages.append(masks.mean(axis=1))
        separate.append(np.stack([masks[:,:,idx].any(axis=2).mean(axis=1) for idx in club],axis=1))
    return np.concatenate(averages), np.concatenate(separate)


def clubs(caps_g,h):
    # On unconstrained gateways after prebooking, grant a best-in-hindsight route.
    keys=COST.copy()
    keys[:,1:]*=(caps_g<h)
    distinct,lookup=np.unique(keys,axis=0,return_inverse=True)
    return distinct,[np.flatnonzero(lookup==j) for j in range(len(distinct))]


def calibrate(data_dir,out):
    train=panels(data_dir,'dev')
    _,classes=contexts(train,'dev',3,1)
    prices={}
    for layout in range(4):
        c=caps(layout,120)
        # Give the fixed-price control fully measured April counterfactual rewards.
        groups=[[j] for j in range(10)]
        tables=[tabulate(m,classes,3,groups)[0] for m in train]
        prices[str(layout)]=shadow(np.concatenate(tables),np.r_[.25,c/120]).tolist()
    rosters,cs=contexts(train,'dev',3,12)
    cap=np.repeat(np.stack([caps(g,120) for g in range(2)]),12,axis=0)
    modes=('joint','round_lin','recent','greedy','eps','thompson','filter','scalar','fixed','no_ids','no_price','lp','lp_mean','periodic')
    records={}
    for m in modes:
        candidates=(.025,.08,.2) if m not in ('lp','lp_mean','fixed','no_price','periodic') else (.08,)
        scalars=(.15,.55,1.) if m in ('joint','round_lin','scalar','no_ids','lp','fixed') else (.55,)
        trials=[]
        for eta in candidates:
            for alpha in scalars:
                ss=Settings(m,alpha=alpha,eta=eta)
                y,b=run(train,rosters,cs,30,ss,gateway_caps=cap,
                      fixed_prices=np.repeat([prices['0'],prices['1']],12,axis=0))
                trials.append(dict(settings=ss.__dict__,mean=float(y.mean())))
        chosen=max(trials,key=lambda r:r['mean'])
        records[m]=dict(chosen=chosen,grid=trials)
    (out/'gateway_calibration.json').write_text(json.dumps(dict(prices=prices,records=records),indent=2)+'\n')
    print('April', {m:round(r['chosen']['mean'],2) for m,r in records.items()},flush=True)


def evaluate(data_dir,out,split,k,repeats,variant):
    train=panels(data_dir,split)
    if variant=='same_links':
        train=np.repeat(train[:,:,:,:1],4,axis=3)
    p,h,n,g=train.shape
    contexts_tag=0
    rs,cs=contexts(train,split,k,repeats,contexts_tag)
    q={'tight':.10,'loose':.5,'no_mirror':0}.get(variant,.25)
    price=json.loads((out/'gateway_calibration.json').read_text())
    old=json.loads((out/'dev_scores.json').read_text())['chosen']['0.25']
    cap=np.stack([caps(i,h,{'tight':.10,'loose':.4}.get(variant,.20)) for i in range(p)])
    if variant=='global':
        cap[:]=h
    lim=int(h*q)
    indcaps=np.repeat(cap,repeats,axis=0)
    fprices=np.repeat([price['prices'][str(i%4)] for i in range(p)],repeats,axis=0)
    scenarios=price['records'] if variant in ('main','global') else {m:price['records'][m] for m in ('joint','no_price','no_ids','recent','filter','fixed')}
    rows=[]
    for m,entry in scenarios.items():
        s=Settings(**entry['chosen']['settings'])
        if variant=='global':
            s=Settings(**old.get(m,old['joint']))
            s=replace(s,mode=m)
        if variant=='long_delay':
            s=replace(s,delay=12)
        use_fixed=fprices if variant!='global' else np.tile([s.initial_price,0.,0.,0.,0.],(p*repeats,1))
        take,actions=run(train,rs,cs,lim,s,gateway_caps=indcaps,
                     fixed_prices=use_fixed,return_trace=True)
        counts=USE[actions].sum(axis=2)
        assert np.all(counts<=indcaps.reshape(p,repeats,g)) and np.all(EXTRA[actions].sum(axis=2)<=lim)
        for i in range(p):
            for j in range(repeats):
                rows.append([i,j,k,variant,m,int(take[i,j].sum()),int(EXTRA[actions[i,j]].sum()),*map(int,counts[i,j])])
    if variant=='main':
        wrong,_=run(train,rs,cs,lim,Settings(**price['records']['joint']['chosen']['settings']),shuffle=True,gateway_caps=indcaps,return_trace=True)
        for i in range(p):
            for j in range(repeats): rows.append([i,j,k,variant,'stale_roster',int(wrong[i,j].sum()),'','','','',''])
    bb=[]
    for i in range(p):
        unique,by=clubs(cap[i],h)
        plain,strong=tabulate(train[i],cs,k,by)
        bound0=dual(plain,np.r_[lim,cap[i]]/h)[0]
        relaxed=dual(strong,np.r_[lim,cap[i]]/h,unique)[0]
        bb.append(dict(panel=i,blind=bound0,prebook=relaxed,missing=float((train[i]==-2).mean()),
                       explicit_loss=float((train[i]==-1).mean()),caps=cap[i].tolist(),h=h))
    pref=f'{split}_{variant}_k{k}'
    with (out/(pref+'.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(('panel','draw','k','scenario','policy','accepted','mirror','fi','br','us','ug'));w.writerows(rows)
    (out/(pref+'_bounds.json')).write_text(json.dumps(bb,indent=2)+'\n')
    print(pref,'complete',len(rows),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('data',type=Path)
    ap.add_argument('action',choices=('calibrate','dev','test'));ap.add_argument('--variant',default='main')
    ap.add_argument('--k',type=int,default=3);ap.add_argument('--repeats',type=int,default=24)
    a=ap.parse_args();sys.path.insert(0,str(a.project.resolve()))
    if a.action=='calibrate':
        calibrate(a.data/'atlas',a.data)
    else:
        evaluate(a.data/'atlas',a.data,a.action,a.k,a.repeats,a.variant)

if __name__=='__main__':main()
