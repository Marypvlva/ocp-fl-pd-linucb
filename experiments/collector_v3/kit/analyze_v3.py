"""Reproduce the report from the per-draw counts, measurements, and signed-protocol run logs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

from live import H, SEEDS, MODES, B_LIMIT
from experiment import panels
from gateway_study import caps


def read(path):
    return json.loads(path.read_text())


def table(path):
    with path.open(newline='') as src:
        return list(csv.DictReader(src))


def stats(values):
    a=np.asarray(values,dtype=float)
    idx=np.random.default_rng(734128).integers(len(a),size=(10000,len(a)))
    lo,hi=np.quantile(a[idx].mean(axis=1),[.025,.975])
    return dict(mean=float(a.mean()),ci=[float(lo),float(hi)],panels=a.tolist(),positive=int((a>0).sum()))


def methods(rows, n_draws, gateway_limits, mirror_limit):
    tallies=defaultdict(dict)
    for row in rows:
        i,j=int(row['panel']),int(row['draw']); value=int(row['accepted']); policy=row['policy']
        assert 0<=i<8 and 0<=j<n_draws and 0<=value<=480
        assert (i,j) not in tallies[policy]
        tallies[policy][i,j]=value
        if row['mirror']!='':
            spent=np.array([int(row[s]) for s in ('fi','br','us','ug')]); mirrored=int(row['mirror'])
            assert spent.sum()==480+mirrored and 0<=mirrored<=mirror_limit
            assert np.all((0<=spent)&(spent<=gateway_limits[i]))
    matrices={m:np.array([[group[i,j] for j in range(n_draws)] for i in range(8)]) for m,group in tallies.items()}
    return {m:a.mean(axis=1) for m,a in matrices.items()}


def main_phase(data, name):
    limits=np.stack([caps(i,480) for i in range(8)])
    y=methods(table(data/(name+'_methods.csv')),24,limits,120)
    assert len(y)==20
    upper=np.array([r['prebook'] for r in read(data/(name+'_methods_bounds.json'))]);joint=y['factor_1_local']
    blind=np.array([r['blind'] for r in read(data/(name+'_methods_bounds.json'))])
    interaction=(joint-y['factor_1_zero'])-(y['factor_0_local']-y['factor_0_zero'])
    strong=np.max([y[m] for m in ('recent','greedy','thompson','fixed','filter','lp_mean','law_greedy','law_recent','greedy_new')],axis=0)
    trace=panels(data/('atlas' if name=='first' else 'atlas_confirmation'),'test')
    return dict(means={m:float(a.mean()) for m,a in y.items()},prebook=stats(upper),blind=stats(blind),
                advantage=stats(joint-upper),interaction=stats(interaction),
                per_site_over_single_price=stats(joint-y['factor_1_global']),
                interaction_global=stats((joint-y['factor_1_global'])-(y['factor_0_local']-y['factor_0_global'])),
                versus_round=stats(joint-y['round_lin']),law_vs_its_greedy=stats(y['law']-y['law_greedy']),
                law_vs_panel_best=stats(y['law']-strong),joint_vs_panel_best=stats(joint-strong),
                missing=float((trace==-2).mean()),loss=float((trace==-1).mean()))


def feedback_phase(data):
    previous=set(read(data/'atlas/provenance.json')['probe_ids']) | set(read(data/'atlas_confirmation/provenance.json')['probe_ids'])
    current=set(read(data/'atlas_feedback/provenance.json')['probe_ids'][32:]);assert len(current)==128 and not current.intersection(previous)
    limits=np.stack([caps(i,480) for i in range(8)])
    y=methods(table(data/'feedback_methods.csv'),24,limits,120);assert len(y)==13
    upper=np.array([row['prebook'] for row in read(data/'feedback_bounds.json')]);joint=y['rtt_matched']
    best=np.max([v for k,v in y.items() if k.startswith(('aggregate','round'))],axis=0)
    trace=panels(data/'atlas_feedback','test')
    return dict(means={m:float(a.mean()) for m,a in y.items()},prebook=stats(upper),advantage=stats(joint-upper),
                best_aggregate=stats(best),binary_vs_best=stats(y['binary_tuned']-best),
                matched_binary_vs_EM=stats(y['binary_matched']-y['aggregate_matched']),
                full_vs_best=stats(y['rtt_tuned']-best),bonus=stats(y['rtt_tuned']-y['rtt_no_bonus']),
                missing=float((trace==-2).mean()),loss=float((trace==-1).mean()))


def protocols(rows, h, seed, mode, contract):
    assert len(rows)==h
    charges=np.zeros(4,dtype=int);mirror=0; ledger=0;previous=None;failed=0;dup=0
    gas=[]; sequence=[]; digests=set()
    for t,x in enumerate(rows,1):
        assert (x['seed'],x['mode'],x['t'])==(seed,mode,t)
        a=x['act'];roster=x['members'];assert 1<=len(a)<=2 and len(set(a))==len(a) and len(set(roster))==2
        assert all(0<=g<4 for g in a) and all(0<=i<4 for i in roster)
        charges[a]+=1;mirror+=len(a)-1;ledger+=x['tx']
        assert np.array_equal(charges,x['count']) and mirror==x['mirror'] and ledger==x['ledger']
        assert np.all(charges<=x['cap']) and mirror<=h//4 and ledger<=B_LIMIT
        seen=x['observed']; assert set(map(int,seen))==set(a)
        arrived=[int(i) for obs in seen.values() for i,ms in obs.items() if 0<ms<=x['slack']+1.e-6]
        assert set(arrived).issubset(roster) and x['accepted']==(set(arrived)==set(roster))
        dup+=len(arrived)-len(set(arrived));failed+=int(not x['accepted'])
        if previous is not None and not x['accepted']:assert previous==x['hash']
        previous=x['hash']; assert len(bytes.fromhex(previous))==32
        sequence.append((x['t'],tuple(roster),x['slack'],x['block']))
        if contract:
            q=x['reservation'];assert q['mask']==sum(1<<g for g in a) and len(bytes.fromhex(q['digest']))==32
            assert q['digest'] not in digests;digests.add(q['digest']);gas.append(q['gas'])
    return dict(accepted=h-failed,charged_failed=failed,duplicates=dup,bytes=ledger,mirror=mirror,
                gas=gas,wall=[x['round_s'] for x in rows]),sequence


def live(data,contract):
    n=24 if contract else H
    seeds=SEEDS[:1] if contract else SEEDS
    modes=('joint','no_ids','recent') if contract else MODES
    root=data/('live_atlas' if contract else 'live');episodes={};gases=[];w=[]; charged=0;dups=0; volumes=[]
    for seed in seeds:
        paired=None
        for mode in modes:
            item,seq=protocols(read(root/f'{seed}_{mode}.json'),n,seed,mode,contract)
            if paired is not None:assert paired==seq
            paired=seq;gases+=item.pop('gas');w+=item.pop('wall');volumes.append(item['bytes']);charged+=item['charged_failed'];dups+=item['duplicates']
            episodes.setdefault(mode,[]).append(item)
    return dict(episodes=episodes,total=len(w),charged_failures=charged,duplicate_replies=dups,
                byte_range=[min(volumes),max(volumes)],wall_med_p95=np.quantile(w,[.5,.95]).tolist(),
                gas_med_p95=np.quantile(gases,[.5,.95]).tolist() if gases else [])


def analyze(data,simulate_only=False):
    a,b=main_phase(data,'first'),main_phase(data,'confirm')
    origin=set(read(data/'atlas/provenance.json')['probe_ids']); later=set(read(data/'atlas_confirmation/provenance.json')['probe_ids'][32:]); assert len(later)==128 and not origin.intersection(later)
    tests={}
    for path in sorted(data.glob('test_*_k*.csv')):
        bounds=read(path.with_name(path.stem+'_bounds.json'))
        rows=table(path);ns=1+max(int(s['draw']) for s in rows);variant=rows[0]['scenario']
        limit=int(480*{'tight':.1,'loose':.5,'no_mirror':0}.get(variant,.25));c=np.array([x['caps'] for x in bounds])
        yy=methods(rows,ns,c,limit);ub=np.array([x['prebook'] for x in bounds]);tests[variant]=dict(means={m:float(x.mean()) for m,x in yy.items()},prebook=float(ub.mean()),advantage=stats(yy['joint']-ub))
    assert abs(tests['main']['means']['joint']-a['means']['factor_1_local'])<1.e-8
    assert max(tests['same_links']['means'].values())-min(tests['same_links']['means'].values())<1.e-10
    measured=defaultdict(list)
    for row in table(data/'measurement_sensitivity.csv'):
        if row['policy']=='joint':
            h=int(row['rounds']); measured[(row['source'],row['scenario'])].append((h,480*float(row['accepted'])/h,480*float(row['prebook'])/h))
    checks={}
    for (name,scope),rows in measured.items():
        arr=np.array(rows);checks.setdefault(name,{})[scope]=dict(count=len(rows),slot_range=[int(x) for x in [arr[:,0].min(),arr[:,0].max()]],normalized=list(arr[:,1:].mean(axis=0)),gain=stats(arr[:,1]-arr[:,2]))
    result=dict(first=a,confirm=b,feedback=feedback_phase(data),stress=tests,measurement=checks)
    options=read(data/'option_bounds.json')
    for name,rows in options.items():
        phase=result[name]; stricter=np.array([z['prebook'] for z in rows]);original=np.array(phase['prebook']['panels'])
        assert np.all(stricter+1e-8>=original)
        phase['option_prebook']=stats(stricter)
        phase['option_advantage']=stats(np.array(phase['advantage']['panels'])+original-stricter)
    lag=defaultdict(dict)
    for row in table(data/'latency_budget.csv'):
        name=row['phase'];dt=int(row['extra_ms']);i=int(row['panel']);j=int(row['draw'])
        assert (i,j) not in lag[(name,dt)]
        lag[(name,dt)][i,j]=int(row['accepted'])
        assert abs(float(row['early_option_oracle'])-result[name]['option_prebook']['panels'][i])<1.e-9
    for (name,dt),cells in lag.items():
        assert len(cells)==192
        vals=np.array([[cells[i,j] for j in range(24)] for i in range(8)]).mean(axis=1)
        gap=stats(vals-np.array(result[name]['option_prebook']['panels']))
        if dt==0:np.testing.assert_allclose(gap['panels'],result[name]['option_advantage']['panels'])
        result[name].setdefault('exclusive_delay',{})[str(dt)]=dict(accepted=float(vals.mean()),gap=gap)
    if not simulate_only:
        result.update(local=live(data,False),atlas_local=live(data,True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path);p.add_argument('--simulate-only',action='store_true');a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.data,a.simulate_only),ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(a.output)
