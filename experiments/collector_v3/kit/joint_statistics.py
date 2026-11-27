"""Paired seed-level summaries; rounds never act as independent replicates."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from statistics_program import paired,holm
from analysis_program import write_json,table


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def analyze(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    real=[]
    for name in ('E1','E5','E3'):
        rows=read(root/f'{name}-audit/phases.csv')
        groups=defaultdict(lambda:defaultdict(dict))
        for r in rows:
            group=groups[r['series']][r['policy']]
            seed=int(r['seed']);n,a=group.get(seed,(0,0))
            group[seed]=n+int(r['rounds']),a+int(r['accepted'])
        for series,methods in groups.items():
            for mode,seq in methods.items():
                if mode=='joint':continue
                seeds=sorted(set(seq)&set(methods['joint']))
                delta=[100*(seq[s][1]/seq[s][0]-methods['joint'][s][1]/methods['joint'][s][0]) for s in seeds]
                real.append(dict(experiment=name,series=series,contrast=mode+' minus joint',seeds=seeds,
                    delta_pp=delta,**paired(delta),
                    inference='Exploratory pilot only. n=2; no confirmatory claims. Constant paired deltas do not imply zero population uncertainty.'))
    write_json(output/'real_seed_contrasts.json',real)
    rows=read(root/'local/validation/episodes.csv')
    methods=('joint_discounted','joint_window','matched_recent','article_pd')
    by_seed=defaultdict(lambda:defaultdict(list))
    for r in rows:by_seed[int(r['seed'])][r['family']].append(float(r['fraction']))
    primary=[]
    for m in methods:
        delta=[100*(np.mean(v[m])-np.mean(v['joint'])) for _,v in sorted(by_seed.items())]
        primary.append(dict(contrast=m+' minus joint',delta_pp=delta,**paired(delta,family=4),
            scope='Local synthetic validation; equal weight across 8 cases within each seed; conditional on this generator.'))
    adjusted=holm({r['contrast']:r['p_exchangeable'] for r in primary})
    for r in primary:r['p_holm']=adjusted[r['contrast']]
    write_json(output/'local_primary.json',primary)
    secondary=[]
    for r in read(root/'local/validation/summary.csv'):
        if r['family'] not in methods:continue
        import json
        delta=json.loads(r['delta_by_seed_pp'])
        secondary.append(dict(case=r['case'],quota_fraction=float(r['quota_fraction']),
            contrast=r['family']+' minus joint',**paired(delta,family=32)))
    adjusted=holm({str(i):r['p_exchangeable'] for i,r in enumerate(secondary)})
    for i,r in enumerate(secondary):r['p_holm']=adjusted[str(i)]
    table(output/'local_secondary.csv',secondary)
    write_json(output/'scope.json',dict(primary_family=4,secondary_family=32,
        caveat='Sign flips require symmetric/exchangeable paired seed differences; t intervals assume approximately normal seed means. New cloud confirmation needs new seeds and time blocks; all old v0.7.0 data are development data.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.out)
