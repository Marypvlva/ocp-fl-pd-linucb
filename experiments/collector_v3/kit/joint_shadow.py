"""Same-history forecast diagnostics, NEVER off-policy reward evaluation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from model import Settings, Policy, SLA, ARMS
from memory_policy import MemoryPolicy
from program import read_plan
from gateway_study import caps
from analysis_program import table, write_json


def candidates(base):
    return dict(joint=base, **{
        f'discount_{g}':replace(base,memory='discount',discount=g) for g in (.9,.95,.98)}, **{
        f'window_{w}':replace(base,memory='window',window=w) for w in (16,32,64)},
        discount_095_legacy_bonus=replace(base,memory='discount',discount=.95,memory_bonus='legacy'))


def analyze(directory,output):
    directory,output=Path(directory).resolve(),Path(output).resolve()
    if output==directory or directory in output.parents:raise ValueError('Read-only originals')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for planpath in sorted(directory.rglob('plan.json')):
        plan,digest=read_plan(planpath)
        SLA[:]=plan['windows_ms']
        for si,seed in enumerate(plan['seeds']):
            base=Settings(**plan['policy_specs']['joint'])
            learners={name:(MemoryPolicy if s.memory!='none' else Policy)(s,1,plan['population'],
                plan['k'],plan['horizon'],plan['mirror_limit'],plan.get('policy_tie_seed',53),
                caps(si,plan['horizon'],plan['site_fraction'])[None,:])
                for name,s in candidates(base).items()}
            for file in sorted((planpath.parent/f'{seed}_joint').glob('round-*.json')):
                r=json.loads(file.read_text())
                arm=ARMS.index(tuple(r['gateways']))
                c=plan['windows_ms'].index(r['window_ms'])
                roster=np.array([r['selected']])
                samples=np.array(r['path_readings'])
                attempted=np.array(r['path_observed'],bool)
                for name,learner in learners.items():
                    learner.decide(c,roster,forced_arm=arm)
                    d=learner.last_decision
                    mean=np.array(d['path_mean_prediction'][0])
                    pathloss=[]; forecast=[]; actual=[]
                    for j,g in enumerate(r['gateways']):
                        for i in range(plan['k']):
                            if attempted[0,i,j]:
                                y=float(samples[0,i,j]>0)
                                pathloss.append((mean[i,g]-y)**2)
                                forecast.append(mean[i,g]);actual.append(y)
                    coverage=all(any(samples[0,i,j]>0 for j in range(len(r['gateways']))) for i in range(plan['k']))
                    prediction=float(np.prod(1-np.prod(1-mean[:,r['gateways']],axis=1)))
                    rows.append(dict(series=str(planpath.parent.relative_to(directory)),plan_sha256=digest,
                        seed=seed,round=r['round'],phase=(r['round']-1)//40,learner=name,
                        path_labels=len(pathloss),path_squared_error=sum(pathloss),
                        path_forecast_sum=sum(forecast),path_successes=sum(actual),
                        coverage_prediction=prediction,coverage=int(coverage),
                        coverage_brier=(prediction-coverage)**2))
                    learner.observe(samples,np.array([r['accepted']]),attempted)
    table(output/'shadow_rounds.csv',rows,latex=False)
    groups={}
    for r in rows:
        key=(r['series'],r['seed'],r['phase'],r['learner'])
        groups.setdefault(key,[]).append(r)
    summary=[]
    for (series,seed,phase,learner),group in groups.items():
        n=sum(r['path_labels'] for r in group)
        summary.append(dict(series=series,seed=seed,phase=phase,learner=learner,labels=n,
            path_brier=sum(r['path_squared_error'] for r in group)/n if n else None,
            path_mean=sum(r['path_forecast_sum'] for r in group)/n if n else None,
            path_success_rate=sum(r['path_successes'] for r in group)/n if n else None,
            coverage_brier=float(np.mean([r['coverage_brier'] for r in group]))))
    table(output/'shadow_phases.csv',summary)
    write_json(output/'scope.json',dict(
        meaning='Predictions of shadow learners on original Joint actions/history; no rewards for alternative actions.',
        tuning='All v0.7.0 data are development data; these forecasts cannot select/confirm superiority on the same data.',
        censoring='Legacy pooling deliberately retained to isolate memory. Unknown unattempted labels omitted from path scores. Coverage includes execution failures.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();analyze(a.directory,a.out)
