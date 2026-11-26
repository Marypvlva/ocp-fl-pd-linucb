"""Independent confirmatory study and a matched two-factor interaction."""
import argparse,csv,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from experiment import contexts,panels,run
from model import Policy,Settings,USE,EXTRA
from gateway_study import caps,clubs,tabulate
from treasury import dual
from lookahead import ForecastPrices


def study(project,data,phase,early=False):
    sys.path.insert(0,str(project.resolve()))
    cal=json.loads((data/'gateway_calibration.json').read_text())
    source= data/('atlas_confirmation' if phase=='confirm' else 'atlas')
    split='dev' if phase=='tune' else 'test'
    tensor=panels(source,split)
    r=8 if phase=='tune' else 24
    p,h,n,g=tensor.shape
    rs,cs=contexts(tensor,split,3,r,tag=19 if phase=='confirm' else 0)
    cap=np.stack([caps(i,h) for i in range(p)])
    perrun=np.repeat(cap,r,axis=0)
    fixed=np.repeat([cal['prices'][str(i%4)] for i in range(p)],r,axis=0)
    base=Settings(**cal['records']['joint']['chosen']['settings'])
    def play(name,settings,planner=Policy,options=None):
        y,arms=run(tensor,rs,cs,h//4,settings,gateway_caps=perrun,fixed_prices=fixed,
                   policy_class=planner,policy_options=options,return_trace=True)
        counts=USE[arms].sum(axis=2);spent=EXTRA[arms].sum(axis=2);assert np.all(counts<=cap[:,None,:]) and spent.max()<=h//4
        return [(i,j,name,int(y[i,j].sum()),int(spent[i,j]),*map(int,counts[i,j])) for i in range(p) for j in range(r)]
    if phase=='tune':
        results=[]
        for rho in (.1,.4,1.):
            for alpha in (.035,.12,.35):
                for eta in (.04,.12):
                    s=replace(base,ridge=rho,alpha=alpha,eta=eta)
                    rows=play('forecast',s,ForecastPrices)
                    results.append(dict(config=s.__dict__,accepted=float(np.mean([row[3] for row in rows]))))
        best=max(results,key=lambda row:row['accepted'])
        (data/'lookahead_dev.json').write_text(json.dumps(dict(grid=results,chosen=best),indent=2))
        print('tuned',best,flush=True);return
    chosen=Settings(**json.loads((data/'lookahead_dev.json').read_text())['chosen']['config'])
    records=[]
    for flags in ((True,'local'),(True,'zero'),(False,'local'),(False,'zero'),(True,'global'),(False,'global')):
        roster,price=flags
        records+=play(f'factor_{int(roster)}_{price}',replace(base,use_current_roster=roster,price_rule=price))
    for mode in ('recent','filter','greedy','round_lin','fixed','thompson','lp_mean','lp'):
        records+=play(mode,Settings(**cal['records'][mode]['chosen']['settings']))
    for name,setting,cls,opt in [
        ('law',chosen,ForecastPrices,{}),('empirical',chosen,ForecastPrices,{'source':'past'}),
        ('new_no_lookahead',chosen,Policy,{}),('law_greedy',replace(chosen,alpha=0.),ForecastPrices,{}),
        ('law_recent',replace(chosen,mode='recent'),ForecastPrices,{}),
        ('greedy_new',replace(chosen,alpha=0.),Policy,{})]:
        records+=play(name,setting,cls,opt)
    with (data/f'{phase}_methods.csv').open('w',newline='') as file:
        wr=csv.writer(file);wr.writerow(('panel','draw','policy','accepted','mirror','fi','br','us','ug'));wr.writerows(records)
    bounds=[]
    for i in range(p):
        unique,by=clubs(cap[i],h);plain,strong=tabulate(tensor[i],cs,3,by)
        target=np.r_[h//4,cap[i]]/h
        bounds.append(dict(panel=i,blind=dual(plain,target)[0],prebook=dual(strong,target,unique)[0]))
    (data/f'{phase}_methods_bounds.json').write_text(json.dumps(bounds,indent=2))
    print(phase,'finished',len(records),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('data',type=Path)
    ap.add_argument('phase',choices=('tune','first','confirm'))
    args=ap.parse_args();study(args.project,args.data,args.phase)
