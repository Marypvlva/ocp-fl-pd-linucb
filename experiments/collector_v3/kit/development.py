"""Tune on April only. All choices and scores remain in dev_scores.json."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from experiment import panels,contexts,run,bounds
from model import Settings

def main():
 p=argparse.ArgumentParser();p.add_argument('project');p.add_argument('atlas',type=Path);p.add_argument('out',type=Path)
 a=p.parse_args();sys.path.insert(0,str(Path(a.project).resolve()))
 x=panels(a.atlas,'dev')
 rs,cs=contexts(x,'dev',3,12)
 tests=[Settings(mode,alpha,eta) for mode in ('joint','round_lin') for alpha in (.15,.55,1.0) for eta in (.025,.08,.20)]
 tests += [Settings(mode,alpha,eta,price) for mode in ('fixed',) for alpha in (.15,.55,1.0) for price in (.08,.25,.5,.8) for eta in (.08,)]
 tests += [Settings(mode,eta=eta) for mode in ('greedy','eps','thompson','recent','filter','periodic','always_single','no_price','no_ids') for eta in ((.025,.08,.2) if mode in ('greedy','eps','thompson','recent','no_ids') else (.08,))]
 out={'scores':{},'chosen':{},'bounds':{}}
 for q in (.10,.25,.50):
  n=int(q*x.shape[1]);rows=[];best={}
  for setting in tests:
   vals,cost=run(x,rs,cs,n,setting)
   avg=float(vals.mean());row=dict(setting=setting.__dict__,mean=avg,spent=float(cost.mean()))
   rows.append(row)
   if setting.mode not in best or avg>best[setting.mode]['mean']: best[setting.mode]=row
  out['chosen'][str(q)]={m:r['setting'] for m,r in best.items()};out['scores'][str(q)]=rows
  out['bounds'][str(q)]=bounds(x,rs,cs,n)
  print(q,{m:round(r['mean'],1) for m,r in best.items()},flush=True)
  print('bounds',{m:round(float(np.mean(v)),1) for m,v in out['bounds'][str(q)].items()},flush=True)
 a.out.write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
