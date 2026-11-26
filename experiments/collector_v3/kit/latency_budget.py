"""Charge only the late planner an extra constant transport/authorization delay. The early oracle keeps its original network."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from experiment import contexts,panels,run
from model import Settings
from gateway_study import caps


def main():
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('data',type=Path);a=p.parse_args();sys.path.insert(0,str(a.project.resolve()))
    s=Settings(**json.loads((a.data/'gateway_calibration.json').read_text())['records']['joint']['chosen']['settings'])
    bounds=json.loads((a.data/'option_bounds.json').read_text());records=[]
    for name,path,tag in [('first','atlas',0),('confirm','atlas_confirmation',19),('feedback','atlas_feedback',43)]:
        x=panels(a.data/path,'test');p,h,_,_=x.shape;r=24
        rs,ws=contexts(x,'test',3,r,tag);limits=np.repeat(np.stack([caps(i,h) for i in range(p)]),r,axis=0)
        for delta in (0,5,10,15,20,25,50):
            changed=np.where(x>0,x+delta,x)
            ys,_=run(changed,rs,ws,h//4,s,gateway_caps=limits)
            for i,by_draw in enumerate(ys):
                for j,val in enumerate(by_draw):records.append((name,delta,i,j,int(val),bounds[name][i]['prebook']))
    with (a.data/'latency_budget.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['phase','extra_ms','panel','draw','accepted','early_option_oracle']);w.writerows(records)

if __name__=='__main__':main()
