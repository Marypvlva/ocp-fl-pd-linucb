"""Measurement-missingness and relay-overhead checks, with the same frozen policies."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from experiment import contexts,panels,run
from gateway_study import caps,clubs,tabulate
from model import Settings
from treasury import dual


def main():
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('data',type=Path)
    a=ap.parse_args();sys.path.insert(0,str(a.project.resolve()))
    cfg=json.loads((a.data/'gateway_calibration.json').read_text())['records']
    rows=[]
    for name in ('atlas','atlas_confirmation'):
        tensor=panels(a.data/name,'test')
        samples,cs=contexts(tensor,'test',3,12,tag=0 if name=='atlas' else 19)
        for scenario in ('measured_slots','missing_fast','overhead_25','overhead_50','common_site_outage'):
            for i,orig in enumerate(tensor):
                x=orig.copy();mask=np.ones(len(x),dtype=bool)
                if scenario=='measured_slots':mask=~(x==-2).any(axis=(1,2))
                if scenario=='missing_fast':x[x==-2]=1.
                if scenario.startswith('overhead'):x[x>0]+=int(scenario.split('_')[-1])
                if scenario=='common_site_outage':
                    # Controlled shared impairment of every gateway on 1 out of 5 time slots.
                    x[np.arange(len(x))%5==0]=-1.
                x=x[mask]
                h=len(x)
                if h<48:continue
                r=samples[i:i+1,:,:,:][:,:,mask,:]
                c=cs[mask];cl=caps(i,h)
                group,by=clubs(cl,h);_,strong=tabulate(x,c,3,by)
                bound=dual(strong,np.r_[h//4,cl]/h,group)[0]
                for m in ('joint','recent'):
                    sett=Settings(**cfg[m]['chosen']['settings'])
                    result,_=run(x[None,:,:,:],r,c,h//4,sett,gateway_caps=np.tile(cl,(r.shape[1],1)))
                    rows.append((name,scenario,i,m,h,result.mean(),bound))
    with (a.data/'measurement_sensitivity.csv').open('w',newline='') as out:
        wr=csv.writer(out);wr.writerow(('source','scenario','panel','policy','rounds','accepted','prebook'));wr.writerows(rows)


if __name__=='__main__':main()
