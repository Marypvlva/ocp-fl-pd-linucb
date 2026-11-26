"""More permissive prebooking bound: discard any reserved resources and use any free gateways after reveal."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from experiment import panels,contexts
from gateway_study import caps,tabulate
from treasury import COST,dual


def main():
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('data',type=Path);a=ap.parse_args()
    sys.path.insert(0,str(a.project.resolve()));out={}
    for name,path,tag in [('first','atlas',0),('confirm','atlas_confirmation',19),('feedback','atlas_feedback',43)]:
        tensor=panels(a.data/path,'test');h=tensor.shape[1];_,windows=contexts(tensor,'test',3,1,tag)
        bb=[]
        for i,m in enumerate(tensor):
            c=caps(i,h);use=COST.copy();use[:,1:]*=(c<h);profiles=np.unique(use,axis=0)
            options=[np.flatnonzero((use<=b).all(axis=1)) for b in profiles]
            _,value=tabulate(m,windows,3,options)
            bound=dual(value,np.r_[h//4,c]/h,profiles)[0];bb.append(dict(panel=i,prebook=bound))
        out[name]=bb
    (a.data/'option_bounds.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
