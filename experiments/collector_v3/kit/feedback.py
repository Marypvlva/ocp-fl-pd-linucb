"""Compare named signed link receipts with the strongest available aggregate-only credit assignment."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
from experiment import panels,contexts,run
from gateway_study import caps,clubs,tabulate
from model import Policy,Settings,SLA,LEFT,RIGHT,EXTRA,USE
from treasury import dual


def conditional_means(p0,p1,pair,success):
    """E[X_ig | AND_i OR_g X_ig = y] for conditionally independent links."""
    coverage=np.where(pair[:,None],1-(1-p0)*(1-p1),p0)
    joint=coverage.prod(axis=1,keepdims=True)
    def posterior(p):
        positive=p/coverage
        negative=(p-joint*positive)/(1-joint)
        return np.clip(np.where(success[:,None],positive,negative),0.,1.)
    return posterior(p0), posterior(p1)


class FeedbackPolicy(Policy):
    def __init__(self,*args,signal,**kwargs):
        super().__init__(*args,**kwargs)
        self.signal=signal

    def observe(self,readings,reward):
        if self.signal.startswith('aggregate'):
            # The learner only receives the end-of-round bit; no named labels or timing in this payload.
            c,members,arm,features=self.last
            self.pending.append((self.t+self.s.delay,(c,members,arm,None,reward.copy(),features)))
            self.t+=1
        else:
            super().observe(readings,reward)

    def _mature(self,payload):
        c,members,arm,readings,reward,features=payload[:6]
        if self.signal=='rtt':
            return super()._mature(payload)
        pair=EXTRA[arm]
        for d in range(len(SLA)):
            granted=(np.ones(self.r,bool) if d==c else
                     (reward if d>c else ~reward) if self.signal=='aggregate_ordered' else np.zeros(self.r,bool))
            if not granted.any():continue
            indices=[self._index(members,g,d) for g in (LEFT[arm],RIGHT[arm])]
            if self.signal=='binary':
                estimates=[(readings[:,:,j]>0).astype(float) for j in range(2)]
            else:
                prob=[np.clip(.5+(self.wins[ix]-.5*self.done[ix])/(self.s.ridge+self.design[ix]),1e-6,1-1e-6) for ix in indices]
                estimates=conditional_means(*prob,pair,reward)
            for j,ix in enumerate(indices):
                valid=(granted & (pair if j==1 else True))[:,None]
                self.wins[ix]+=estimates[j]*valid
                self.done[ix]+=valid
                if d!=c:self.design[ix]+=valid


def series(tensor,rs,windows,limits,settings,signal):
    p,r,h,_=rs.shape
    klass=Policy if signal in ('round','recent') else FeedbackPolicy
    opt={} if klass==Policy else dict(signal=signal)
    y,arms=run(tensor,rs,windows,h//4,settings,gateway_caps=np.repeat(limits,r,axis=0),return_trace=True,policy_class=klass,policy_options=opt)
    spent=USE[arms].sum(axis=2);mir=EXTRA[arms].sum(axis=2)
    assert (spent<=limits[:,None,:]).all() and (mir<=h//4).all()
    return y.sum(axis=2),mir,spent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('data',type=Path);ap.add_argument('phase',choices=('tune','test'))
    a=ap.parse_args();sys.path.insert(0,str(a.project.resolve()));tune=a.phase=='tune'
    x=panels(a.data/('atlas' if tune else 'atlas_feedback'),'dev' if tune else 'test');p,h,n,_=x.shape
    r=12 if tune else 24
    rs,ws=contexts(x,'dev' if tune else 'test',3,r,0 if tune else 43)
    limits=np.stack([caps(i,h) for i in range(p)])
    kinds=('binary','aggregate','aggregate_ordered','round','rtt','recent')
    base=Settings('joint',.15,.2); calfile=a.data/'feedback_calibration.json'
    if tune:
        calibration={}
        for signal in kinds:
            rows=[]
            mode={'round':'round_lin','recent':'recent'}.get(signal,'joint')
            for alpha in (0.,.035,.15,.55):
                for eta in (.04,.2):
                    for rho in (.4,1.):
                        settings=replace(base,mode=mode,alpha=alpha,eta=eta,ridge=rho)
                        ys,_,_=series(x,rs,ws,limits,settings,signal)
                        rows.append(dict(mean=float(ys.mean()),settings=settings.__dict__))
            calibration[signal]=dict(chosen=max(rows,key=lambda z:z['mean']),grid=rows)
        calfile.write_text(json.dumps(calibration,indent=2)+'\n');return
    calibration=json.loads(calfile.read_text());plans=[]
    for signal in kinds:
        s=Settings(**calibration[signal]['chosen']['settings']);plans.append((signal+'_tuned',s,signal))
        mode={'round':'round_lin','recent':'recent'}.get(signal,'joint');plans.append((signal+'_matched',replace(base,mode=mode),signal))
        if signal=='rtt':plans.append(('rtt_no_bonus',replace(s,alpha=0),signal))
    rows=[]
    for name,settings,signal in plans:
        ys,mir,spent=series(x,rs,ws,limits,settings,signal)
        for i in range(p):
            for j in range(r): rows.append((i,j,name,ys[i,j],mir[i,j],*spent[i,j]))
    with (a.data/'feedback_methods.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(('panel','draw','policy','accepted','mirror','fi','br','us','ug'));wr.writerows(rows)
    bounds=[]
    for i in range(p):
        unique,by=clubs(limits[i],h);_,values=tabulate(x[i],ws,3,by)
        bounds.append(dict(panel=i,prebook=dual(values,np.r_[h//4,limits[i]]/h,unique)[0]))
    (a.data/'feedback_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')

if __name__=='__main__':main()
