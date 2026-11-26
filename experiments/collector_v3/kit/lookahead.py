"""Estimate future demand from the known distribution of OCP rosters, without synthetic reward labels."""
import numpy as np
from model import Policy,CHANNELS,LEFT,RIGHT,EXTRA
from treasury import COST


class ForecastPrices(Policy):
    def __init__(self,*args,source='sampler',refresh=16,forecast_alpha=.0,**kwargs):
        super().__init__(*args,**kwargs)
        self.source=source
        self.refresh=refresh
        self.forecast_alpha=forecast_alpha
        self.roster_log=[]
        self.class_log=[]
        self.virtual_rng=np.random.default_rng(53491)

    def virtual_market(self):
        # Samples the same uniform k-subset law as OCP's shuffle. These draws authorize no real work.
        size=72
        if self.source=='past' and self.roster_log:
            history=np.stack(self.roster_log[-96:],axis=1)
            classes=np.array(self.class_log[-96:])
            ids=self.virtual_rng.integers(len(classes),size=size)
            members=history[:,ids]
            cl=classes[ids]
        else:
            ranks=self.virtual_rng.random((size,self.n))
            members=np.broadcast_to(np.argpartition(ranks,self.k-1,axis=1)[:,:self.k],(self.r,size,self.k))
            cl=np.tile(np.arange(3),size//3)
        b=self.b[:,None,None]
        left=self.h-self.t
        target=np.column_stack([self.budget-self.spent,self.caps-self.gateway_spent])/left
        reward=np.zeros((self.r,size,len(EXTRA)))
        for c in range(3):
            ii=np.flatnonzero(cl==c)
            if len(ii)==0:continue
            n=self.design[:,:,:,c]
            est=.5+(self.wins[:,:,:,c]-.5*self.done[:,:,:,c])/(self.s.ridge+n)
            est=np.clip(est+self.forecast_alpha/np.sqrt(self.s.ridge+n),0,1)
            pred=est[b,members[:,ii,:]]
            cov=1-(1-pred[:,:,:,LEFT])*(1-pred[:,:,:,RIGHT])
            cov[:,:,:,:CHANNELS]=pred
            reward[:,ii]=np.prod(cov,axis=2)
        prices=np.column_stack((self.price,self.gprice))
        average=np.zeros_like(prices)
        for step in range(20):
            choices=np.argmax(reward-(prices@COST.T)[:,None,:],axis=2)
            excess=COST[choices].mean(axis=1)-target
            prices=np.maximum(0.,prices+.20/np.sqrt(1+step/4)*excess)
            if step>=10:average+=prices/10
        self.price=.25*self.price+.75*average[:,0]
        self.gprice=.25*self.gprice+.75*average[:,1:]

    def decide(self,c,roster):
        while self.pending and self.pending[0][0]<=self.t:
            _,payload=self.pending.pop(0); self._mature(payload)
        if self.t and self.t%self.refresh==0:
            self.virtual_market()
        self.roster_log.append(roster.copy());self.class_log.append(c)
        return super().decide(c,roster)
