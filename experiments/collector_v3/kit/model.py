"""Partial-feedback policy for leasing up to two registered proof collectors."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

CHANNELS = 4
SLA = np.array([80.,130.,200.])
ARMS = [(g,) for g in range(CHANNELS)] + list(combinations(range(CHANNELS),2))
LEFT = np.array([a[0] for a in ARMS])
RIGHT = np.array([a[-1] for a in ARMS])
EXTRA = np.arange(len(ARMS)) >= CHANNELS
USE = np.array([[int(g in arm) for g in range(CHANNELS)] for arm in ARMS])


@dataclass(frozen=True)
class Settings:
    mode: str = 'joint'
    alpha: float = .55
    eta: float = .08
    initial_price: float = .25
    ridge: float = 1.
    delay: int = 2
    epsilon: float = .05
    use_current_roster: bool = True
    price_rule: str = 'local' 
    allowed_arms: tuple | None = None
    lp_window: int = 96
    # Opt-in research variants. Historical policies retain exactly their old semantics.
    memory: str = 'none'
    discount: float = .95
    window: int = 32
    memory_bonus: str = 'weighted'
    feedback_rule: str = 'legacy'


class Policy:
    """Each row is an independent run. The learner never receives an unleased path."""
    def __init__(self, settings, runs, population, k, horizon, budget, seed=43, gateway_caps=None, fixed_prices=None):
        if horizon <= 0 or not 0 <= budget <= horizon or not 0 < k <= population:
            raise ValueError('Invalid policy dimensions or mirror budget')
        if type(settings.lp_window) is not int or settings.lp_window<2:
            raise ValueError('LP history window must be at least two')
        if settings.allowed_arms is not None and (not settings.allowed_arms or
                any(type(a) is not int or not 0 <= a < len(ARMS) for a in settings.allowed_arms)):
            raise ValueError('Invalid allowed arms')
        self.s = settings
        self.r, self.n, self.k, self.h, self.budget = runs,population,k,horizon,budget
        self.b = np.arange(runs)
        shape = (runs,population,CHANNELS,len(SLA))
        self.done = np.zeros(shape)
        self.wins = np.zeros(shape)
        self.design = np.zeros(shape)
        self.recent = np.full(shape,.5)
        self.price = np.full(runs,settings.initial_price)
        self.spent = np.zeros(runs,dtype=int)
        self.caps = gateway_caps
        self.gateway_spent=np.zeros((runs,CHANNELS),dtype=int)
        self.gprice=np.zeros_like(self.gateway_spent,dtype=float)
        self.fixed_prices=fixed_prices
        self.lp_scores=[]
        self.rng=np.random.default_rng(seed)
        self.pending=[]
        self.t=0
        self.dim = 1+population+len(SLA)
        if self.s.mode=='round_lin':
            self.inv=np.broadcast_to(np.eye(self.dim)/2,(runs,len(ARMS),self.dim,self.dim)).copy()
            self.res=np.zeros((runs,len(ARMS),self.dim))
        self.hist = np.zeros((runs,horizon))
        self.bank = np.zeros(runs)

    def _index(self, members, gateways, c):
        return self.b[:,None],members,gateways[:,None],c

    def _mature(self,payload):
        c,roster,arm,readings,reward,features=payload[:6]
        attempted=payload[6] if len(payload)>6 else np.ones_like(readings,dtype=bool)
        # One reading per selected worker and attempted channel only.
        for branch,g in enumerate((LEFT[arm],RIGHT[arm])):
            used = np.ones(self.r,bool) if branch==0 else EXTRA[arm]
            obs = readings[:,:,branch]
            for i,d in enumerate(SLA):
                known = used[:,None] & attempted[:,:,branch] & ((obs>0) | (d<=SLA[c]))
                success = known & (obs>0) & (obs<=d)
                ix = self._index(roster,g,i)
                self.done[ix] += known
                if i!=c:
                    self.design[ix] += known
                self.wins[ix] += success
                self.recent[ix] = np.where(known,.65*success+.35*self.recent[ix],self.recent[ix])
            # A booked but unexecuted projection is not a failed route measurement.
            # Remove its pending design only when this delayed feedback matures.
            self.design[self._index(roster,g,c)] -= used[:,None] & ~attempted[:,:,branch]
        if self.s.mode=='round_lin':
            self.res[self.b,arm] += features*(reward-.5)[:,None]

    def _score(self,c,roster):
        n=self.design[:,:,:,c]
        completed=self.done[:,:,:,c]
        wins=self.wins[:,:,:,c]
        mu=.5+(wins-.5*completed)/(self.s.ridge+n)
        self.path_mean_prediction=np.clip(mu[self.b[:,None],roster,:],0,1)
        self.path_pooled_prediction=np.clip(mu.mean(axis=1),0,1)
        mode=self.s.mode
        if mode=='thompson':
            pred=self.rng.beta(wins+.5,completed-wins+.5)
        elif mode=='recent':
            pred=self.recent[:,:,:,c]
        else:
            a=0 if mode in ('greedy','eps','filter','lp_mean') else self.s.alpha
            pred=np.minimum(1.,mu+a/np.sqrt(self.s.ridge+n))
        if mode=='no_ids' or not self.s.use_current_roster:
            pred=pred.mean(axis=1)[:,None,:].repeat(self.k,axis=1)
        else:
            pred=pred[self.b[:,None],roster,:]
        union=1-(1-pred[:,:,LEFT])*(1-pred[:,:,RIGHT])
        union[:,:,:CHANNELS]=pred
        return np.prod(union,axis=1)

    def decide(self,c,roster,forced_arm=None):
        while self.pending and self.pending[0][0]<=self.t:
            _,payload=self.pending.pop(0)
            self._mature(payload)
        mode=self.s.mode
        price_before=self.price.copy()
        gateway_price_before=self.gprice.copy()
        features=None
        if mode=='round_lin':
            features=np.zeros((self.r,self.dim))
            features[:,0]=1
            features[:,1:1+self.n]=-self.k/self.n/np.sqrt(self.k)
            features[self.b[:,None],1+roster]+=1/np.sqrt(self.k)
            features[:,1+self.n+c]=1
            vec=np.einsum('raij,rj->rai',self.inv,features)
            base=np.clip(.5+np.einsum('rai,rai->ra',vec,self.res),0,1)
            score=base+self.s.alpha*np.sqrt(np.maximum(0,np.einsum('rai,ri->ra',vec,features)))
        elif mode in ('uniform','static_rank'):
            score=np.zeros((self.r,len(ARMS)))
            self.path_mean_prediction=None
            self.path_pooled_prediction=None
        else:
            score=self._score(c,roster)
        predictions=score.copy()
        fee=self.price.copy()
        if mode in ('no_price','always_single'):
            fee[:]=0.
        elif mode=='fixed':
            fee[:]=self.s.initial_price
        elif mode=='filter':
            # Same named/censored observations, posterior means, resolving by past marginal gains.
            gain=score[:,CHANNELS:].max(axis=1)-score[:,:CHANNELS].max(axis=1)
            self.hist[:,self.t]=gain
            left=self.h-self.t
            frac=np.clip((self.budget-self.spent)/left,0,1)
            window=self.hist[:,max(0,self.t-95):self.t+1]
            pos=np.minimum(window.shape[1]-1,((1-frac)*window.shape[1]).astype(int))
            fee=np.sort(window,axis=1)[self.b,pos]
        if self.s.price_rule=='zero':
            fee=np.zeros_like(fee)
        feasible=self.spent<self.budget
        if mode=='always_single':
            feasible[:]=False
        if mode=='periodic':
            self.bank+=self.budget/self.h
            force=feasible & (self.bank>=1.-1e-8)
            self.bank-=force
            fee=np.where(force,-2.,2.)
        if self.caps is not None:
            gprice=self.gprice
            if mode in ('no_price','scalar') or self.s.price_rule in ('zero','global'):
                gprice=np.zeros_like(self.gprice)
            if mode=='fixed':
                fee=self.fixed_prices[:,0]
                gprice=self.fixed_prices[:,1:]
            if mode in ('lp','lp_mean'):
                from treasury import shadow
                self.lp_scores.append(score.copy())
                if self.t % 24 == 0 and self.t>=24:
                    hist=np.stack(self.lp_scores[-self.s.lp_window:-1],axis=1)
                    rem=np.column_stack([self.budget-self.spent,self.caps-self.gateway_spent])/(self.h-self.t)
                    self.lp_prices=np.stack([shadow(hist[i],rem[i]) for i in self.b])
                if hasattr(self,'lp_prices'):
                    fee=self.lp_prices[:,0]
                    gprice=self.lp_prices[:,1:]
            score=score-gprice@USE.T
            valid=np.all(self.gateway_spent[:,None,:]+USE[None,:,:]<=self.caps[:,None,:],axis=2)
            score=np.where(valid,score,-np.inf)
        score=score-fee[:,None]*EXTRA
        score[:,CHANNELS:]=np.where(feasible[:,None],score[:,CHANNELS:],-np.inf)
        if self.s.allowed_arms is not None:
            score[:,[a for a in range(len(ARMS)) if a not in self.s.allowed_arms]]=-np.inf
        feasible_arms=np.isfinite(score)
        if not np.all(feasible_arms.any(axis=1)):
            raise ValueError('No feasible collector action')
        if mode=='uniform':
            score=np.where(feasible_arms,0.,-np.inf)
        elif mode=='static_rank':
            # Declared priority among singles; quota guards remain identical.
            score=np.where(feasible_arms,-np.arange(len(ARMS))[None,:],-np.inf)
        scores=score.copy()
        # Identical random tie sequence across deterministic ablations.
        score+=1.e-8*self.rng.random(score.shape)
        arm=np.argmax(score,axis=1)
        if mode=='eps':
            random_arm=np.argmax(np.where(np.isfinite(score),self.rng.random(score.shape),-np.inf),axis=1)
            arm=np.where(self.rng.random(self.r)<self.s.epsilon,random_arm,arm)
        if forced_arm is not None:
            if type(forced_arm) is not int or not 0<=forced_arm<len(ARMS) or not feasible_arms[:,forced_arm].all():
                raise ValueError('Forced diagnostic action is infeasible')
            arm=np.full(self.r,forced_arm,dtype=int)
        if mode=='round_lin':
            val=vec[self.b,arm]
            self.inv[self.b,arm] -= val[:,:,None]*val[:,None,:]/(1+np.sum(val*features,axis=1))[:,None,None]
        target=(self.budget-self.spent)/(self.h-self.t)
        if self.caps is not None:
            gtarget=(self.caps-self.gateway_spent)/(self.h-self.t)
            self.gateway_spent+=USE[arm]
            self.gprice=np.maximum(0.,self.gprice+self.s.eta*(USE[arm]-gtarget))
            assert np.all(self.gateway_spent<=self.caps)
        self.spent+=EXTRA[arm]
        self.price=np.maximum(0,self.price+self.s.eta*(EXTRA[arm].astype(float)-target))
        assert np.all(self.spent<=self.budget)
        if mode not in ('uniform','static_rank'):
            for branch,g in enumerate((LEFT[arm],RIGHT[arm])):
                used=np.ones(self.r,bool) if branch==0 else EXTRA[arm]
                self.design[self._index(roster,g,c)]+=used[:,None]
        self.last=(c,roster.copy(),arm.copy(),features)
        self.last_decision=dict(
            step=self.t+1, window_class=int(c), roster=roster.tolist(),
            predictions=predictions.tolist(),
            penalized_scores=[[float(v) if np.isfinite(v) else None for v in row] for row in scores],
            feasible_arms=feasible_arms.tolist(), arm=arm.tolist(),
            price_before=price_before.tolist(), price_after=self.price.tolist(),
            gateway_prices_before=gateway_price_before.tolist(),
            gateway_prices_after=self.gprice.tolist(), effective_mirror_price=fee.tolist(),
            effective_gateway_prices=gprice.tolist() if self.caps is not None else None,
            mirror_spent=self.spent.tolist(), gateway_spent=self.gateway_spent.tolist(),
            pending_labels=len(self.pending), observed_path_labels=float(self.done.sum()),
            chosen_probability=[1. / int(row.sum()) for row in feasible_arms]
                if mode=='uniform' and forced_arm is None else None,
            forced_diagnostic=forced_arm is not None,
            path_mean_prediction=self.path_mean_prediction.tolist() if mode not in ('round_lin','uniform','static_rank') else None,
            path_pooled_prediction=self.path_pooled_prediction.tolist() if mode not in ('round_lin','uniform','static_rank') else None,
        )
        return arm

    def observe(self,readings,reward,attempted=None):
        c,roster,arm,features=self.last
        if self.s.mode in ('uniform','static_rank'):
            self.t+=1
            return
        if attempted is None:
            attempted=np.ones_like(readings,dtype=bool)
        if readings.shape != (self.r,self.k,2) or attempted.shape != readings.shape:
            raise ValueError('Invalid path feedback shape')
        self.pending.append((self.t+self.s.delay,(c,roster,arm,readings.copy(),reward.copy(),features,attempted.copy())))
        self.t+=1
