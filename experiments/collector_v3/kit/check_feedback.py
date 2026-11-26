"""Check the aggregate-only comparator against exact enumeration of the hidden links."""
from itertools import product
import numpy as np
from feedback import conditional_means,FeedbackPolicy
from model import Settings


def main():
    for k in (2,3):
        for width in (1,2):
            prob=np.linspace(.14,.91,k*width).reshape(k,width)
            truth=np.zeros((2,k,width));weights=np.zeros(2)
            for state in product((0,1),repeat=k*width):
                v=np.array(state).reshape(k,width);mass=np.where(v,prob,1-prob).prod();y=int(v.any(axis=1).all())
                truth[y]+=mass*v;weights[y]+=mass
            for y in (0,1):
                a,b=conditional_means(prob[:,0][None,:],prob[:,-1][None,:],np.array([width==2]),np.array([bool(y)]))
                for j,x in enumerate((a,b)[:width]):np.testing.assert_allclose(x[0],truth[y,:,j]/weights[y],atol=1.e-12)
    for signal in ('aggregate','aggregate_ordered'):
        def new():return FeedbackPolicy(Settings('joint',.15,.2),1,4,2,30,6,31,np.array([[30,30,8,8]]),signal=signal)
        a,b=new(),new();rng=np.random.default_rng(951)
        for t in range(30):
            group=rng.choice(4,size=2,replace=False)[None,:];y=np.array([t%3==0]);d=t%3
            np.testing.assert_array_equal(a.decide(d,group),b.decide(d,group))
            a.observe(np.full((1,2,2),-1.),y);b.observe(rng.uniform(2,300,(1,2,2)),y)
        np.testing.assert_array_equal(a.wins,b.wins)
    print('Exact conditional means match exhaustive link enumeration; unobserved link labels cannot alter either aggregate-only comparator.')

if __name__=='__main__':main()
