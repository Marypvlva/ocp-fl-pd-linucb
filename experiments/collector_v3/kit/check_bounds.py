"""Independent finite example for the pre-reveal information upper bound."""
from itertools import combinations,product
from fractions import Fraction
from math import comb
import numpy as np
from scipy.optimize import linprog
from treasury import dual


def primal(values,costs,budget):
    h,a=values.shape
    A_eq=np.kron(np.eye(h),np.ones((1,a)))
    A_ub=np.tile(costs.T,(1,h))
    ans=linprog(-values.ravel(),A_eq=A_eq,b_eq=np.ones(h),A_ub=A_ub,b_ub=budget,bounds=(0,1),method='highs')
    assert ans.success
    return -ans.fun


def main():
    rng=np.random.default_rng(94877)
    cases=0
    for h in (1,2,4,13):
        val=rng.random((h,4));cost=np.array([[0,0],[0,1],[1,0],[1,1]])
        for b in ((0,0),(h,h),(h//2,h//3)):
            opt,_=dual(val,np.array(b)/h,cost)
            assert abs(opt-primal(val,cost,b))<1.e-8
            cases+=1
    # Two regional collectors, both free as a single route. A duplicated round costs 1.
    k=2;n=16;t=480;b=120
    members=list(combinations(range(n),k))
    mix=Fraction(sum((i<8)!=(j<8) for i,j in members),len(members))
    assert mix==Fraction(8,15)
    # A grant-before-roster controller may route freely after reveal in each prechosen class.
    pre=Fraction(t)*(1-mix)+b*mix
    # A causal policy using known topology spends on the first B mixed rosters only.
    probs=[comb(t,i)*mix**i*(1-mix)**(t-i) for i in range(t+1)]
    joint=t*(1-mix)+sum(p*min(i,b) for i,p in enumerate(probs))
    assert pre==288 and abs(float(joint)-344)<1e-8
    print(f'{cases} primal-dual checks passed; mix={mix}, before={float(pre):.8f}, after={float(joint):.8f}, delta={float(joint-pre):.8f}.')

if __name__=='__main__': main()
