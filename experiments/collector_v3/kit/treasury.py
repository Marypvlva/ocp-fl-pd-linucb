"""Dual of the fractional budget allocation. Used by matched controls and bounds."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from model import USE,EXTRA
COST = np.column_stack((EXTRA,USE)).astype(float)


def dual(values, target, costs=COST):
    """Independent round maxima with resource prices, exact LP relaxation."""
    h,na=values.shape
    nr=costs.shape[1]
    n=h*na
    rows=np.arange(n)
    si=np.tile(costs,(h,1))
    rr,cc=np.nonzero(si)
    idx=np.r_[rr,rows]
    col=np.r_[cc,nr+np.repeat(np.arange(h),na)]
    coeff=np.r_[-si[rr,cc],-np.ones(n)]
    matrix=coo_matrix((coeff,(idx,col)),shape=(n,nr+h)).tocsr()
    result=linprog(np.r_[np.asarray(target)*h,np.ones(h)],A_ub=matrix,b_ub=-values.ravel(),
                   bounds=[(0,None)]*nr+[(None,None)]*h,method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun),result.x[:nr]


def shadow(history,remaining):
    return dual(history,remaining)[1]
