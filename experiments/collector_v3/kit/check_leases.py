"""Local contract checks, with genuine finalized OCP certificates."""
from hashlib import sha256
from pathlib import Path
import argparse,sys
from eth_tester.exceptions import TransactionFailed
from lease import deploy,reserve,check_rejections


def main():
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);a=p.parse_args()
    sys.path.insert(0,str(a.project.resolve()))
    from adaptive_planning.ocp_chain import OcpChain
    chain=OcpChain.local(4,2,genesis_timestamp=2145916800)
    book,cfg=deploy(chain,{'case':'limits'},1,[3,1,3,3])
    def state():
        return (book.functions.lastRound().call(),book.functions.mirrorLeft().call(),
                [book.functions.siteLeft(i).call() for i in range(4)])
    def rejected(t,mask,signer):
        before=state()
        try:
            book.functions.reserve(t,mask,80).call({'from':signer})
        except TransactionFailed:
            assert state()==before
            return
        raise AssertionError('Unexpected admissible request')
    rejected(1,1,chain.oracle)
    def ticket(t):
        return chain.run_local_ceremony(t,seeds=[sha256(f'cap-test/{t}/{j}'.encode()).digest() for j in range(4)])
    t1=ticket(1); reserve(chain,book,cfg,t1,(0,1),80)
    check_rejections(chain,book,t1,(0,1),80)
    assert state()==(1,0,[2,0,3,3])
    t2=ticket(2)
    rejected(2,2,chain.oracle);rejected(2,5,chain.oracle)
    rejected(2,13,chain.oracle);rejected(2,8,chain.clients[0])
    # A modified committee must not consume a valid lease.
    wrong=dict(t2,selected=list(reversed(t2['selected'])))
    try:
        reserve(chain,book,cfg,wrong,(2,),80)
    except ValueError:
        assert state()[0]==1
    else: raise AssertionError('Incorrect certificate accepted')
    reserve(chain,book,cfg,t2,(2,),80)
    assert state()==(2,0,[2,0,2,3])
    print('Lease state, replay, unauthorised/invalid request, site cap, mirror cap and certificate binding verified.')


if __name__=='__main__':main()
