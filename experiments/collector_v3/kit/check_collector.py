"""Exercise the collector's normal signature, duplicate, partial-coverage and authorization boundaries."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main():
    a=argparse.ArgumentParser();a.add_argument('tree',type=Path);args=a.parse_args();sys.path.insert(0,str(args.tree.resolve()))
    from adaptive_planning import collector_routes as transport,randomized_check as proof
    from adaptive_planning.clock import now
    from adaptive_planning.tcp_transport import ProtocolError
    chall=proof.Challenge(bytes(16),1,3,bytes([5])*32,7,(bytes([1])*32,bytes([2])*32))
    keys=[Ed25519PrivateKey.generate() for _ in range(2)]
    cmd={'hash':chall.candidate_hash.hex(),'seeds':[s.hex() for s in chall.seeds],'dimension':7}
    class Fixture:
        def __init__(self,case):
            self.session=chall.session.hex();self.round_id=3;self.public_keys=[k.public_key() for k in keys]
            self.keys={'coordinator':{'w0':bytes([7])*32,'w1':bytes([8])*32}};self.case=case
        def call(self,receiver,rd,kind,body,deadline):
            g=int(receiver[1:]);i=body['worker'];key=self.keys['coordinator'][f'w{i}']
            assert kind=='relay_projection' and rd==3 and body['token']==transport.token(key,self.session,rd,i,g,body['command'])
            if self.case=='complement' and i!=g: raise TimeoutError()
            tags=[i+1,13];signer=i;context=chall
            if self.case=='wrong_signer' and i==1: signer=0
            if self.case=='old_candidate' and i==1:context=replace(chall,candidate_hash=bytes([6])*32)
            if self.case=='conflict' and i==0 and g==1:tags[0]+=1
            msg=proof.encoded_message(context,i,tags)
            return dict(tags=tags,prepare_s=0.,prepare_cpu_s=0.),keys[signer].sign(msg)
    for case,expected in [('duplicates',2),('complement',2),('wrong_signer',1),('old_candidate',1),('past_deadline',0)]:
        f=Fixture(case);metrics={};deadline=now()+(-1 if case=='past_deadline' else 2.)
        replies,errors=transport.collect(f,[0,1],[0,1],cmd,chall,deadline,None,metrics)
        assert len(replies)==expected
        assert sum(len(v) for v in metrics['collector_ms'].values())=={'duplicates':4,'complement':2,'wrong_signer':2,'old_candidate':2,'past_deadline':0}[case]
    try:transport.collect(Fixture('conflict'),[0,1],[0,1],cmd,chall,now()+2.,None,{})
    except ProtocolError:pass
    else:raise AssertionError('Conflicting signed replies counted')
    key=bytes([3])*32
    genuine=transport.token(key,'session',1,1,0,cmd)
    for fields in [('other',1,1,0,cmd),('session',2,1,0,cmd),('session',1,0,0,cmd),('session',1,1,1,cmd),('session',1,1,0,dict(cmd,dimension=8))]:
        assert transport.token(key,*fields)!=genuine
    print('Coverage, timely receipts, duplicate handling, expected signer, candidate binding, conflict rejection and route-token binding passed.')

if __name__=='__main__':main()
