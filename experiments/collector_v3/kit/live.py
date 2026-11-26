"""Full local EVM/TCP training with causal collector choices and real byte caps."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np
from model import ARMS,Policy,Settings,SLA
from gateway_study import caps
from meter import settled_metrics

MODES=('joint','no_ids','scalar','recent','round_lin','lp_mean')
SEEDS=(37101,37102,37103,37104)
H=48
B_LIMIT=12_000_000


def physical_links(seed):
    rng=np.random.default_rng(seed+13357)
    centers=np.array([[18,260,95,245],[27,250,228,116],[255,26,101,255],[250,22,240,110]])
    traces=centers+rng.uniform(-6,6,(H,4,4))
    for t in range(H):
        if t%14 in (10,11):
            traces[t,:,((t//14)+seed)%4]+=90
    return traces


def run(root, data, seed, sample, modes, actual, cached=False):
    sys.path.insert(0,str(root.resolve()))
    from adaptive_planning.ocp_chain import OcpChain
    from adaptive_planning.tcp_runtime import Cluster,flatten_model,make_model
    from adaptive_planning.training import Q_SCALE,make_workers
    from eth_hash.auto import keccak
    cal=json.loads((data/'gateway_calibration.json').read_text())
    outdir=data/('live_atlas' if actual else 'live')
    outdir.mkdir(exist_ok=True)
    def completed(mode):
        f=outdir/f'{seed}_{mode}.json'
        if not f.exists():
            return False
        rows=json.loads(f.read_text())
        if len(rows)!=sample:
            raise ValueError('Different episode length is already stored')
        return True
    if all(completed(mode) for mode in modes):
        return
    groups,val,_=make_workers(4)
    chain=OcpChain.local(4,2,genesis_timestamp=2145916800)
    tickets=[]
    for t in range(sample):
        keys=[sha256(f'joint-proof-3/{seed}/{t}/{j}'.encode()).digest() for j in range(4)]
        tickets.append(chain.run_local_ceremony(t+1,seeds=keys))
    authority=chain
    if cached:
        from finalized_view import FinalizedView
        authority=FinalizedView(chain,tickets)
    tester=chain.w3.provider.ethereum_tester
    priors=cal['prices'][str(SEEDS.index(seed)%4)]
    limits=caps(SEEDS.index(seed),sample)[None,:]
    if actual:
        from experiment import panels
        links=panels(data/'atlas','test')[0,:sample,:4,:]
    else:
        links=physical_links(seed)[:sample]
    classes=np.resize(np.arange(3),sample)
    rng=np.random.default_rng(seed+200)
    rng.shuffle(classes)
    for mode in rng.permutation(modes).tolist():
        output=outdir/f'{seed}_{mode}.json'
        if completed(mode):
            continue
        snap=tester.take_snapshot()
        receipts_before=len(chain.receipts)
        events_before=len(chain.model_events)
        gate=Policy(Settings(**cal['records'][mode]['chosen']['settings']),1,4,2,sample,sample//4,53,limits,np.array([priors]))
        model=make_model(17,64,.5)
        rows=[]
        if actual:
            from lease import deploy,reserve,check_rejections
            book,cfg=deploy(chain,gate.s.__dict__,sample//4,limits[0])
        with Cluster(groups,hidden=64,rpc_timeout=.6,selection_authority=authority,app_byte_budget=B_LIMIT) as c:
            for t,(ticket,slack,latencies) in enumerate(zip(tickets,classes,links,strict=True)):
                members=list(authority.authorize(ticket,t+1,4))
                arm=int(gate.decide(slack,np.array([members]))[0])
                gateways=ARMS[arm]
                reservation=None
                if actual:
                    reservation=reserve(chain,book,cfg,ticket,gateways,int(SLA[slack]))
                    guarded=[book.functions.siteLeft(i).call() for i in range(4)]
                    assert guarded==(limits[0]-gate.gateway_spent[0]).tolist()
                    assert book.functions.mirrorLeft().call()==sample//4-int(gate.spent[0])
                    if t==0: check_rejections(chain,book,ticket,gateways,int(SLA[slack]))
                old=flatten_model(model)
                total=sum(len(groups[i][1]) for i in members)
                exact=np.stack([np.rint(model.gradient(*groups[i])*(len(groups[i][1])/total)*Q_SCALE).astype(np.int64) for i in members]).sum(axis=0)
                previous=c.shaper.byte_budget.state()['spent']
                res=c.run_round(model,'F+P',selection=ticket,deadline_s=1.5,proof_collectors=gateways,
                                proof_budget_ms=float(SLA[slack]),emulated_link_ms=latencies)
                if res.accepted:
                    np.testing.assert_array_equal(res.vector,exact)
                else:
                    assert old==flatten_model(model)
                receipts=res.metrics['collector_ms']
                samples=np.full((1,2,2),-1.)
                for j,g in enumerate(gateways):
                    for n,i in enumerate(members):
                        if str(i) in receipts[str(g)]:
                            samples[0,n,j]=receipts[str(g)][str(i)]
                covered=(samples[0]>0).any(axis=1).all()
                assert bool(covered)==res.accepted
                gate.observe(samples,np.array([res.accepted]))
                met,state=settled_metrics(c,t+1,previous)
                assert not state['denied_frames'] and state['spent']<=B_LIMIT
                digest=chain.record_model(t+1,flatten_model(model))
                assert digest==keccak(flatten_model(model)).hex()
                assert chain.contract.functions.roundMeta(t+1).call()[-1].hex()==digest
                rows.append(dict(seed=seed,mode=mode,t=t+1,members=members,act=list(gateways),accepted=res.accepted,
                                 cap=limits.tolist()[0],count=gate.gateway_spent.tolist()[0],mirror=int(gate.spent[0]),
                                 slack=int(SLA[slack]),errors=res.errors,observed=receipts,
                                 tx=met['app_tx_bytes'],ledger=state['spent'],hash=digest,block=ticket['block_hash'],
                                 reservation=reservation,certificate_lookup='finalized_cache' if cached else 'events',round_s=res.metrics['round_wall_s']))
            assert sum(row['tx'] for row in rows)==rows[-1]['ledger']
        tmp=output.with_suffix('.tmp')
        tmp.write_text(json.dumps(rows,indent=1))
        tmp.replace(output)
        print(output.name,sum(x['accepted'] for x in rows),rows[-1]['ledger'],flush=True)
        tester.revert_to_snapshot(snap)
        del chain.receipts[receipts_before:]
        del chain.model_events[events_before:]
        chain.model_blocks.clear()


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('data',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS));p.add_argument('--modes',nargs='+',default=list(MODES))
    p.add_argument('--cached-finality',action='store_true');p.add_argument('--atlas',action='store_true');p.add_argument('--rounds',type=int,default=H)
    a=p.parse_args()
    for s in a.seeds: run(a.root,a.data,s,a.rounds,a.modes,a.atlas,a.cached_finality)

if __name__=='__main__':
    if '--plan' in sys.argv:
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
        from distributed_live import main as distributed_main
        distributed_main(Path(__file__).resolve().parents[3])
    else:
        main()
