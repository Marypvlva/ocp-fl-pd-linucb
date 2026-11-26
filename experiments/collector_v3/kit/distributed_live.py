"""Frozen, resumable sequential crossover experiments on RemoteCluster."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time

import numpy as np
from model import ARMS, Policy, Settings, SLA
from gateway_study import caps
from lease import deploy, reserve
from meter import settled_metrics


def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2)+'\n')
    os.replace(tmp, path)


def sources(root):
    paths = [*root.glob('deploy/timeweb/*'),root/'pyproject.toml',root/'requirements-linux-x86_64.lock',*root.glob('adaptive_planning/*.py'), *root.glob('experiments/collector_v3/kit/*'),
             *root.glob('contracts/*.sol'), *root.glob('contracts/artifacts/*.json')]
    return {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file()}


def plan(root, placement, output, profile, rounds=None, seeds=None, modes=None, network_mode="natural", chain_mode="local_evm", windows=None, external_digest=None):
    from adaptive_planning.distributed import load_placement
    windows=windows or [80,130,200]
    if len(windows)!=3 or windows!=sorted(set(windows)) or not all(type(x) is int and 0<x<=60000 for x in windows):
        raise ValueError("Choose three increasing positive millisecond windows")
    n = 16 if profile == 'confirmatory' else 4
    mapping = load_placement(placement, n)
    count = 12 if profile == 'confirmatory' else 2
    roster_seeds = list(range(81101, 81101+count)) if seeds is None else seeds
    horizon = rounds or (480 if profile == 'confirmatory' else 24)
    if profile == 'confirmatory' and len({mapping[f'e{i}'] for i in range(4)}) != 4:
        raise ValueError('Confirmatory profile requires four distinct gateway hosts')
    modes = modes or ['joint','no_ids','scalar','recent','round_lin','lp_mean','greedy']
    order = {str(s):np.random.default_rng(s+7).permutation(modes).tolist() for s in roster_seeds}
    config = dict(version=1, profile=profile, population=n, k=3 if n==16 else 2,
        horizon=horizon, seeds=roster_seeds, modes=modes, order=order,
        placement=json.loads(Path(placement).read_text()), source_hashes=sources(root),
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        calibration=json.loads((root/'experiments/collector_v3/data/gateway_calibration.json').read_text()),
        bytes_per_episode=250_000*horizon, deadline_s=3., end_to_end_deadline_s=600.,
        reward_delay=2, windows_ms=windows, external_config_digest=external_digest, network_mode=network_mode, chain_mode=chain_mode, created=time.time(),
        paired_scope='Same roster seeds, initial model, empty history, quotas and settings; randomized sequential crossover. Natural network outcomes differ by time.',
        uncertainty='Bootstrap whole seed/time blocks. Confirmatory sample size must be reassessed from pilot variance in a new plan before collecting confirmatory outcomes.')
    output.mkdir(parents=True,exist_ok=False)
    dump(output/'plan.json',config)
    (output/'plan.sha256').write_text(sha256((output/'plan.json').read_bytes()).hexdigest()+'\n')
    print(output/'plan.json')


def wait_blocks(w3, block, confirmations, timeout=300):
    start=time.perf_counter()
    while w3.eth.block_number-block+1 < confirmations or w3.eth.get_block('finalized').number < block:
        if time.perf_counter()-start > timeout:
            raise TimeoutError('Chain finality timeout')
        time.sleep(.25)


def external_chain(root, config, mode):
    from web3 import Web3
    from adaptive_planning.ocp_chain import OcpChain
    from adaptive_planning.ocp_randomness import BeaconSpec
    w3=Web3(Web3.HTTPProvider(config['rpc'],request_kwargs={'timeout':15}))
    from deploy.timeweb.chain_setup import signing
    signing(w3)
    w3.eth.get_block('finalized')
    entry=config['sources'][mode]
    artifact=json.loads((root/'contracts/artifacts/OcpfScheduler.json').read_text())
    # Externally provisioned research RPC must have the experiment accounts unlocked.
    # No credentials are saved in the plan or logs. Do not use a production funded account.
    source=w3.eth.contract(address=entry['contract'],abi=artifact['abi'])
    chain=OcpChain(w3,source,entry['oracle'],BeaconSpec(**config['beacon']),config['confirmations'])
    if source.functions.lastRound().call() != 0:
        raise ValueError('External experiment requires fresh scheduling contracts per episode')
    return chain


def ceremony(chain, rd, seed, n, external):
    keys=[sha256(f'timeweb-v3/{seed}/{rd}/{i}'.encode()).digest() for i in range(n)]
    if not external:
        return chain.run_local_ceremony(rd,seeds=keys)
    from adaptive_planning.ocp_randomness import DrandBeacon
    now=chain.timestamp
    commit_end, reveal_end=now+max(60,n*5),now+max(120,n*10)
    beacon_round=chain.spec.round_after(reveal_end)
    chain.open_round(rd,commit_end,reveal_end,beacon_round)
    for i,key in enumerate(keys):chain.commit(rd,i,key)
    def until(stamp):
        start=time.perf_counter()
        while chain.timestamp < stamp:
            if time.perf_counter()-start>300:raise TimeoutError('Chain clock stalled')
            time.sleep(.25)
    until(commit_end)
    for i,key in enumerate(keys):chain.reveal(rd,i,key)
    until(chain.spec.time_for(beacon_round))
    receipt=chain.finalize(rd,DrandBeacon(chain.spec).fetch(beacon_round))
    wait_blocks(chain.w3,receipt.blockNumber,chain.finality)
    return chain.certificate(rd)


def episode(root, output, config, seed, mode, runtime):
    from adaptive_planning.ocp_chain import OcpChain
    from adaptive_planning.tcp_runtime import flatten_model, make_model, Fault
    from adaptive_planning.distributed import RemoteCluster, load_placement
    from adaptive_planning.distributed_control import ControlClient, ControlServer, Ledger, RemoteBudget
    from adaptive_planning.lease_guard import DistributedShaper, LeaseGuard, read_evidence
    from adaptive_planning.training import make_workers, Q_SCALE
    from eth_hash.auto import keccak
    SLA[:]=config['windows_ms']
    n,h,k=config['population'],config['horizon'],config['k']
    directory=output/f'{seed}_{mode}'
    if (directory/'complete.json').exists():return
    if directory.exists():
        raise ValueError(f'Interrupted episode {directory}. Preserve it and use a new run directory; credit is never reset automatically.')
    directory.mkdir()
    external=runtime.get('chain')
    chain=external_chain(root,external,mode) if external else OcpChain.local(n,k,genesis_timestamp=2145916800)
    if len(chain.clients)!=n or chain.size!=k:raise ValueError('Chain population differs from frozen plan')
    groups,val,_=make_workers(n)
    settings=Settings(**config['calibration']['records'][mode]['chosen']['settings'])
    limit=caps(config['seeds'].index(seed),h)[None,:]
    prices=np.array([config['calibration']['prices'][str(config['seeds'].index(seed)%4)]])
    policy=Policy(settings,1,n,k,h,h//4,53,limit,prices)
    model=make_model(17,64,.5)
    initial=keccak(flatten_model(model)).hex()
    binding=dict(settings=settings.__dict__,plan_sha256=sha256((output/'plan.json').read_bytes()).hexdigest(),seed=seed,mode=mode,windows=config['windows_ms'],bytes=config['bytes_per_episode'])
    book,cfg=deploy(chain,binding,h//4,limit[0])
    lease_pin=dict(chain_id=chain.chain_id,source=chain.contract.address,lease=book.address,
        source_code=chain.code_hash.hex(),lease_code=keccak(bytes(chain.w3.eth.get_code(book.address))).hex(),cfg=cfg.hex(),confirmations=chain.finality,require_finalized=bool(external),windows=[int(d) for d in SLA])
    names=['coordinator']+[f'w{i}' for i in range(n)]+[f'e{i}' for i in range(4)]
    credentials={name:secrets.token_bytes(32) for name in names}
    ledger=Ledger(directory/'budget.sqlite',config['bytes_per_episode'])
    service=ControlServer((runtime['bind'],runtime['port']),ledger,credentials,
        lambda request:read_evidence(chain.w3,lease_pin,request),runtime.get('cert'),runtime.get('key')).start()
    url=runtime['url']
    if url=='loopback':url=f'http://127.0.0.1:{service.http.server_port}'
    clients={name:dict(url=url,owner=name,key=credentials[name].hex(),ca=runtime.get('ca'),timeout=5) for name in names}
    cguard=LeaseGuard('coordinator',lease_pin,lambda request:read_evidence(chain.w3,lease_pin,request))
    coordinator_shaper=DistributedShaper(RemoteBudget(ControlClient(**clients['coordinator'])),cguard)
    placement=config['placement'];mapping=load_placement(placement,n)
    classes=np.resize(np.arange(3),h);np.random.default_rng(seed+200).shuffle(classes)
    rows=[]
    try:
        with RemoteCluster(groups,placement=mapping,hidden=64,rpc_timeout=1.5,selection_authority=chain,
                control=clients,lease_pin=lease_pin,source_hashes=config['source_hashes'],chain_rpc=external['rpc'] if external else None,
                **placement.get('ssh',{})) as cluster:
            cluster.shaper=coordinator_shaper
            dump(directory/'role-environment.json',cluster.role_environment)
            dump(directory/'worker-keys.json',{str(i):key.public_bytes_raw().hex() for i,key in cluster.public_keys.items()})
            for t,c in enumerate(classes):
                rd=t+1; started=time.perf_counter(); wall_started=time.time()
                ticket=ceremony(chain,rd,seed,n,bool(external));after_final=time.perf_counter()
                members=list(chain.authorize(ticket,rd,n))
                arm=int(policy.decide(c,np.array([members]))[0]);gateways=ARMS[arm]
                lease=reserve(chain,book,cfg,ticket,gateways,int(SLA[c]))
                if external:wait_blocks(chain.w3,lease['block'],chain.finality)
                else:chain.w3.provider.ethereum_tester.mine_blocks(chain.finality-1)
                request=dict(round=rd,mask=lease['mask'],window=int(SLA[c]),digest=lease['digest'],
                    block=lease['block'],block_hash=chain.w3.eth.get_block(lease['block']).hash.hex(),selected=members,expires_at=wall_started+config['end_to_end_deadline_s'])
                cguard.install(request);acks=cluster.install_grant(request);after_booking=time.perf_counter()
                before=coordinator_shaper.byte_budget.state()['spent'];old=flatten_model(model)
                remaining=config['end_to_end_deadline_s']-(time.perf_counter()-started)
                if remaining<=0:raise TimeoutError('End-to-end deadline exhausted before execution')
                fault=Fault(failed_edges=(rd%4,)) if runtime['network_mode']=='controlled' and rd%5==0 else None
                result=cluster.run_round(model,'F+P',selection=ticket,deadline_s=min(config['deadline_s'],remaining),
                    proof_collectors=gateways,proof_budget_ms=float(SLA[c]),emulated_link_ms=None,fault=fault)
                after_data=time.perf_counter()
                # Verification against the exact gradient is outside the measured round.
                met,state=settled_metrics(cluster,rd,before)
                digest=chain.record_model(rd,flatten_model(model))
                if external:wait_blocks(chain.w3,chain.model_blocks[rd],chain.finality)
                after_commit=time.perf_counter()
                drift=(time.time()-wall_started)-(after_commit-started)
                if abs(drift)>.005:raise ValueError('Coordinator wall clock stepped during round')
                ontime=after_commit-started<=config['end_to_end_deadline_s']
                if not ontime:raise TimeoutError('End-to-end deadline exceeded at model finality')
                if result.accepted:
                    total=sum(len(groups[i][1]) for i in members)
                    reference=np.stack([np.rint(make_gradient(old,model,groups[i])*(len(groups[i][1])/total)*Q_SCALE).astype(np.int64) for i in members]).sum(axis=0)
                    np.testing.assert_array_equal(result.vector,reference)
                elif old!=flatten_model(model):raise AssertionError('Rejected update changed model')
                timings=result.metrics.get('collector_ms',{str(g):{} for g in gateways})
                samples=np.full((1,k,2),-1.)
                for j,g in enumerate(gateways):
                    for i,member in enumerate(members):
                        samples[0,i,j]=timings[str(g)].get(str(member),-1.)
                policy.observe(samples,np.array([result.accepted]))
                row=dict(round=rd,seed=seed,mode=mode,selected=members,gateways=list(gateways),ticket=ticket,
                    grant=request,node_grant_checks=acks,accepted=result.accepted,errors=result.errors,
                    network_mode=runtime['network_mode'],emulated_link_ms=None,chain_mode='external_rpc' if external else 'local_evm',
                    physical_hosts=sorted(set(mapping.values())),initial_model_hash=initial,previous_model_hash=keccak(old).hex(),model_hash=digest,
                    validation=list(map(float,model.score(*val))),window_ms=int(SLA[c]),byte_ledger=state,telemetry=met,metrics=result.metrics,
                    started_at=wall_started,total_round_s=after_commit-started,roster_finality_s=after_final-started,
                    booking_and_node_validation_s=after_booking-after_final,data_and_update_s=after_data-after_booking,
                    accounting_and_model_finality_s=after_commit-after_data,
                    mirror=int(policy.spent[0]),gateway_spent=policy.gateway_spent[0].tolist())
                rows.append(row);dump(directory/f'round-{rd:04}.json',row)
            dump(directory/'chain.json',chain.export())
            dump(directory/'complete.json',dict(rounds=h,accepted=sum(r['accepted'] for r in rows),initial_model_hash=initial,finished=time.time()))
    except BaseException as exc:
        dump(directory/'failure.json',dict(type=type(exc).__name__,message=str(exc)[:500],ledger=ledger.apply('coordinator','state',{}),at=time.time()))
        raise
    finally:
        service.close()


def make_gradient(old,model,group):
    from adaptive_planning.tcp_runtime import make_model,load_model
    reference=make_model(17,64,.5);load_model(reference,old)
    return reference.gradient(*group)


def run(root, output, runtime):
    if sha256((output/'plan.json').read_bytes()).hexdigest() != (output/'plan.sha256').read_text().strip():
        raise ValueError('Frozen plan digest mismatch')
    config=json.loads((output/'plan.json').read_text())
    if config['source_hashes']!=sources(root):raise ValueError('Source differs from frozen plan')
    if config['network_mode']!=runtime['network_mode'] or config['chain_mode']!=('external_rpc' if runtime.get('chain') else 'local_evm'):
        raise ValueError('Network or chain mode differs from frozen plan')
    if runtime.get('chain') and sha256(json.dumps(runtime['chain'],sort_keys=True).encode()).hexdigest()!=config['external_config_digest']:
        raise ValueError('External chain configuration differs from frozen plan')
    if runtime.get('chain') and (len(config['seeds'])!=1 or len(config['modes'])!=1):
        raise ValueError('External consensus experiment uses one fresh source contract and policy; natural drand rosters cannot be assumed paired across independent ceremonies')
    lock=output/'RUNNING'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        dump(output/'environment.json',dict(python=sys.version,platform=platform.platform(),runtime={k:v for k,v in runtime.items() if k not in ('key','chain')},started=time.time()))
        for seed in config['seeds']:
            for mode in config['order'][str(seed)]:episode(root,output,config,seed,mode,runtime)
    finally:lock.unlink()


def main(root):
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--control-url',required=True);parser.add_argument('--control-bind',default='0.0.0.0');parser.add_argument('--control-port',type=int,default=9443)
    parser.add_argument('--tls-cert');parser.add_argument('--tls-key');parser.add_argument('--tls-ca');parser.add_argument('--chain-config',type=Path)
    parser.add_argument('--network-mode',choices=('natural','controlled'),default='natural')
    a=parser.parse_args(sys.argv[1:]);runtime=dict(url=a.control_url,bind=a.control_bind,port=a.control_port,cert=a.tls_cert,key=a.tls_key,ca=a.tls_ca,network_mode=a.network_mode)
    if a.chain_config:runtime['chain']=json.loads(a.chain_config.read_text())
    run(root.resolve(),a.plan.resolve().parent,runtime)
