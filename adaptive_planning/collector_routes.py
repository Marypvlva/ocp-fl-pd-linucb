"""Deliver one coordinator-issued challenge through one or two authorized collectors."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import hmac
import json
import math
from time import sleep
from time import perf_counter, thread_time

from cryptography.exceptions import InvalidSignature
from .clock import now
from . import randomized_check as proof
from .tcp_transport import exact_fields, integer, remaining, require, rpc


def token(key, session, round_id, worker, gateway, command):
    body = json.dumps([session,round_id,worker,gateway,command],sort_keys=True,separators=(',',':')).encode()
    return hmac.new(key,b'OCP/proof-relay/1\x00'+body,sha256).hexdigest()


def forward(edge, header, payload, server):
    body=header['body']
    require(header['sender']=='coordinator' and not payload,'Relay needs coordinator authorization')
    exact_fields(body,{'worker','command','token','emulated_ms'} |
                 ({'workload'} if 'workload' in body else set()))
    i=integer(body['worker'],0,len(edge.workers)-1)
    delay=body['emulated_ms']
    require(type(delay) in (int,float) and math.isfinite(delay) and -2<=delay<=60000,'Invalid link delay')
    deadline=header['deadline_ns']/1e9
    work=body.get('workload',dict(cpu_iterations=0, service_ms=0.))
    exact_fields(work,{'cpu_iterations','service_ms'} |
                 ({'unavailable'} if 'unavailable' in work else set()))
    require(type(work.get('unavailable',False)) is bool,'Invalid collector availability control')
    require(not work.get('unavailable',False),'Controlled proof service unavailable')
    iterations=integer(work['cpu_iterations'],0,100000)
    service=work['service_ms']
    require(type(service) in (int,float) and math.isfinite(service) and 0<=service<=60000,
            'Invalid controlled service delay')
    start,cpu=perf_counter(),thread_time()
    if service:
        sleep(min(remaining(deadline),service/1000))
    block=bytes(65536) if iterations else b''
    for step in range(iterations):
        if step%16==0:remaining(deadline)
        sha256(block).digest()
    work_wall,work_cpu=perf_counter()-start,thread_time()-cpu
    if delay<0:
        sleep(remaining(deadline))
        raise TimeoutError('No measured response')
    if delay:
        sleep(min(remaining(deadline),delay/1000))
    remaining(deadline)
    reply,data=rpc(edge.workers[i],server.identity,server.keys,server.session,header['round'],
               'relayed_projection',{'command':body['command'],'token':body['token']},
               b'',deadline,server.meter,shaper=server.shaper)
    if 'workload' in body:
        reply=dict(reply, controlled_work=dict(cpu_iterations=iterations,service_ms=service,
                   wall_s=work_wall,cpu_s=work_cpu))
    return reply,data


def collect(cluster, roster, gateways, command, challenge, phase_end, emulated_ms, metrics,
            *,cpu_iterations=(0,0,0,0),service_ms=(0.,0.,0.,0.),unavailable=(),
            phase_started=None, observation_horizon_ms=None):
    require(len(cpu_iterations)==len(service_ms)==4,'Four collector workloads required')
    calls=[(g,i) for g in gateways for i in roster]
    started=now() if phase_started is None else phase_started
    metrics['collector_phase_budget_ms']=(max(0.,1000*(phase_end-started))
        if observation_horizon_ms is None else observation_horizon_ms)
    metrics['collector_attempts']=[list(pair) for pair in calls]
    metrics['collector_workload_observed']=[]
    def one(g,i):
        auth=token(cluster.keys['coordinator'][f'w{i}'],cluster.session,cluster.round_id,i,g,command)
        msg={'worker':i,'command':command,'token':auth,
             'emulated_ms':0. if emulated_ms is None else min(60000.,float(emulated_ms[i,g]))}
        if cpu_iterations[g] or service_ms[g] or g in unavailable:
            msg['workload']=dict(cpu_iterations=cpu_iterations[g],service_ms=service_ms[g],
                                 unavailable=g in unavailable)
        reply=cluster.call(f'e{g}',cluster.round_id,'relay_projection',msg,deadline=phase_end)
        return reply,now()
    replies={}
    receipts=[]
    errors=[]
    timing={str(g):{} for g in gateways}
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures={pool.submit(one,g,i):(g,i) for g,i in calls}
        for future in as_completed(futures):
            g,i=futures[future]
            try:
                (body,raw),arrived=future.result()
                if 'controlled_work' in body:
                    metrics['collector_workload_observed'].append(dict(gateway=g,worker=i,
                        **body['controlled_work']))
                    body={key:value for key,value in body.items() if key!='controlled_work'}
                require(arrived<=phase_end,'Reply after proof cut-off')
                exact_fields(body,{'tags','prepare_s','prepare_cpu_s'})
                encoded=proof.encoded_message(challenge,i,body['tags'])
                cluster.public_keys[i].verify(raw,encoded)
            except (OSError,ValueError,InvalidSignature) as e:
                errors.append(f'e{g}/w{i}: {type(e).__name__}')
                continue
            elapsed=arrived-started
            receipts.append(dict(worker=i,gateway=g,tags=body['tags'],signature=raw.hex(),
                session=challenge.session.hex(),epoch=challenge.epoch,round=challenge.round_id,
                candidate_hash=challenge.candidate_hash.hex(),seeds=[s.hex() for s in challenge.seeds],
                dimension=challenge.dimension,arrived_at=arrived,elapsed_ms=1000*elapsed))
            name=f'w{i}'
            if name in replies:
                old,_=replies[name]
                require(old[1]==raw and old[0]['tags']==body['tags'],'Conflicting signed response')
            replies[name]=((body,raw),elapsed)
            timing[str(g)][str(i)]=1000*elapsed
    metrics['collector_ms']=timing
    metrics['collector_receipts']=receipts
    return replies,errors
