"""Deliver one coordinator-issued challenge through one or two authorized collectors."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import hmac
import json
import math
from time import sleep

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
    exact_fields(body,{'worker','command','token','emulated_ms'})
    i=integer(body['worker'],0,len(edge.workers)-1)
    delay=body['emulated_ms']
    require(type(delay) in (int,float) and math.isfinite(delay) and -2<=delay<=60000,'Invalid link delay')
    deadline=header['deadline_ns']/1e9
    if delay<0:
        sleep(remaining(deadline))
        raise TimeoutError('No measured response')
    if delay:
        sleep(min(remaining(deadline),delay/1000))
    remaining(deadline)
    return rpc(edge.workers[i],server.identity,server.keys,server.session,header['round'],
               'relayed_projection',{'command':body['command'],'token':body['token']},
               b'',deadline,server.meter,shaper=server.shaper)


def collect(cluster, roster, gateways, command, challenge, phase_end, emulated_ms, metrics):
    calls=[(g,i) for g in gateways for i in roster]
    started=now()
    def one(g,i):
        auth=token(cluster.keys['coordinator'][f'w{i}'],cluster.session,cluster.round_id,i,g,command)
        msg={'worker':i,'command':command,'token':auth,
             'emulated_ms':0. if emulated_ms is None else min(60000.,float(emulated_ms[i,g]))}
        reply=cluster.call(f'e{g}',cluster.round_id,'relay_projection',msg,deadline=phase_end)
        return reply,now()
    replies={}
    errors=[]
    timing={str(g):{} for g in gateways}
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures={pool.submit(one,g,i):(g,i) for g,i in calls}
        for future in as_completed(futures):
            g,i=futures[future]
            try:
                (body,raw),arrived=future.result()
                require(arrived<=phase_end,'Reply after proof cut-off')
                exact_fields(body,{'tags','prepare_s','prepare_cpu_s'})
                encoded=proof.encoded_message(challenge,i,body['tags'])
                cluster.public_keys[i].verify(raw,encoded)
            except (OSError,ValueError,InvalidSignature) as e:
                errors.append(f'e{g}/w{i}: {type(e).__name__}')
                continue
            elapsed=arrived-started
            name=f'w{i}'
            if name in replies:
                old,_=replies[name]
                require(old[1]==raw and old[0]['tags']==body['tags'],'Conflicting signed response')
            replies[name]=((body,raw),elapsed)
            timing[str(g)][str(i)]=1000*elapsed
    metrics['collector_ms']=timing
    return replies,errors
