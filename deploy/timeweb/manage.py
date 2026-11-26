"""Run on the coordinator through the Timeweb browser console."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
STATE=ROOT/'deploy/timeweb/state'


def execute(args,**kwargs):
    return subprocess.run(list(map(str,args)),check=True,**kwargs)


def load():
    return json.loads((STATE/'inventory.json').read_text())


def ssh(host,command,**kwargs):
    return execute(['ssh','-i',STATE/'coordinator_key','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',f'root@{host}',command],**kwargs)


def initialize(a):
    for host in [a.coordinator,*a.workers,*a.gateways]:ipaddress.ip_address(host)
    STATE.mkdir(parents=True,exist_ok=True);STATE.chmod(0o700)
    if (STATE/'inventory.json').exists():raise ValueError('Inventory exists. Preserve it and use it for this stand.')
    inv=dict(coordinator=a.coordinator,workers=a.workers,edges=a.gateways)
    (STATE/'inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
    execute(['ssh-keygen','-q','-t','ed25519','-N','','-f',STATE/'coordinator_key'])
    execute(['openssl','req','-x509','-newkey','rsa:3072','-nodes','-days','365','-subj','/CN=OCP research coordinator',
        '-addext',f'subjectAltName=IP:{a.coordinator},IP:127.0.0.1','-addext','basicConstraints=critical,CA:TRUE',
        '-keyout',STATE/'tls.key','-out',STATE/'tls.crt'])
    (STATE/'tls.key').chmod(0o600)
    placement=dict(workers=a.workers,edges=a.gateways,ssh=dict(ssh_user='root',
        ssh_options=['-i',str(STATE/'coordinator_key'),'-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes'],
        remote_dir=str(ROOT),remote_python='.venv/bin/python'))
    (STATE/'placement.json').write_text(json.dumps(placement,indent=2)+'\n')
    print('Add this PUBLIC key to authorized_keys on every worker/gateway using its browser console:')
    print((STATE/'coordinator_key.pub').read_text())
    execute(['openssl','x509','-in',STATE/'tls.crt','-noout','-fingerprint','-sha256'])


def deploy():
    inv=load();hosts=sorted(set(inv['workers']+inv['edges']))
    for host in hosts:
        # First host-key trust is an explicit interactive SSH step in RUNBOOK.
        ssh(host,'true')
        ssh(host,f'mkdir -p {shlex.quote(str(ROOT))}')
        execute(['rsync','-az','--exclude=.git','--exclude=.venv','--exclude=.bootstrap','--exclude=results',
            '--exclude=deploy/timeweb/state','--exclude=__pycache__',
            '-e',f'ssh -i {shlex.quote(str(STATE/"coordinator_key"))} -o BatchMode=yes -o StrictHostKeyChecking=yes',
            str(ROOT)+'/',f'root@{host}:{ROOT}/'])
        ssh(host,f'mkdir -p {ROOT}/deploy/timeweb/state && chmod 700 {ROOT}/deploy/timeweb/state')
        execute(['scp','-i',STATE/'coordinator_key','-o','StrictHostKeyChecking=yes',STATE/'tls.crt',f'root@{host}:{STATE}/tls.crt'])
        ssh(host,f'cd {ROOT} && bash deploy/timeweb/bootstrap.sh')
    print('Deployed',len(hosts),'peer VMs')


def diagnose():
    inv=load();hosts=sorted(set([inv['coordinator'],*inv['workers'],*inv['edges']]))
    # Coordinator SSH access to itself uses the same pinned stand key.
    listeners=[];report={}
    try:
        for host in hosts:
            check=ssh(host,'chronyc waitsync 10 0.005 && chronyc tracking',capture_output=True,text=True)
            cmd=f'cd {ROOT} && .venv/bin/python deploy/timeweb/diagnose.py listen'
            child=subprocess.Popen(['ssh','-i',str(STATE/'coordinator_key'),'-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',f'root@{host}',cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            listeners.append(child)
            if child.stdout.readline().strip()!='READY':raise RuntimeError(f'Probe did not start at {host}')
            report[host]=dict(clock=check.stdout)
        for host in hosts:
            code=f'cd {ROOT} && .venv/bin/python deploy/timeweb/diagnose.py probe --hosts '+ ' '.join(hosts)
            report[host]['rtt_ms']=json.loads(ssh(host,code,capture_output=True,text=True).stdout)
        (STATE/'diagnostics.json').write_text(json.dumps(dict(at=time.time(),hosts=report),indent=2)+'\n')
        print('Clock synchronization and all peer TCP directions passed')
    finally:
        for process in listeners:
            process.stdin.close()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    init=sub.add_parser('init');init.add_argument('--coordinator',required=True);init.add_argument('--workers',nargs='+',required=True);init.add_argument('--gateways',nargs=4,required=True)
    sub.add_parser('deploy');sub.add_parser('diagnose')
    plan=sub.add_parser('plan');plan.add_argument('--out',type=Path,required=True);plan.add_argument('--profile',choices=['pilot','confirmatory'],default='pilot');plan.add_argument('--network-mode',choices=['natural','controlled'],default='natural');plan.add_argument('--chain-mode',choices=['local_evm','external_rpc'],default='local_evm');plan.add_argument('--rounds',type=int);plan.add_argument('--seeds',type=int,nargs='+');plan.add_argument('--modes',nargs='+',choices=['joint','no_ids','scalar','recent','round_lin','lp_mean','greedy'])
    run=sub.add_parser('run');run.add_argument('--out',type=Path,required=True);run.add_argument('--chain-config',type=Path)
    a=p.parse_args()
    if a.action=='init':initialize(a)
    elif a.action=='deploy':deploy()
    elif a.action=='diagnose':diagnose()
    else:
        sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/collector_v3/kit'))
        if a.action=='plan':
            from distributed_live import plan
            plan(ROOT,STATE/'placement.json',a.out,a.profile,a.rounds,a.seeds,a.modes,a.network_mode,a.chain_mode)
        else:
            diag=json.loads((STATE/'diagnostics.json').read_text())
            if time.time()-diag['at']>3600:raise ValueError('Run diagnose again within one hour of the study')
            inv=load();config=json.loads((a.out/'plan.json').read_text())
            (a.out/'network-diagnostics.json').write_text(json.dumps(diag,indent=2))
            args=[ROOT/'.venv/bin/python',ROOT/'experiments/collector_v3/kit/live.py','--plan',a.out/'plan.json',
                '--control-url',f'https://{inv["coordinator"]}:9443','--tls-cert',STATE/'tls.crt','--tls-key',STATE/'tls.key','--tls-ca',STATE/'tls.crt','--network-mode',config['network_mode']]
            if a.chain_config:args += ['--chain-config',a.chain_config]
            execute(args,env=dict(os.environ,ADAPTIVE_CLOCK='wall',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))

if __name__=='__main__':main()
