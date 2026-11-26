"""Recompute all WAN replay counts from the supplied matrices in a fresh directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('project',type=Path);p.add_argument('source',type=Path);p.add_argument('target',type=Path);a=p.parse_args()
    project,source,target=a.project.resolve(),a.source.resolve(),a.target.resolve();kit=Path(__file__).resolve().parent
    if target.exists(): raise SystemExit('Choose a fresh replay directory')
    target.mkdir(parents=True)
    for sub in ('atlas','atlas_confirmation','atlas_feedback'): shutil.copytree(source/sub,target/sub)
    def call(script,*params):
        subprocess.run([sys.executable,str(kit/(script+'.py')),str(project),*map(str,params)],check=True)
    call('development',target/'atlas',target/'dev_scores.json')
    call('gateway_study',target,'calibrate');call('confirmation',target,'tune');call('feedback',target,'tune')
    for name,k,r in [('main',3,24),('global',3,24),('tight',3,12),('loose',3,12),('long_delay',3,12),('no_mirror',3,12),('same_links',3,12),('group2',2,12),('group4',4,12)]:
        call('gateway_study',target,'test','--variant',name,'--k',k,'--repeats',r)
    for phase in ('first','confirm'):call('confirmation',target,phase)
    call('feedback',target,'test');call('sensitivity',target);call('option_bounds',target);call('latency_budget',target)
    for path in sorted(source.glob('*.csv')):
        if (target/path.name).read_bytes()!=path.read_bytes():raise RuntimeError(f'Different replay values: {path.name}')
    subprocess.run([sys.executable,str(kit/'analyze_v3.py'),str(target),str(target/'computed.json'),'--simulate-only'],check=True)
    print('All published CSV files match. Panel analyses are in computed.json.')

if __name__=='__main__':main()
