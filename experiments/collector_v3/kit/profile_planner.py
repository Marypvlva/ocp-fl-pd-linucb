"""Measure single-controller CPU/wall elapsed time and resident NumPy buffers, excluding network and EVM."""
import argparse,json,platform,time
from pathlib import Path
import numpy as np
from model import Policy,Settings
from lookahead import ForecastPrices
from experiment import observed,panels
from gateway_study import caps


def main():
    ap=argparse.ArgumentParser();ap.add_argument('data',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    x=panels(a.data/'atlas','test')[0];h,n,_=x.shape
    base=Settings(**json.loads((a.data/'gateway_calibration.json').read_text())['records']['joint']['chosen']['settings'])
    tuned=Settings(**json.loads((a.data/'lookahead_dev.json').read_text())['chosen']['config'])
    rs=np.random.default_rng(4486);seats=[rs.choice(n,3,replace=False)[None,:] for _ in range(h)]; windows=np.resize(np.arange(3),h)
    records={}
    for name,klass,setting in [('joint',Policy,base),('low_alpha',Policy,tuned),('forecast',ForecastPrices,tuned)]:
        times=[];cpu=[];final_array=0
        for rep in range(4):
            policy=klass(setting,1,n,3,h,120,53,caps(0,h)[None,:]);batch=[];cb=[]
            for t,(s,c) in enumerate(zip(seats,windows)):
                start=time.perf_counter_ns();ct=time.process_time_ns();arm=policy.decide(c,s);ct1=time.process_time_ns();end=time.perf_counter_ns()
                obs,y=observed(x[None,t],arm,s,c)
                start2=time.perf_counter_ns();ct2=time.process_time_ns();policy.observe(obs,y);ct3=time.process_time_ns();end2=time.perf_counter_ns()
                batch.append((end-start+end2-start2)/1e6);cb.append((ct1-ct+ct3-ct2)/1e6)
            if rep>0:times.extend(batch);cpu.extend(cb)
        def buffers(obj,seen):
            if id(obj) in seen:return 0
            seen.add(id(obj))
            if isinstance(obj,np.ndarray):return obj.nbytes
            if isinstance(obj,dict):return sum(buffers(v,seen) for v in obj.values())
            if isinstance(obj,(tuple,list)):return sum(buffers(v,seen) for v in obj)
            return 0
        final_array=buffers(vars(policy),set())
        records[name]=dict(wall_ms_med_p95=np.quantile(times,[.5,.95]).tolist(),cpu_ms_med_p95=np.quantile(cpu,[.5,.95]).tolist(),resident_numpy_logical_bytes=final_array,samples=len(times))
    a.output.write_text(json.dumps(dict(platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),numpy=np.__version__,methods=records),indent=2)+'\n')

if __name__=='__main__':main()
