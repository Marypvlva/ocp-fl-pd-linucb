"""Read published mesh ping results and keep first-packet RTT. Does not schedule measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np

MEASUREMENTS = (7714918, 11688943, 15393461, 12427662)  # Finland, Brazil, USA, Uganda
DATES = ('2025-04-01', '2025-05-06', '2025-06-03', '2025-07-01', '2025-08-05')
SLOT = 360
T = 120
API = 'https://atlas.ripe.net/api/v2/measurements'


def utc(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def get(url):
    for attempt in range(4):
        try:
            req = Request(url, headers={'Accept-Encoding':'gzip','User-Agent':'Research archive replay/1.0'})
            with urlopen(req, timeout=100) as response:
                raw = response.read()
                if response.headers.get('Content-Encoding')=='gzip':
                    raw = gzip.decompress(raw)
            return json.loads(raw)
        except (OSError, ValueError):
            if attempt==3:
                raise
            time.sleep(2**attempt)


def select_population():
    sets=[]
    for mid in MEASUREMENTS:
        start=utc(DATES[0])
        rows=get(f'{API}/{mid}/results/?start={start}&stop={start+1200}')
        sets.append({r['prb_id'] for r in rows if r.get('prb_id')})
    common=set.intersection(*sets)
    assert len(common)>=160
    return sorted(common,key=lambda i:sha256(f'roster-study-v3/{i}'.encode()).digest())[:160]


def fetch(task):
    day, mid, ids, out = task
    start=utc(day)
    # -2 = missing sample, -1 = explicit loss, >0 = packet returned.
    rtt=np.full((T,len(ids)),-2.,dtype=np.float32)
    first=np.full((T,len(ids)),2**40,dtype=np.int64)
    index={i:j for j,i in enumerate(ids)}
    for off in range(0,len(ids),32):
        selected=ids[off:off+32]
        text=','.join(map(str,selected))
        url=f'{API}/{mid}/results/?start={start}&stop={start+T*SLOT-1}&probe_ids={text}'
        for row in get(url):
            pid=row.get('prb_id')
            if pid not in index or row.get('msm_id')!=mid:
                continue
            ts=row['timestamp']
            t=(ts-start)//SLOT
            if not (0<=t<T) or ts>=first[t,index[pid]]:
                continue
            packets=row.get('result',[])
            if not packets:
                continue
            first[t,index[pid]]=ts
            val=packets[0].get('rtt',-1)
            rtt[t,index[pid]]=float(val) if isinstance(val,(float,int)) and val>0 else -1.
    file=out/f'{day}_{mid}.npz'
    np.savez_compressed(file,rtt=rtt,ids=np.array(ids,dtype=np.int64))
    print(day,mid,'observed',round(float((rtt!=-2).mean()),4),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    ap.add_argument('--part',choices=('dev','test','all'),default='all')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.out/'provenance.json'
    if manifest.exists():
        ids=json.loads(manifest.read_text())['probe_ids']
    else:
        ids=select_population()
        manifest.write_text(json.dumps(dict(measurements=MEASUREMENTS,probe_ids=ids,dates=DATES,
               seconds_per_slot=SLOT,slots_per_day=T,first_32_development=True,
               antileakage='Random order from ID hash, inclusion based solely on April sample presence. Independent test calendar and held-out probe IDs.',
               sources=[f'{API}/{m}/' for m in MEASUREMENTS]),indent=2))
    dates={'dev':DATES[:1],'test':DATES[1:],'all':DATES}[args.part]
    tasks=[(day,mid,ids[:32] if day==DATES[0] else ids[32:],args.out) for day in dates for mid in MEASUREMENTS
           if not (args.out/f'{day}_{mid}.npz').exists()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,tasks))


if __name__=='__main__':
    main()
