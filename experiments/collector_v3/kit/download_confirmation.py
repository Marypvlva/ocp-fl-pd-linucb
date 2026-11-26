"""Second, disjoint April ID block and untouched September–December 2025 windows."""
from concurrent.futures import ThreadPoolExecutor
import argparse,json
from hashlib import sha256
from pathlib import Path
from download_atlas import MEASUREMENTS, get, utc, fetch, API
DATES2=['2025-09-02','2025-10-07','2025-11-04','2025-12-02']

def main():
    a=argparse.ArgumentParser();a.add_argument('destination',type=Path);args=a.parse_args()
    args.destination.mkdir(parents=True,exist_ok=True)
    f=args.destination/'provenance.json'
    if not f.exists():
        vals=[];start=utc('2025-04-01')
        for mid in MEASUREMENTS:
            rows=get(f'{API}/{mid}/results/?start={start}&stop={start+1200}')
            vals.append({r['prb_id'] for r in rows if r.get('prb_id')})
        ordered=sorted(set.intersection(*vals),key=lambda i:sha256(f'roster-study-v3/{i}'.encode()).digest())
        # Same catalog as primary work, following 128 positions.
        ids=ordered[160:288]
        f.write_text(json.dumps(dict(measurements=MEASUREMENTS,probe_ids=[*ordered[:32],*ids],
                                     dates=['2025-04-01',*DATES2],excluded_positions='32:160 primary',
                                     used_positions='160:288',seconds_per_slot=360,slots_per_day=120),indent=2))
    ids=json.loads(f.read_text())['probe_ids'][32:]
    tasks=[(day,m,ids,args.destination) for day in DATES2 for m in MEASUREMENTS if not (args.destination/f'{day}_{m}.npz').exists()]
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch,tasks))
if __name__=='__main__':main()
