"""Third disjoint ID block and unseen January–April 2026 windows."""
from concurrent.futures import ThreadPoolExecutor
import argparse,json
from hashlib import sha256
from pathlib import Path
from download_atlas import MEASUREMENTS, get, utc, fetch, API
DATES2=['2026-01-06','2026-02-03','2026-03-03','2026-04-07']

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
        ids=ordered[288:416]
        assert len(ids)==128
        f.write_text(json.dumps(dict(measurements=MEASUREMENTS,probe_ids=[*ordered[:32],*ids],
                                     dates=['2025-04-01',*DATES2],excluded_positions='32:288 A and B',
                                     used_positions='288:416',seconds_per_slot=360,slots_per_day=120),indent=2))
    ids=json.loads(f.read_text())['probe_ids'][32:]
    tasks=[(day,m,ids,args.destination) for day in DATES2 for m in MEASUREMENTS if not (args.destination/f'{day}_{m}.npz').exists()]
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch,tasks))
if __name__=='__main__':main()
