#!/usr/bin/env bash
# Run from ~/adaptive_planning on the coordinator after bootstrap.sh.
# This is an acceptance/variance pilot. Create a separate confirmatory plan after it.
set -euo pipefail
OUT=${1:-study_cloud_v4_pilot}
PY=.venv/bin/python
P=deploy/yandex/placement.json
export ADAPTIVE_CLOCK=wall
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "$OUT"
for mode in refuse silent; do
  "$PY" -m adaptive_planning.stand_check --placement "$P" --workers 8 --outage "$mode" \
    --rpc-timeout 2.0 --output "$OUT/stand_check_$mode.json" > /dev/null
  echo "stand check ($mode): PASS"
done
cp deploy/yandex/clock_check.txt "$OUT/clock_check.txt"
if [ ! -f "$OUT/plan.json" ]; then
  "$PY" deploy/live_study.py plan --out "$OUT" --placement "$P" \
    --cells joint_tight joint_medium D7_backup_and_upload_drift \
    --policies AF_L filter_only --seed0 720000 --seeds 4 --horizon 48 \
    --deadline 3.0 --rpc-timeout 2.0 --edge-outage refuse
fi
# Sequential episodes avoid interference between policy pairs on shared VMs.
# The run can be resumed with this same command and immutable sources.
"$PY" deploy/live_study.py run --out "$OUT" --placement "$P" --parallel 1 --stop-on-error
"$PY" deploy/live_study.py summarize --out "$OUT"
"$PY" deploy/audit_study.py "$OUT"
