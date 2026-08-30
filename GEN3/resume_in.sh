#!/bin/bash
# resume_in.sh <seconds> [shards] [nice] — wait, then resume the probe.
# Detached and nohup'd so it survives this terminal. The phase is resumable, so a
# restart re-probes nothing that already has a counts/*.npz.
DELAY="${1:-3600}"; SH="${2:-3}"; NI="${3:-15}"
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "$(date +%H:%M:%S) sleeping ${DELAY}s, then probe on $SH shards"
sleep "$DELAY"
echo "$(date +%H:%M:%S) resuming: $(ls "$HERE/_cache/counts" | wc -l)/77 already done"
bash "$HERE/run_phase.sh" probe "$SH" "$NI"
echo "$(date +%H:%M:%S) probe finished: $(ls "$HERE/_cache/counts" | wc -l)/77"
