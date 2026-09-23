#!/bin/bash
# run_finalframe.sh [n_shards] — drive verify_final_frame.py to completion.
# The engine dies at roughly its 46th FootballEnv and one clip costs 23 of them, so a
# shard is expected to die every couple of clips. Restart it until nothing is left to do.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
N="${1:-3}"
mkdir -p "$HERE/_cache/logs"
for ((i=0;i<N;i++)); do
  (
    while true; do
      left=$(python3 - "$HERE" "$i" "$N" <<'PY'
import json,sys
from pathlib import Path
h,i,n=Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
picks=json.loads((h/'_cache/picks.json').read_text())
print(sum(1 for k,p in enumerate(picks)
          if k%n==i and not (h/'_cache/finalframe'/f"{p['clip']}.npz").exists()))
PY
)
      [ "$left" = "0" ] && break
      nice -n 10 python3 "$HERE/verify_final_frame.py" "$i" "$N" >> "$HERE/_cache/logs/finalframe_$i.log" 2>&1
    done
    echo "shard $i done" >> "$HERE/_cache/logs/finalframe_$i.log"
  ) &
done
wait
