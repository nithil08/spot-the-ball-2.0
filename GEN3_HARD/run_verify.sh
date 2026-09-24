#!/bin/bash
# run_verify.sh [n_shards] — drive fill_gaps.py verify to completion.
# One window per process: the verification is 23 FootballEnvs and the engine dies at ~46.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
N="${1:-5}"
mkdir -p "$HERE/_cache/logs"
for ((i=0;i<N;i++)); do
  (
    while true; do
      nice -n 10 python3 "$HERE/fill_gaps.py" verify "$i" "$N" \
        >> "$HERE/_cache/logs/fillverify_$i.log" 2>&1
      [ $? -eq 3 ] && break
    done
  ) &
done
wait
echo "== verify complete =="
