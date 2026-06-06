#!/bin/bash
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

LIB_EXTENSION="so"

if [[ "$OSTYPE" == "darwin"* ]] ; then
    LIB_EXTENSION="dylib"
fi

# Take into account # of cores and available RAM for deciding on compilation parallelism.
# psutil may be missing during isolated `pip wheel` builds — fall back to cpu count.
_PY_FOR_COUNT="${GFOOTBALL_PYTHON:-python3}"
PARALLELISM=$("$_PY_FOR_COUNT" <<'PY'
import multiprocessing as mp
try:
    import psutil
    avail_gb = psutil.virtual_memory().available / 1e9
    print(int(max(1, min((avail_gb - 1) / 0.5, mp.cpu_count()))))
except ImportError:
    print(max(1, mp.cpu_count()))
PY
)

# Drop stale CMake state (otherwise FindPython can keep Homebrew 3.14 from an old run).
rm -f third_party/gfootball_engine/CMakeCache.txt
rm -rf third_party/gfootball_engine/CMakeFiles

# Prefer the interpreter used by `pip install` (set from setup.py). Otherwise CMake
# may pick another python3 on PATH (e.g. Homebrew 3.14) and the wrong Boost.Python.
# (POSIX-friendly: avoid bash arrays so /bin/sh compatibility is not required.)
# Subshell so a failed cmake does not leave cwd inside third_party (which broke the next step).
(
  cd third_party/gfootball_engine || exit 1
  if [ -n "${GFOOTBALL_PYTHON:-}" ]; then
    _PYROOT="$(dirname "$(dirname "${GFOOTBALL_PYTHON}")")"
    cmake . \
      -DPython_EXECUTABLE="${GFOOTBALL_PYTHON}" \
      -DPython_ROOT_DIR="${_PYROOT}"
  else
    cmake .
  fi
  make -j "$PARALLELISM"
)
(
  cd third_party/gfootball_engine && ln -sf "libgame.$LIB_EXTENSION" _gameplayfootball.so
)
