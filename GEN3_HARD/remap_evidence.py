"""remap_evidence.py — carry each kept clip's final-frame measurement to its new number.

The measurement is of a WINDOW, and a kept clip's window did not change — only the number
in front of it did. Re-rendering 23 passes to learn what is already on disk would be
waste; this renames it, keyed by the window, and leaves the new clips to be measured.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = HERE / "_cache" / "finalframe"
OLD = json.loads((HERE / "_cache" / "picks_before_repair.json").read_text())
NEW = json.loads((HERE / "_cache" / "picks.json").read_text())

by_window = {f"{p['match']}_{p['start']}_{p['end']}": p["clip"] for p in OLD}
staging = FINAL.parent / "finalframe_old"
if not staging.exists():
    FINAL.rename(staging)
FINAL.mkdir(exist_ok=True)
moved = missing = 0
for p in NEW:
    key = f"{p['match']}_{p['start']}_{p['end']}"
    src = staging / f"{by_window.get(key, '')}.npz"
    if by_window.get(key) and src.exists():
        (FINAL / f"{p['clip']}.npz").write_bytes(src.read_bytes())
        moved += 1
    else:
        missing += 1
print(f"remapped {moved} kept measurements; {missing} clips still to measure")
