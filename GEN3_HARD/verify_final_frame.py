"""verify_final_frame.py — re-measure the FINAL FRAME of every shipped GEN3_HARD clip,
at full resolution, and keep the evidence.

Why this exists: the shipped counts disagreed with each other. `picks.json["count"]` (what
clip_classification.csv published) came from the phase-4 `probe`, which measured candidate
end frames at the offset planned for the MATCH; `_cache/trace` re-measured the windows that
actually shipped. Where the picked window or offset moved after the probe, the two differ,
and nothing in the pipeline re-checked which was right.

This is the same solo-probe mechanism — plate with all 22 hidden, then 22 passes each
showing exactly one player — but:

  * full resolution (down=1), not half, so a distant body is not thresholded away;
  * only the final frame is kept, so the memory cost is trivial;
  * the per-player DIFF MASK is kept, not just a boolean, which gives a bounding box and
    a pixel count per player and lets the result be drawn on the shipped frame and checked
    by eye;
  * body pixels are separated from SHADOW-ONLY pixels. A player can stand outside the
    camera frustum and still cast a shadow into the shot: the earlier probe counted that
    as a person in frame, which is wrong for a "how many people can you see" label.

Writes _cache/finalframe/<clip>.npz per clip: the full-visibility frame, the plate, and
per-slot masks compressed to bbox + pixel counts.

    python3 verify_final_frame.py [shard_i shard_n]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen3_lib as G                                                  # noqa: E402

PICKS = G.CACHE / "picks.json"
OUT = G.CACHE / "finalframe"

DIFF = 30          # per-pixel sum-of-channel difference that counts as "changed"
MIN_PIX = 20       # full-resolution changed pixels before a slot counts at all


def _render_final(lvl, seed, end, hide, offset):
    """One pass, keeping only frame `end - 1` at full resolution."""
    frames, _ = G.render_window(lvl, seed, 0, end, keep={end - 1}, hide_slots=hide,
                                offset=offset)
    return np.asarray(frames[-1], dtype=np.int16)


def measure(p):
    off = (p["offset_x"], p["offset_y"])
    lvl = f"g3h_{p['shape']}"
    end = p["end"]

    full = _render_final(lvl, p["seed"], end, "", off)                # everyone visible
    plate = _render_final(lvl, p["seed"], end, ",".join(G.ALL_SLOTS), off)

    rec = {"clip": p["clip"], "kind": p["kind"], "match": p["match"], "end": end,
           "offset": list(off), "slots": []}
    for slot in G.ALL_SLOTS:
        hide = ",".join(s for s in G.ALL_SLOTS if s != slot)
        solo = _render_final(lvl, p["seed"], end, hide, off)
        d = np.abs(solo - plate).sum(axis=2)
        m = d > DIFF
        n = int(m.sum())
        if n == 0:
            rec["slots"].append({"slot": slot, "pix": 0})
            continue
        ys, xs = np.nonzero(m)
        # A body replaces the grass; a shadow only darkens it. Split the changed pixels on
        # that: brighter-or-recoloured against strictly-darker-and-desaturated.
        sp = solo[m].astype(np.int16)
        pp = plate[m].astype(np.int16)
        darker = (sp.sum(axis=1) < pp.sum(axis=1)) & (np.abs(sp - pp).max(axis=1) < 60)
        rec["slots"].append({
            "slot": slot, "pix": n, "shadow_pix": int(darker.sum()),
            "body_pix": int((~darker).sum()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "cx": float(xs.mean()), "cy": float(ys.mean()),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{p['clip']}.npz", full=full.astype(np.uint8),
                        plate=plate.astype(np.uint8), rec=json.dumps(rec))
    body = sum(1 for s in rec["slots"] if s["pix"] >= MIN_PIX and s.get("body_pix", 0) >= MIN_PIX)
    any_ = sum(1 for s in rec["slots"] if s["pix"] >= MIN_PIX)
    print(f"  [final] {p['clip']} {p['match']}: bodies {body}, any-trace {any_}", flush=True)


def main(argv):
    from lib import use_bundle
    from scenario_factory import write_scenario
    shard_i, shard_n = (int(argv[0]), int(argv[1])) if len(argv) >= 2 else (0, 1)
    use_bundle(G.BUNDLE_VIS)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    picks = json.loads(PICKS.read_text())
    todo = [p for i, p in enumerate(picks)
            if i % shard_n == shard_i and not (OUT / f"{p['clip']}.npz").exists()]
    print(f"shard {shard_i}/{shard_n}: {len(todo)} clips", flush=True)
    for p in todo:
        measure(p)


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", os.environ.get("SDL_VIDEODRIVER", ""))
    main(sys.argv[1:])
