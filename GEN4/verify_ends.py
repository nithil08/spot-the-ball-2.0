"""verify_ends.py — re-measure the FIRST and LAST frame of every shipped GEN4 clip at
full resolution, and keep the evidence.

The probe that selection runs on is exact in method but coarse in resolution: down=2 with
a 6-pixel body floor. That is enough to choose windows and it applies the same body/shadow
rule, but the number that ships should not rest on a downsample. This re-renders the two
frames that matter at full size, keeps a pixel mask per player, and writes the count the
ground truth publishes.

Both frames, not just the level: GEN4's level is the OPENING count, and the closing count
is published alongside it, so both have to be measured the same way.

    python3 verify_ends.py [shard_i shard_n]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen4_lib as G                                                  # noqa: E402

PICKS = G.CACHE / "picks.json"
OUT = G.CACHE / "endframes"

DIFF = 30          # per-pixel sum-of-channel difference that counts as "changed"
MIN_PIX = G.MIN_BODY   # full-resolution body pixels before a person is in shot


def _render_ends(lvl, seed, start, end, hide, offset):
    """One pass, keeping the window's FIRST and LAST frame at full resolution."""
    frames, _ = G.render_window(lvl, seed, 0, end, keep={start, end - 1},
                                hide_slots=hide, offset=offset)
    return np.asarray(frames, dtype=np.int16)          # (2, H, W, 3)


def measure(p):
    off = (p["offset_x"], p["offset_y"])
    lvl = G.SCEN_PREFIX + p["shape"]
    s, e = p["start"], p["end"]

    full = _render_ends(lvl, p["seed"], s, e, "", off)                # everyone visible
    plate = _render_ends(lvl, p["seed"], s, e, ",".join(G.ALL_SLOTS), off)

    rec = {"clip": p["clip"], "kind": p["kind"], "match": p["match"], "window": [s, e],
           "offset": list(off), "first": [], "last": []}
    for slot in G.ALL_SLOTS:
        hide = ",".join(x for x in G.ALL_SLOTS if x != slot)
        solo = _render_ends(lvl, p["seed"], s, e, hide, off)
        for which, i in (("first", 0), ("last", 1)):
            d = np.abs(solo[i] - plate[i]).sum(axis=2)
            m = d > DIFF
            n = int(m.sum())
            if n == 0:
                rec[which].append({"slot": slot, "pix": 0})
                continue
            ys, xs = np.nonzero(m)
            sp = solo[i][m].astype(np.int16)
            pp = plate[i][m].astype(np.int16)
            # A body replaces the grass; a shadow only darkens it.
            darker = (sp.sum(axis=1) < pp.sum(axis=1)) & (np.abs(sp - pp).max(axis=1) < 60)
            rec[which].append({
                "slot": slot, "pix": n, "shadow_pix": int(darker.sum()),
                "body_pix": int((~darker).sum()),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "cx": float(xs.mean()), "cy": float(ys.mean())})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{p['clip']}.npz", first=full[0].astype(np.uint8),
                        last=full[1].astype(np.uint8), rec=json.dumps(rec))
    cnt = {w: sum(1 for x in rec[w] if x["pix"] >= MIN_PIX
                  and x.get("body_pix", 0) >= MIN_PIX) for w in ("first", "last")}
    print(f"  [ends] {p['clip']} {p['match']}: opens with {cnt['first']} "
          f"(target {p['count']}), ends with {cnt['last']}", flush=True)


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
