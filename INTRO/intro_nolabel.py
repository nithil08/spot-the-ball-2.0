"""intro_nolabel.py — the two 20 s intro clips with the grid LINES but no A-F / 1-16
captions, to match the unlabelled graded clips in the interactive human experiment.

Recomposed from `_cache/raw`, so the engine is never started and the frames are the same
pixels that produced the shipped intro clips. Same window, same red start circle, same
1 s/19 s split, same grid geometry — only the captions are gone.

    python3 intro_nolabel.py verify    # prove the recompose reproduces the SHIPPED clips
    python3 intro_nolabel.py build     # write the unlabelled clips

`verify` recomposes with labels ON and checks the result is byte-identical to what is
already in `clips/`. Run it before trusting `build`.

Memory: each raw pass is 921 MB and a 500-frame composed list is another 921 MB, so this
holds ~2 GB at peak. The passes are read with mmap so only the composed list is resident.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent / "GEN3"
sys.path.insert(0, str(GEN3))

import gen3_lib as G                                                  # noqa: E402

CACHE = HERE / "_cache"
PICKS = CACHE / "picks.json"
RAW = CACHE / "raw"
VIS_FRAMES = G.VIS_FRAMES
VARIANTS = ("full_visibility", "split_1s_19s")


def compose_one(p, labels):
    """(start_px, start_py), {variant: [frames]} for one intro clip."""
    from grid import build_grid, burn_grid, circle_ball, detect_ball_or_none
    overlay = build_grid(labels=labels)
    vis = np.load(RAW / f"{p['clip']}_vis.npy", mmap_mode="r")
    inv = np.load(RAW / f"{p['clip']}_inv.npy", mmap_mode="r")

    start = next((b for b in (detect_ball_or_none(np.asarray(vis[i]), np.asarray(inv[i]))
                              for i in range(G.MARK_FRAMES)) if b), None)
    if start is None:
        sys.exit(f"{p['clip']}: no ball in the first {G.MARK_FRAMES} frames")
    spx, spy = start

    def stream(hide_after=None):
        for i in range(len(vis)):
            src = inv if (hide_after is not None and i >= hide_after) else vis
            g = burn_grid(np.asarray(src[i]), overlay)
            yield circle_ball(g, spx, spy) if i < G.MARK_FRAMES else g

    return (spx, spy), {"full_visibility": lambda: list(stream()),
                        "split_1s_19s": lambda: list(stream(hide_after=VIS_FRAMES))}


def write(frames, path):
    from lib import frames_to_mov
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_mov(frames, path, fps=G.FPS, crop_hud=False)


def cmd_verify(argv):
    tmp = CACHE / "_nolabel_verify"
    tmp.mkdir(parents=True, exist_ok=True)
    bad = []
    for p in json.loads(PICKS.read_text()):
        _, make = compose_one(p, labels=True)
        for v in VARIANTS:
            shipped = HERE / "clips" / v / f"{p['clip']}.mov"
            rebuilt = tmp / f"{p['clip']}_{v}.mov"
            frames = make[v]()
            write(frames, rebuilt)
            del frames
            same = shipped.read_bytes() == rebuilt.read_bytes()
            if not same:
                bad.append(f"{p['clip']}/{v}")
            print(f"  {p['clip']} {v:16} {'byte-identical' if same else 'DIFFERS'}",
                  flush=True)
            rebuilt.unlink()
    print()
    if bad:
        print(f"VERIFY FAILED on {len(bad)}: {bad}")
        return 1
    print("VERIFY OK — recompose reproduces the shipped intro clips byte for byte")
    return 0


def cmd_build(argv):
    out = HERE / "clips_nolabels"
    for p in json.loads(PICKS.read_text()):
        (spx, spy), make = compose_one(p, labels=False)
        for v in VARIANTS:
            frames = make[v]()
            write(frames, out / v / f"{p['clip']}.mov")
            del frames
        print(f"  [nolabel] {p['clip']}: start {G.cell_of(spx, spy)}", flush=True)
    print(f"\nbuild: clips in {out}")
    return 0


CMDS = {"verify": cmd_verify, "build": cmd_build}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(__doc__)
    sys.exit(CMDS[sys.argv[1]](sys.argv[2:]))
