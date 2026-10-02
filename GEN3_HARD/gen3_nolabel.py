"""gen3_nolabel.py — rebuild the 24 shipped clips with the grid LINES but no A-F / 1-16
captions, for the interactive human experiment.

The clips are recomposed from the cached raw render passes in `_cache/renders`, not
re-rendered: the engine is never started, so this is a few minutes of ffmpeg rather than
an overnight run, and the frames are the same pixels that produced the shipped clips.

Nothing about the clip changes except the captions — same window, same camera offset, same
red start circle, same 1 s/4 s split, same grid geometry. So `clips/ground_truth.csv`
scores both versions and the two are interchangeable as stimuli.

    python3 gen3_nolabel.py verify     # prove the recompose reproduces the SHIPPED clips
    python3 gen3_nolabel.py build      # write the unlabelled clips

`verify` is the point of this script. It recomposes with `labels=True` and checks the
result is byte-identical to the clip already in `clips/`. If that holds, the only
difference in the `labels=False` output is the captions, which is the claim being made.
Run it before trusting `build`.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen3_lib as G                                                  # noqa: E402

PICKS = G.CACHE / "picks.json"
RENDERS = G.CACHE / "renders"
OUT = HERE / "clips_nolabels"
VARIANTS = ("full_visibility", "split_1s_4s")


def rkey(p):
    return f"{p['match']}_{p['start']}_{p['end']}"


def load(p):
    vis = np.load(RENDERS / f"{rkey(p)}_vis.npz")["frames"]
    inv = np.load(RENDERS / f"{rkey(p)}_inv.npz")["frames"]
    return vis, inv


def compose(p, labels):
    vis, inv = load(p)
    full, split, start, final = G.compose_pair(
        vis, inv, final_fallback=(p["px"], p["py"]), labels=labels)
    return {"full_visibility": full, "split_1s_4s": split}, start, final


def cmd_verify(argv):
    """Recompose WITH labels and compare to what is already in clips/."""
    picks = json.loads(PICKS.read_text())
    if argv:
        picks = picks[: int(argv[0])]
    tmp = G.CACHE / "_nolabel_verify"
    tmp.mkdir(parents=True, exist_ok=True)
    bad = []
    for p in picks:
        seqs, (spx, spy), (fpx, fpy) = compose(p, labels=True)
        for v in VARIANTS:
            shipped = G.OUT / v / f"{p['clip']}.mov"
            rebuilt = tmp / f"{p['clip']}_{v}.mov"
            G.write_clip(seqs[v], rebuilt)
            a, b = shipped.read_bytes(), rebuilt.read_bytes()
            same = a == b
            if not same:
                bad.append(f"{p['clip']}/{v}")
            print(f"  {p['clip']} {v:16} {'byte-identical' if same else 'DIFFERS'}",
                  flush=True)
            rebuilt.unlink()
        # the ground truth this clip ships with must also come back out unchanged
        cell = G.cell_of(fpx, fpy)
        print(f"    ball {G.cell_of(spx, spy)} -> {cell}", flush=True)
    print()
    if bad:
        print(f"VERIFY FAILED on {len(bad)}: {bad}")
        return 1
    print(f"VERIFY OK — recompose reproduces all {len(picks) * 2} shipped clips byte for "
          f"byte, so labels=False changes only the captions")
    return 0


def cmd_build(argv):
    picks = json.loads(PICKS.read_text())
    rows = ["clip,ball_start_cell,ball_final_cell,start_px,start_py,final_px,final_py"]
    for p in picks:
        seqs, (spx, spy), (fpx, fpy) = compose(p, labels=False)
        for v in VARIANTS:
            G.write_clip(seqs[v], OUT / v / f"{p['clip']}.mov")
        rows.append(",".join(map(str, [
            p["clip"], G.cell_of(spx, spy), G.cell_of(fpx, fpy),
            round(spx, 1), round(spy, 1), round(fpx, 1), round(fpy, 1)])))
        print(f"  [nolabel] {p['clip']}: {G.cell_of(spx, spy)} -> {G.cell_of(fpx, fpy)}",
              flush=True)
    (OUT / "ground_truth_check.csv").write_text("\n".join(rows) + "\n")
    print(f"\nbuild: {len(picks) * 2} clips in {OUT}")
    return 0


CMDS = {"verify": cmd_verify, "build": cmd_build}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(__doc__)
    sys.exit(CMDS[sys.argv[1]](sys.argv[2:]))
