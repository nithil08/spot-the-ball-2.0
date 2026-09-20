"""render_goal.py — cut a clip around a GOAL out of the cached GEN3 sweep.

Every shipped batch deliberately excludes goals: GEN3's `continuous()` rejects any
window where the ball crosses a goal line between the posts, because a goal respots
the ball at the centre spot and breaks the one-unbroken-passage-of-play rule the
spot-the-ball task needs. So there is no goal clip to find — but the sweep that the
batches were picked FROM has plenty, and the matches are deterministic, so a goal can
simply be re-cut from `GEN3/_cache/sweep/<shape>_s<seed>.npz`.

find_goals() reports every score change in the cached sweeps (no engine run).
render() replays that match and keeps the window around one of them.

Run:  python3 render_goal.py list                       # every goal in the sweep
      python3 render_goal.py render mid_push 319 146    # clip that goal
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "GEN3"))

import gen3_lib as G                                                  # noqa: E402

OUT = HERE / "clips"
LEAD_IN = 100          # frames of build-up kept before the ball crosses the line (4 s)
LEAD_OUT = 25          # frames kept after it (1 s), so the net bulges on camera


def find_goals():
    """(shape, seed, frame, score_before, score_after, ball_on_the_line) per goal.

    `frame` is the sweep index at which the score changes; the ball is still on the
    goal line at frame - 1 and is respotted at the centre circle from `frame` on.
    """
    out = []
    for f in sorted(G.SWEEP.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        score = d["score"]
        for i in np.flatnonzero((np.diff(score, axis=0) != 0).any(axis=1)):
            shape, seed = f.stem.rsplit("_s", 1)
            out.append({"shape": shape, "seed": int(seed), "frame": int(i) + 1,
                        "before": score[i].tolist(), "after": score[i + 1].tolist(),
                        "ball": [round(float(v), 3) for v in d["ball"][i]]})
    return out


def render(shape, seed, frame, lead_in=LEAD_IN, lead_out=LEAD_OUT, out=None):
    from lib import use_bundle
    from scenario_factory import write_scenario

    use_bundle(G.BUNDLE_VIS)
    level = write_scenario(G.shape_spec(shape), force=True)
    start, end = max(0, frame - lead_in), frame + lead_out
    frames, _balls = G.render_window(level, seed, start, end)
    out = out or OUT / f"goal_{shape}_s{seed}_f{frame}.mov"
    G.write_clip(frames, out)
    print(f"{out}  {len(frames)} frames  {len(frames) / G.FPS:.1f}s  "
          f"[{start}, {end}) goal at {frame}", flush=True)
    return out, frames


def sheet(frames, path, every=10, cols=6):
    """Contact sheet so the goal can be eyeballed without opening the clip."""
    from PIL import Image
    picks = list(range(0, len(frames), every))
    rows = (len(picks) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    sh = Image.new("RGB", (cols * w // 4, rows * h // 4))
    for n, i in enumerate(picks):
        im = Image.fromarray(np.asarray(frames[i])).resize((w // 4, h // 4))
        sh.paste(im, ((n % cols) * w // 4, (n // cols) * h // 4))
    sh.save(path)
    print(f"{path}  {len(picks)} thumbs", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        goals = find_goals()
        print(f"{len(goals)} goals in {len({(g['shape'], g['seed']) for g in goals})} matches")
        for g in goals:
            print(f"  {g['shape']:<10} s{g['seed']} f{g['frame']:<5} "
                  f"{g['before']} -> {g['after']}  ball {g['ball']}")
    elif cmd == "render":
        shape, seed, frame = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        OUT.mkdir(parents=True, exist_ok=True)
        path, frames = render(shape, seed, frame)
        sheet(frames, path.with_suffix(".png"))
    else:
        raise SystemExit(__doc__)
