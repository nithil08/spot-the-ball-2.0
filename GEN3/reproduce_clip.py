"""reproduce_clip.py — rebuild named GEN3 clips from the written recipe, then measure how
far the rebuild lands from the batch that shipped.

WHAT "REPRODUCE" MEANS HERE
  Not a file copy. The only inputs are the two tracked recipe files —

      clips/ground_truth.csv      shape, seed, start/end frame, camera offset, and the
                                  plate measurement of the final ball (the fallback
                                  `compose` needs when a player stands in front of it)
      clips/reproducibility.csv   the sha256 of the scenario source and of the play

  — plus the SHAPES table in gen3_lib. The scenario module is re-emitted from SHAPES, the
  match is replayed from frame 0 to the clip's end frame, both render passes are taken
  fresh, and the grid, red circle and 1 s/4 s split are burned in again. Nothing is read
  from `_cache/renders`, which is where the shipped frames live. That is the point: if a
  rebuild used the cache it would prove only that np.load works.

WHY THE PASSES ARE SEPARATE PROCESSES
  `use_bundle` sets GFOOTBALL_DATA_DIR and only takes effect before the engine loads, so
  the visible (gen3) and invisible (gen3_ball_invisible) passes cannot share a process —
  a mid-process swap would silently render both passes from the same bundle and the ball
  diff would come out empty. The no-argument form drives the phases in order for you.

THREE LEVELS OF COMPARISON, because "the same clip" can fail in three different places
  1. THE PLAY      play_sha over the full replay: ball xyz, all 22 players, possession,
                   game mode, score, every step. Exact equality, no tolerance.
  2. THE PIXELS    the fresh render passes against the cached shipped ones, before any
                   encoding. This is where a changed bundle, kit or camera would show.
  3. THE VIDEO     the finished .mov decoded back to RGB against the shipped .mov, also
                   decoded. What a viewer actually sees, H.264 round trip included.

  Measured on clips 01-03: all three levels identical, and the .mov files come out
  byte-identical too — this ffmpeg build's QuickTime muxer stamps no creation time, so
  there is nothing non-deterministic left in the chain. Do not rely on that last part as
  a test, though: a muxer that did stamp one would fail it while every frame matched.
  Decoded frames are the check that means something.

USAGE
    python3 reproduce_clip.py clip_01 clip_02 clip_03     # all phases, in order
    python3 reproduce_clip.py pass_vis clip_01            # one phase (own process)
    python3 reproduce_clip.py pass_inv clip_01
    python3 reproduce_clip.py compose  clip_01
    python3 reproduce_clip.py compare  clip_01

  Output lands in `_repro/`, never in `clips/` — the shipped batch is not touched.
"""
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))

import gen3_lib as G                                                    # noqa: E402
import verify_reproducible as VR                                        # noqa: E402

GT_CSV = HERE / "clips" / "ground_truth.csv"
REPRO_CSV = HERE / "clips" / "reproducibility.csv"
SHIPPED = HERE / "clips"
SHIPPED_RENDERS = G.CACHE / "renders"

OUT = HERE / "_repro"
RAW = G.CACHE / "_repro_raw"          # fresh render passes; under _cache, so gitignored

VARIANTS = ("full_visibility", "split_1s_4s")
HEADER = ("clip,shape,seed,match,start_frame,end_frame,offset_x,offset_y,frames,"
          "ball_start_cell,start_px,start_py,ball_final_cell,final_px,final_py")


def recipes(clips):
    """The recipe rows for the named clips, from the tracked CSVs only."""
    gt = {r["clip"]: r for r in csv.DictReader(GT_CSV.open())}
    rp = {r["clip"]: r for r in csv.DictReader(REPRO_CSV.open())}
    out = []
    for c in clips:
        if c not in gt:
            sys.exit(f"unknown clip {c!r}; known: {' '.join(sorted(gt))}")
        r = dict(gt[c])
        r["scenario_sha"] = rp.get(c, {}).get("scenario_sha", "")
        r["play_sha"] = rp.get(c, {}).get("play_sha", "")
        out.append(r)
    return out


def frame_sha(arr):
    h = hashlib.sha256()
    a = np.ascontiguousarray(arr)
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def note_path(clip, tag):
    return RAW / f"{clip}_{tag}.sha"


# ══ phases 1-2: the two render passes, one bundle each ════════════════════════
def cmd_pass(tag, clips):
    from lib import use_bundle
    from scenario_factory import write_scenario, SCENARIOS_DIR
    use_bundle(G.BUNDLE_VIS if tag == "vis" else G.BUNDLE_INV)
    RAW.mkdir(parents=True, exist_ok=True)

    for r in recipes(clips):
        clip, shape, seed = r["clip"], r["shape"], int(r["seed"])
        start, end = int(r["start_frame"]), int(r["end_frame"])
        off = (float(r["offset_x"]), float(r["offset_y"]))

        # Re-emit the scenario from SHAPES rather than trusting the module on disk, then
        # check it against the fingerprint that shipped. A changed SHAPES table is
        # otherwise completely invisible: same level name, different football.
        write_scenario(G.shape_spec(shape), force=True)
        sha = hashlib.sha256((SCENARIOS_DIR / f"g3_{shape}.py").read_bytes()).hexdigest()
        ok = "match" if sha == r["scenario_sha"] else "DIFFERENT"
        print(f"  [{tag}] {clip}: scenario g3_{shape} sha {sha[:12]} ({ok})", flush=True)

        lines = [f"scenario_sha {sha}"]
        if tag == "vis":
            # The play itself, hashed exactly as verify_reproducible does. Costs one extra
            # replay and is the only check that separates "different football" from
            # "different picture".
            log = G.run_log(f"g3_{shape}", seed, end, offset=off)
            play = VR.log_sha(log)
            ok = "match" if play == r["play_sha"] else "DIFFERENT"
            print(f"  [{tag}] {clip}: play sha {play[:12]} ({ok})", flush=True)
            lines.append(f"play_sha {play}")

        frames, _ = G.render_window(f"g3_{shape}", seed, start, end,
                                    hide_slots="", offset=off)
        arr = np.array(frames, dtype=np.uint8)
        np.savez_compressed(RAW / f"{clip}_{tag}.npz", frames=arr)
        lines.append(f"frames_sha {frame_sha(arr)}")
        note_path(clip, tag).write_text("\n".join(lines) + "\n")
        print(f"  [{tag}] {clip}: {len(arr)} frames rendered fresh", flush=True)
        del frames, arr
    return 0


# ══ phase 3: compose — grid, red circle, and the 1 s/4 s split ════════════════
def cmd_compose(clips):
    rows = [HEADER]
    for r in recipes(clips):
        clip = r["clip"]
        vis = np.load(RAW / f"{clip}_vis.npz")["frames"]
        inv = np.load(RAW / f"{clip}_inv.npz")["frames"]
        full, split, (spx, spy), (fpx, fpy) = G.compose_pair(
            vis, inv, final_fallback=(float(r["final_px"]), float(r["final_py"])))
        G.write_clip(full, OUT / "full_visibility" / f"{clip}.mov")
        G.write_clip(split, OUT / "split_1s_4s" / f"{clip}.mov")
        rows.append(",".join(map(str, [
            clip, r["shape"], r["seed"], r["match"], r["start_frame"], r["end_frame"],
            r["offset_x"], r["offset_y"], len(full),
            G.cell_of(spx, spy), round(spx, 1), round(spy, 1),
            G.cell_of(fpx, fpy), round(fpx, 1), round(fpy, 1)])))
        print(f"  [compose] {clip}: ball {G.cell_of(spx, spy)} -> {G.cell_of(fpx, fpy)}",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ground_truth.csv").write_text("\n".join(rows) + "\n")
    print(f"compose: {len(rows) - 1} clips -> {OUT}")
    return 0


# ══ phase 4: compare ══════════════════════════════════════════════════════════
def decode(path):
    """A .mov back to (N, H, W, 3) uint8, so two encodes are compared as pictures."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True)
    w, h = (int(x) for x in probe.stdout.strip().split("x"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(raw.stdout, np.uint8).reshape(-1, h, w, 3)


def diff_stats(a, b):
    """How two stacks of frames differ, in the terms that matter for a render.

    Frame by frame rather than on the whole stack: a 125-frame clip is 230 MB as uint8
    and the int16 promotion the diff needs doubles it, so the whole-stack form peaks
    around a gigabyte for what is a per-frame question.
    """
    if a.shape != b.shape:
        return {"shape": f"{a.shape} vs {b.shape}", "identical": False}
    npx = max_abs = touched = 0
    total_abs = 0.0
    for i in range(len(a)):
        d = np.abs(a[i].astype(np.int16) - b[i].astype(np.int16))
        k = int((d.max(axis=-1) > 0).sum())
        npx += k
        touched += k > 0
        max_abs = max(max_abs, int(d.max()))
        total_abs += float(d.sum())
    return {"identical": npx == 0,
            "px_differing": npx,
            "px_total": int(a.shape[0] * a.shape[1] * a.shape[2]),
            "max_abs": max_abs,
            "mean_abs": total_abs / a.size,
            "frames_touched": touched}


def read_notes(clip):
    out = {}
    for tag in ("vis", "inv"):
        p = note_path(clip, tag)
        if p.exists():
            for line in p.read_text().split("\n"):
                if line.strip():
                    k, v = line.split()
                    out[f"{tag}_{k}"] = v
    return out


def cmd_compare(clips):
    gt_new = {r["clip"]: r for r in csv.DictReader((OUT / "ground_truth.csv").open())}
    md = ["# GEN3 rebuild vs the shipped batch", "",
          "Rebuilt from `clips/ground_truth.csv` + `clips/reproducibility.csv` alone: "
          "scenario re-emitted from `SHAPES`, match replayed from frame 0, both render "
          "passes taken fresh, grid and split burned in again. Nothing read from "
          "`_cache/renders`.", ""]
    all_same = True

    for r in recipes(clips):
        clip = r["clip"]
        notes = read_notes(clip)
        md += [f"## {clip} — {r['shape']} seed {r['seed']}, frames "
               f"{r['start_frame']}-{r['end_frame']}, offset "
               f"({r['offset_x']}, {r['offset_y']})", ""]

        # 1. the play
        for label, key, want in (("scenario", "vis_scenario_sha", r["scenario_sha"]),
                                 ("play", "vis_play_sha", r["play_sha"])):
            got = notes.get(key, "")
            same = bool(got) and got == want
            all_same &= same
            md.append(f"- **{label} sha** `{got[:16]}` vs shipped `{want[:16]}` — "
                      f"{'identical' if same else 'DIFFERENT'}")

        # 2. the pixels, before encoding
        for tag in ("vis", "inv"):
            cached = SHIPPED_RENDERS / f"{r['match']}_{r['start_frame']}_{r['end_frame']}_{tag}.npz"
            if not cached.exists():
                md.append(f"- **{tag} render pass** — shipped frames not in the cache, "
                          "skipped")
                continue
            s = diff_stats(np.load(RAW / f"{clip}_{tag}.npz")["frames"],
                           np.load(cached)["frames"])
            all_same &= s["identical"]
            md.append(f"- **{tag} render pass** (raw frames, pre-encode) — "
                      + fmt(s))

        # 3. the finished video
        for v in VARIANTS:
            a, b = OUT / v / f"{clip}.mov", SHIPPED / v / f"{clip}.mov"
            if not b.exists():
                md.append(f"- **{v}.mov** — shipped clip missing, skipped")
                continue
            s = diff_stats(decode(a), decode(b))
            all_same &= s["identical"]
            same_bytes = (hashlib.sha256(a.read_bytes()).hexdigest()
                          == hashlib.sha256(b.read_bytes()).hexdigest())
            md.append(f"- **{v}.mov** (decoded RGB) — " + fmt(s)
                      + f"; file bytes {'identical' if same_bytes else 'differ'}")

        # 4. the answer the clip is asking for
        n, o = gt_new[clip], r
        fields = ("ball_start_cell", "start_px", "start_py",
                  "ball_final_cell", "final_px", "final_py")
        bad = [f for f in fields if n[f] != o[f]]
        all_same &= not bad
        if bad:
            for f in bad:
                md.append(f"- **ground truth {f}** — rebuilt `{n[f]}` vs shipped "
                          f"`{o[f]}` — DIFFERENT")
        else:
            md.append(f"- **ground truth** — ball {n['ball_start_cell']} "
                      f"({n['start_px']}, {n['start_py']}) -> {n['ball_final_cell']} "
                      f"({n['final_px']}, {n['final_py']}), identical to shipped")
        md.append("")

    md += ["## Verdict", "",
           "Every level identical: the rebuild is the shipped clip."
           if all_same else
           "At least one level differs — see the lines marked DIFFERENT above.", ""]
    (OUT / "COMPARISON.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten to {OUT / 'COMPARISON.md'}")
    return 0


def fmt(s):
    if s.get("shape"):
        return f"shape mismatch {s['shape']}"
    if s["identical"]:
        return f"identical ({s['px_total']:,} pixels, max abs diff 0)"
    return (f"{s['px_differing']:,} of {s['px_total']:,} pixels differ "
            f"({s['px_differing'] / s['px_total']:.3%}), across "
            f"{s['frames_touched']} frames, max abs {s['max_abs']}, "
            f"mean abs {s['mean_abs']:.4f}")


# ══ driver ════════════════════════════════════════════════════════════════════
def drive(clips):
    for phase in ("pass_vis", "pass_inv", "compose", "compare"):
        print(f"\n== {phase} ==", flush=True)
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), phase, *clips]).returncode
        if rc != 0:
            sys.exit(f"{phase} failed (exit {rc})")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    phases = {"pass_vis": lambda c: cmd_pass("vis", c),
              "pass_inv": lambda c: cmd_pass("inv", c),
              "compose": cmd_compose, "compare": cmd_compare}
    if args[0] in phases:
        sys.exit(phases[args[0]](args[1:] or ["clip_01"]))
    sys.exit(drive([a if a.startswith("clip_") else f"clip_{int(a):02d}" for a in args]))
