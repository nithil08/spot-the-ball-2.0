"""gen_player_delta.py — GEN2 clips controlled by how many players are IN FRAME AT THE END.

WHAT CHANGED FROM THE GEN1 "8 -> 6" CLIPS
  The old MATCH_SITUATIONS/05_frame_visibility_8to6 set pinned BOTH ends: exactly 8
  players in frame on the first frame and exactly 6 on the last. Per the GEN2 brief the
  start count is not interesting — "it doesn't really matter how many players we start
  with in the 11 v 11 game, all I need to control is how many players we end with". So
  only the FINAL frame's count is constrained here; the opening count is whatever the
  play happens to give, which also makes the clips look less staged.

  Targets: 5 clips each ending with 6, 10, 12 and 16 players in frame.

ALL 22 PLAYERS ARE ALWAYS IN THE MATCH
  The count on screen is a property of the CAMERA, not the squad. Every clip is a single
  continuous render with a SINGLE fixed hide set — no splice, no mid-clip pop-in. Players
  that are hidden are hidden for all 50 frames; the ones you see leave or enter frame
  because they ran and the camera tracked the ball, which is what a real broadcast looks
  like. Hiding is render-only (renderScale, the same engine trick that shrinks referees),
  so all 23 passes over a match replay the identical play.

HOW "IN FRAME" IS MEASURED, NOT GUESSED
  For each base match we render a `plate` with all 22 players hidden, then `solo_i` with
  only player i shown, and diff them frame by frame. A non-trivial pixel difference means
  player i's body is inside the camera frustum on that frame. That gives an exact
  per-frame, per-player on-screen map, from which a fixed hide set is solved so that the
  shown players number exactly N on the final frame.

  Players never on camera during the window are left SHOWN — they need no hiding, they are
  simply out of shot, and hiding them would be wasted work.

Run:  python3 gen_player_delta.py all
      (phases: probe -> pick -> vis -> inv -> compose; the bundle must be chosen before
       gfootball is imported, so each engine phase is its own process)
Out:  GEN2/04_player_delta/{full_visibility,split_1s_4s}/
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from gen2_lib import (ALL_SLOTS, BUNDLE_INV, BUNDLE_VIS, CACHE, CLIP_FRAMES, HERE,
                      compose_pair, continuous, match_spec, render_window, write_clip)

OUT = HERE / "04_player_delta"
PD = CACHE / "player_delta"
MAP = PD / "onscreen.json"
PICKS = PD / "picks.json"

END_COUNTS = [6, 10, 12, 16]
PER_COUNT = 5
PROBE_STEPS = 120          # how far into each match the on-screen map is measured
DOWN = 2                   # probe renders at half size; a body is still tens of pixels
MIN_PIX = 40               # changed (half-size) pixels before a player counts as in frame

# Base matches. Spread across the pitch so the camera sees different densities: play near
# a touchline or a goal naturally frames fewer bodies, midfield play frames more, which is
# what makes both the low (6) and high (16) end-counts reachable.
BASES = [
    ("pd01", (0.00, 0.00), (0.00, 0.00), (0.80, 0.80)),
    ("pd02", (0.20, 0.12), (0.25, 0.15), (0.80, 0.80)),
    ("pd03", (-0.15, -0.10), (0.15, 0.30), (0.90, 0.70)),
    ("pd04", (0.35, -0.20), (0.40, 0.20), (0.80, 0.90)),
    ("pd05", (-0.05, 0.25), (0.30, 0.35), (0.70, 0.90)),
    ("pd06", (0.45, 0.05), (0.50, 0.25), (0.90, 0.80)),
    ("pd07", (0.10, -0.28), (0.20, 0.10), (0.85, 0.85)),
    ("pd08", (0.60, 0.18), (0.55, 0.30), (0.95, 0.60)),
    ("pd09", (-0.40, 0.05), (0.20, 0.45), (0.75, 0.85)),
    ("pd10", (0.05, 0.32), (0.35, 0.25), (0.85, 0.75)),
]
SEEDS = [11, 23, 37]       # each base x seed is a different match (30 probes ≈ 50 min)


def bases():
    for name, ball, (pl, pr), diff in BASES:
        yield name, match_spec(f"g2_{name}", ball=ball, offsides=True, difficulty=diff,
                               push_left=pl, push_right=pr)


# ── PHASE probe ────────────────────────────────────────────────────────────────
def phase_probe():
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(BUNDLE_VIS)
    PD.mkdir(parents=True, exist_ok=True)
    maps = json.loads(MAP.read_text()) if MAP.exists() else {}

    for name, spec in bases():
        level = write_scenario(spec, force=True)
        for seed in SEEDS:
            key = f"{level}_s{seed}"
            if key in maps:
                continue
            plate, balls = render_window(level, seed, 0, PROBE_STEPS,
                                         hide_slots=",".join(ALL_SLOTS))
            if not continuous(balls):
                print(f"  skip {key}: play breaks (goal / restart)", flush=True)
                maps[key] = None
                continue
            plate = np.array(plate, dtype=np.int16)[:, ::DOWN, ::DOWN]
            on = np.zeros((len(ALL_SLOTS), len(plate)), dtype=bool)
            for si, slot in enumerate(ALL_SLOTS):
                hide = ",".join(s for s in ALL_SLOTS if s != slot)   # show ONLY this one
                solo, _ = render_window(level, seed, 0, PROBE_STEPS, hide_slots=hide)
                solo = np.array(solo, dtype=np.int16)[:, ::DOWN, ::DOWN]
                d = np.abs(solo - plate).sum(axis=3)
                on[si] = (d > 30).reshape(len(plate), -1).sum(axis=1) > MIN_PIX
            maps[key] = {"level": level, "seed": seed, "on": on.tolist()}
            tot = on.sum(axis=0)
            print(f"  [probe] {key}: in-frame min={int(tot.min())} max={int(tot.max())}",
                  flush=True)
            MAP.write_text(json.dumps(maps))       # checkpoint every match
    print(f"phase probe: {sum(1 for v in maps.values() if v)} usable matches")


# ── PHASE pick ─────────────────────────────────────────────────────────────────
def choose_hide_set(on, s, e, want_end):
    """Minimal fixed hide set giving exactly `want_end` players in frame on frame `e`.

    ONLY the end count is pinned, so hide as little as possible: of the players in frame
    on the last frame, hide just the surplus (len(end_in) - want_end) and leave the entire
    rest of the squad shown. Everyone not in frame at the end stays visible precisely
    because they are out of shot by then anyway — but they may well be on camera earlier,
    which is what lets the count fall naturally as the camera tracks away from them.

    Hiding the surplus by SHORTEST dwell keeps the group that stays on screen the stable
    one, so the count settles rather than flickering at the frame edge.

    An earlier version kept only end-frame players and hid everyone else. That was wrong
    for GEN2: it forced n_first <= want_end, so the on-screen count could never fall
    during a clip, and the start was implicitly pinned — exactly what the brief says not
    to control.

    Returns (hide_slots, n_first, n_last, per_frame), or None if the window cannot make
    the requested count (e.g. only 5 bodies are ever on camera and 16 are wanted).
    """
    n_slots = len(on)
    end_in = [i for i in range(n_slots) if on[i][e]]
    if len(end_in) < want_end:
        return None
    dwell = {i: sum(on[i][s:e + 1]) for i in range(n_slots)}
    surplus = sorted(end_in, key=lambda i: dwell[i])[:len(end_in) - want_end]
    hidden = set(surplus)
    shown = [i for i in range(n_slots) if i not in hidden]
    n_last = sum(1 for i in shown if on[i][e])
    if n_last != want_end:
        return None
    n_first = sum(1 for i in shown if on[i][s])
    per_frame = [sum(1 for i in shown if on[i][t]) for t in range(s, e + 1)]
    return [ALL_SLOTS[i] for i in sorted(hidden)], n_first, n_last, per_frame


def phase_pick():
    """Assign each of the 20 clips its own source match.

    ONE CLIP PER MATCH, GLOBALLY. An earlier version reserved matches per end-count only,
    which let the same match supply all four counts — and because the windows were offset
    by just 5 frames, end16_01 and end12_01 came out sharing 45 of their 50 frames. Those
    are not four clips, they are one passage of play with different players hidden. With
    matches consumed globally the 20 clips are 20 different matches.

    Hardest count first: 16 needs a crowded frame and only some matches ever have 16
    bodies on camera at once, while 6 can be made from almost any window — so taking the
    6s first would eat the crowded matches the 16s depend on.
    """
    maps = json.loads(MAP.read_text())
    usable = {k: v for k, v in maps.items() if v}
    picks, used_matches = [], set()
    for want in sorted(END_COUNTS, reverse=True):
        got = 0
        for key, rec in usable.items():
            if got >= PER_COUNT:
                break
            if key in used_matches:
                continue
            on = rec["on"]
            T = len(on[0])
            for s in range(0, T - CLIP_FRAMES, 5):
                e = s + CLIP_FRAMES - 1
                res = choose_hide_set(on, s, e, want)
                if res is None:
                    continue
                hide, n0, n1, per_frame = res
                picks.append({"key": key, "level": rec["level"], "seed": rec["seed"],
                              "start": s, "end": e + 1, "end_count": want,
                              "hide": ",".join(hide), "n_first": n0, "n_last": n1,
                              "n_hidden": len(hide), "in_frame_per_frame": per_frame})
                used_matches.add(key)
                got += 1
                break
        print(f"  end_count={want:2d}: {got}/{PER_COUNT}")
    n_matches = len({p['key'] for p in picks})
    print(f"phase pick: {len(picks)} windows from {n_matches} distinct matches "
          f"({len(usable)} usable) -> {PICKS}")
    if n_matches < len(picks):
        print("  WARNING: a match is reused — probe more base matches for full diversity")
    PICKS.write_text(json.dumps(picks, indent=2))


# ── PHASE vis / inv ────────────────────────────────────────────────────────────
def _render(bundle, tagname):
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(bundle)
    for _name, spec in bases():
        write_scenario(spec, force=True)     # levels must match what was probed
    picks = json.loads(PICKS.read_text())
    for i, p in enumerate(picks):
        frames, _b = render_window(p["level"], p["seed"], p["start"], p["end"],
                                   hide_slots=p["hide"])
        np.savez_compressed(PD / f"pd{i:02d}_{tagname}.npz", frames=np.array(frames))
        print(f"  [{tagname}] {p['key']} end={p['end_count']} ({len(frames)} frames)",
              flush=True)


def phase_vis():
    _render(BUNDLE_VIS, "vis")


def phase_inv():
    _render(BUNDLE_INV, "inv")


# ── PHASE compose ──────────────────────────────────────────────────────────────
def phase_compose():
    from grid import cell_of
    picks = json.loads(PICKS.read_text())
    rows = [("clip", "end_count", "players_in_frame_first", "players_in_frame_last",
             "players_in_play", "hidden_slots", "level", "seed", "start_frame",
             "end_frame", "n_frames", "seconds", "ball_start_cell", "start_px",
             "start_py", "ball_final_cell", "final_px", "final_py")]
    counters = {}
    for i, p in enumerate(picks):
        vis = np.load(PD / f"pd{i:02d}_vis.npz")["frames"]
        inv = np.load(PD / f"pd{i:02d}_inv.npz")["frames"]
        full, split, (spx, spy), (fpx, fpy) = compose_pair(vis, inv)
        n = counters.get(p["end_count"], 0) + 1
        counters[p["end_count"]] = n
        clip = f"end{p['end_count']:02d}_{n:02d}"
        write_clip(full, OUT / "full_visibility" / f"{clip}.mov")
        write_clip(split, OUT / "split_1s_4s" / f"{clip}.mov")
        rows.append((clip, p["end_count"], p["n_first"], p["n_last"], 22, p["n_hidden"],
                     p["level"], p["seed"], p["start"], p["end"], len(full),
                     round(len(full) / 10, 1), cell_of(spx, spy), round(spx, 1),
                     round(spy, 1), cell_of(fpx, fpy), round(fpx, 1), round(fpy, 1)))
        print(f"  [compose] {clip}: in frame {p['n_first']} -> {p['n_last']}, "
              f"ball {cell_of(spx, spy)} -> {cell_of(fpx, fpy)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ground_truth.csv").write_text(
        "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    print(f"phase compose: {len(picks)} situations -> {len(picks) * 2} clips in {OUT}")


def phase_all():
    for mode in ("probe", "pick", "vis", "inv", "compose"):
        print(f"\n===== phase {mode} =====", flush=True)
        subprocess.run([sys.executable, __file__, mode], check=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"probe": phase_probe, "pick": phase_pick, "vis": phase_vis, "inv": phase_inv,
     "compose": phase_compose, "all": phase_all}[mode]()
