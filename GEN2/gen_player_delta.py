"""gen_player_delta.py — clips that END with an exact number of players in frame,
with NOBODY hidden.

THE REWRITE, AND WHY
  The first version hit the target count by hiding players (renderScale, the trick used on
  referees). All 20 of those clips were rejected, for four reasons, and the fourth kills
  the mechanism outright:

    1. The ball sat stranded in empty grass. Hiding chose its victims by shortest dwell in
       frame, and the players moving fastest through frame are exactly the ones chasing
       the ball — so it deleted the players involved in the play and kept the bystanders.
    2. The frames were too empty to read as a match.
    3. Windows opened on kickoff clumps: the probe only ever covered frames 0-119, which
       is the restart, so players were bunched on the centre spot and nothing was live.
    4. Hiding players is the wrong approach, full stop.

  The count is therefore obtained by SELECTION, not subtraction. All 22 players render in
  every frame of every clip; we search real match windows for ones the camera happens to
  frame with exactly the target number of bodies on the last frame. Nothing is removed, so
  nothing can look removed. Fixing (4) this way also fixes (1) and (2) for free — the
  emptiness was the hiding.

TWO CHEAPER WAYS TO COUNT WERE TRIED AND REJECTED — do not retry them
  * Fitting the camera frustum geometrically from world coordinates, so the 600 cached
    sweep matches could be scanned for free. Measured against the rendered ground truth:
    93.6% correct per player, but the per-frame COUNT was exactly right only 36% of the
    time (sd 1.48). A box in world space is not the frustum, and geometry cannot know that
    one player is standing behind another.
  * Counting kit-coloured blobs in the rendered frame. Worse: bias +5.8 players, and some
    frames overcounted by 40+. The advertising hoardings are saturated blue and red, and a
    single player's shirt and shorts split into separate blobs.

  Both are fine as heuristics and useless as ground truth, and the count IS the ground
  truth here. So the exact solo probe is used, just over a deeper frame range.

HOW THE COUNT IS MEASURED
  For each base match we render a plate with all 22 players hidden, then 22 more passes
  each showing exactly one player, and diff them frame by frame. A non-trivial pixel
  difference means that player's body is inside the camera frustum on that frame. Hiding
  is render-only, so all 23 passes replay the identical match. That is 23 replays per
  match — affordable because the probe is sharded across cores and covers frames 0-450 in
  one go, giving ~250 candidate windows per match rather than one.

  The published count is this measurement. Clips hide nobody; the probe passes exist only
  to count, and are thrown away.

WHAT ELSE A WINDOW MUST SATISFY  (this is what fixes 1-3)
  * at least MIN_NEAR players within NEAR_RADIUS of the ball on BOTH the first and last
    frame, so the ball is never alone
  * the ball actually travels rather than sitting at someone's feet
  * the on-screen players are gathered around the play, not strung out to the edges
  * no set piece, goal or restart inside the window
  * nothing from the opening SKIP_START frames, which is the kickoff

THE PROBE PROCESS DIES AFTER ABOUT 46 ENGINE INSTANCES — RUN IT ONE MATCH AT A TIME
  Observed three times, and the third time pinned it down: four independent shards each
  completed EXACTLY 2 matches and then died, with no traceback and no error line. Each
  match creates 23 environments (a plate plus 22 solo passes), so every process died at
  roughly its 46th FootballEnv. That is a resource leak in the engine — an env that is
  closed but does not give back its GL context or file descriptors — not a bug in the
  logic here, and it will not show up in a short test run.

  The workaround is to let the OS reclaim everything between matches: invoke the probe
  once PER MATCH from a shell loop, not once for a whole shard. Each invocation writes its
  own map file and already-probed matches are skipped, so the loop is safe to re-run.

      for i in 0 1 2 3; do
        ( while python3 gen_player_delta.py probe_one --shard $i/4; do :; done ) &
      done

  `probe_one` probes a single unprobed match from the shard and exits 0 if it did work,
  1 when the shard has nothing left — so the `while` loop terminates on its own.

Run:  python3 gen_player_delta.py probe_one [--shard i/n]  # ONE match, then exits
      python3 gen_player_delta.py probe [--shard i/n]      # all of a shard: DIES, see above
      python3 gen_player_delta.py pick
      python3 gen_player_delta.py vis && ... inv && ... compose
Out:  GEN2/04_player_delta/{full_visibility,split_1s_4s}/
"""
import json
import subprocess
import sys

import numpy as np

from gen2_lib import (ALL_SLOTS, BUNDLE_INV, BUNDLE_VIS, CACHE, CLIP_FRAMES, HERE,
                      compose_pair, continuous, match_spec, render_window, run_log,
                      write_clip)

OUT = HERE / "04_player_delta"
PD = CACHE / "player_delta"
MAPS = PD / "counts"                 # one .npz per probed match
PICKS = PD / "picks_natural.json"

END_COUNTS = [6, 10, 12, 16]
PER_COUNT = 5

PROBE_END = 450        # frames probed per match; deep enough to leave the kickoff behind
SKIP_START = 150       # kickoff / settling — no window may start before this
DOWN = 2               # probe renders at half size; a body is still tens of pixels
MIN_PIX = 40           # changed (half-size) pixels before a player counts as in frame

NEAR_RADIUS = 0.13     # "near the ball" in pitch units (~13% of the 105 m length)
MIN_NEAR = 3           # bodies near the ball on the first and last frame
MIN_TRAVEL = 0.10      # the ball must cover at least this much ground across the window
STRIDE = 5

SETPIECE_MODES = (1, 2, 3, 4, 5, 6)      # anything that is not GM_NORMAL

# Base matches, spread across the pitch so the camera sees different densities: play near
# a touchline or a goal frames fewer bodies, midfield play frames more, which is what
# makes both the low (6) and the high (16) targets reachable.
BASES = [
    ("g2_pd01", (0.00, 0.00), (0.00, 0.00), (0.80, 0.80)),
    ("g2_pd02", (0.20, 0.12), (0.25, 0.15), (0.80, 0.80)),
    ("g2_pd03", (-0.15, -0.10), (0.15, 0.30), (0.90, 0.70)),
    ("g2_pd04", (0.35, -0.20), (0.40, 0.20), (0.80, 0.90)),
    ("g2_pd05", (-0.05, 0.25), (0.30, 0.35), (0.70, 0.90)),
    ("g2_pd06", (0.45, 0.05), (0.50, 0.25), (0.90, 0.80)),
    ("g2_pd07", (0.10, -0.28), (0.20, 0.10), (0.85, 0.85)),
    ("g2_pd08", (0.60, 0.18), (0.55, 0.30), (0.95, 0.60)),
]
SEEDS = [11, 23, 37]                      # 8 bases x 3 seeds = 24 matches


def bases():
    for name, ball, (pl, pr), diff in BASES:
        yield name, match_spec(name, ball=ball, offsides=True, difficulty=diff,
                               push_left=pl, push_right=pr)


def jobs():
    for name, _spec in bases():
        for seed in SEEDS:
            yield name, seed


def parse_shard(argv):
    for i, a in enumerate(argv):
        if a.startswith("--shard"):
            spec = a.split("=", 1)[1] if "=" in a else argv[i + 1]
            n, d = spec.split("/")
            return int(n), int(d)
    return 0, 1


# ── PHASE probe: exact per-frame, per-player on-screen map ──────────────────────
def _shard_todo(shard_i, shard_n):
    return [(lvl, sd) for k, (lvl, sd) in enumerate(jobs())
            if k % shard_n == shard_i and not (MAPS / f"{lvl}_s{sd}.npz").exists()]


def probe_match(lvl, seed):
    """Measure, for every frame, which of the 22 players is inside the camera frame.

    Continuity is NOT checked here. It is a property of the 50-frame window we eventually
    cut, not of the whole 450-frame probe: rejecting a match because a goal happened at
    frame 300 throws away frames 150-250, which are fine. Probing is the expensive part,
    so probe everything and let phase_pick reject per window.
    """
    plate, _balls = render_window(lvl, seed, 0, PROBE_END,
                                  hide_slots=",".join(ALL_SLOTS))
    plate = np.array(plate, dtype=np.int16)[:, ::DOWN, ::DOWN]
    on = np.zeros((len(ALL_SLOTS), len(plate)), dtype=bool)
    for si, slot in enumerate(ALL_SLOTS):
        hide = ",".join(s for s in ALL_SLOTS if s != slot)         # show ONLY this one
        solo, _ = render_window(lvl, seed, 0, PROBE_END, hide_slots=hide)
        solo = np.array(solo, dtype=np.int16)[:, ::DOWN, ::DOWN]
        d = np.abs(solo - plate).sum(axis=3)
        on[si] = (d > 30).reshape(len(plate), -1).sum(axis=1) > MIN_PIX
    log = run_log(lvl, seed, PROBE_END)          # positions for the football-quality tests
    T = min(on.shape[1], len(log["ball"]))
    np.savez_compressed(
        MAPS / f"{lvl}_s{seed}.npz", usable=np.array(True), on=on[:, :T],
        ball=log["ball"][:T], left=log["left"][:T], right=log["right"][:T],
        game_mode=log["game_mode"][:T])
    tot = on[:, :T].sum(axis=0)
    print(f"  [probe] {lvl}_s{seed}: in-frame min={int(tot.min())} "
          f"max={int(tot.max())}", flush=True)


def _probe_setup(shard_i):
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(BUNDLE_VIS)
    MAPS.mkdir(parents=True, exist_ok=True)
    for _name, spec in bases():
        write_scenario(spec, force=(shard_i == 0))


def phase_probe_one():
    """Probe ONE unprobed match from this shard, then exit.

    Exit 0 = did work, exit 1 = nothing left. Driven from a shell `while` loop so each
    match gets a fresh process — the engine leaks resources and a process dies at roughly
    its 46th FootballEnv, which is two matches' worth of passes. See the module docstring.
    """
    shard_i, shard_n = parse_shard(sys.argv[2:])
    _probe_setup(shard_i)
    todo = _shard_todo(shard_i, shard_n)
    if not todo:
        print(f"shard {shard_i}/{shard_n}: nothing left to probe")
        sys.exit(1)
    lvl, seed = todo[0]
    print(f"shard {shard_i}/{shard_n}: {len(todo)} left, probing {lvl}_s{seed}", flush=True)
    probe_match(lvl, seed)
    sys.exit(0)


def phase_probe():
    """Probe a whole shard in one process. NOTE: this reliably dies after ~2 matches —
    use probe_one from a shell loop instead. Kept because it is the clearest statement of
    what the phase does, and it is fine for probing a single match by hand."""
    shard_i, shard_n = parse_shard(sys.argv[2:])
    _probe_setup(shard_i)
    todo = _shard_todo(shard_i, shard_n)
    print(f"shard {shard_i}/{shard_n}: {len(todo)} matches to probe", flush=True)
    for lvl, seed in todo:
        probe_match(lvl, seed)
    print(f"shard {shard_i}: done")


# ── PHASE pick: windows the camera naturally frames at the target count ─────────
def near_ball(ball_xy, left, right, radius=NEAR_RADIUS):
    pl = np.concatenate([left, right], axis=0)
    return int((np.linalg.norm(pl - ball_xy, axis=1) < radius).sum())


def score_window(d, on, s, e, want):
    """Quality score for [s, e] at `want` players on screen, or None if unusable.

    Every test here answers one of the four rejections: the ball must be with people (1),
    the frame must read as football (2), and the play must be live rather than a restart
    or a jog (3).
    """
    if int(on[:, e].sum()) != want:
        return None
    ball = d["ball"][s:e + 1]
    if not continuous(ball):
        return None
    if any(m in SETPIECE_MODES for m in d["game_mode"][s:e + 1]):
        return None

    near_first = near_ball(ball[0][:2], d["left"][s], d["right"][s])
    near_last = near_ball(ball[-1][:2], d["left"][e], d["right"][e])
    if near_first < MIN_NEAR or near_last < MIN_NEAR:
        return None

    travel = float(np.linalg.norm(ball[-1][:2] - ball[0][:2]))
    if travel < MIN_TRAVEL:
        return None
    path = float(np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1).sum())

    shown = np.concatenate([d["left"][e], d["right"][e]], axis=0)[on[:, e]]
    spread = float(np.linalg.norm(shown - ball[-1][:2], axis=1).mean()) if len(shown) else 9.9
    return {"near_first": near_first, "near_last": near_last,
            "travel": round(travel, 3), "path": round(path, 3),
            "spread": round(spread, 3),
            "score": near_last * 3 + near_first * 2 + path * 6 - spread * 8}


def phase_pick():
    files = sorted(MAPS.glob("*.npz"))
    if not files:
        sys.exit(f"no probe maps in {MAPS} — run: python3 gen_player_delta.py probe")
    cands = {w: [] for w in END_COUNTS}
    n_usable = 0
    for path in files:
        z = np.load(path)
        if not bool(z["usable"]):
            continue
        n_usable += 1
        d = {k: z[k] for k in ("ball", "left", "right", "game_mode")}
        on = z["on"]
        T = on.shape[1]
        for s in range(SKIP_START, T - CLIP_FRAMES, STRIDE):
            e = s + CLIP_FRAMES - 1
            for want in END_COUNTS:
                r = score_window(d, on, s, e, want)
                if r:
                    cands[want].append({"match": path.stem, "level": path.stem.rsplit("_s", 1)[0],
                                        "seed": int(path.stem.rsplit("_s", 1)[1]),
                                        "start": s, "end": e + 1, "end_count": want, **r})

    picks, used = [], set()
    for want in sorted(END_COUNTS, reverse=True):     # 16 is the scarcest, fill it first
        rows = sorted(cands[want], key=lambda x: -x["score"])
        got = 0
        for r in rows:
            if got >= PER_COUNT:
                break
            if r["match"] in used:
                continue                  # one clip per match: 20 clips, 20 matches
            used.add(r["match"])
            picks.append(r)
            got += 1
        print(f"  end_count={want:2d}: {got}/{PER_COUNT} from "
              f"{len(cands[want])} candidate windows")
    PICKS.write_text(json.dumps(picks, indent=2, default=float))
    print(f"phase pick: {len(picks)} windows from {len({p['match'] for p in picks})} "
          f"distinct matches ({n_usable} usable) -> {PICKS}")


# ── PHASE vis / inv: render with NOBODY hidden ─────────────────────────────────
def _render(bundle, tag):
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(bundle)
    for _n, spec in bases():
        write_scenario(spec, force=True)
    picks = json.loads(PICKS.read_text())
    for i, p in enumerate(picks):
        # hide_slots deliberately empty — all 22 players are rendered.
        frames, _b = render_window(p["level"], p["seed"], p["start"], p["end"])
        np.savez_compressed(PD / f"nd{i:02d}_{tag}.npz", frames=np.array(frames))
        print(f"  [{tag}] {p['match']} end={p['end_count']} ({len(frames)} frames)",
              flush=True)


def phase_vis():
    _render(BUNDLE_VIS, "vis")


def phase_inv():
    _render(BUNDLE_INV, "inv")


def phase_compose():
    from grid import cell_of
    picks = json.loads(PICKS.read_text())
    rows = [("clip", "end_count", "players_in_frame_last", "players_in_play",
             "players_hidden", "near_ball_first", "near_ball_last", "ball_path",
             "level", "seed", "match", "start_frame", "end_frame", "n_frames", "seconds",
             "ball_start_cell", "start_px", "start_py",
             "ball_final_cell", "final_px", "final_py")]
    counters = {}
    for i, p in enumerate(picks):
        vis = np.load(PD / f"nd{i:02d}_vis.npz")["frames"]
        inv = np.load(PD / f"nd{i:02d}_inv.npz")["frames"]
        full, split, (spx, spy), (fpx, fpy) = compose_pair(vis, inv)
        n = counters.get(p["end_count"], 0) + 1
        counters[p["end_count"]] = n
        clip = f"end{p['end_count']:02d}_{n:02d}"
        write_clip(full, OUT / "full_visibility" / f"{clip}.mov")
        write_clip(split, OUT / "split_1s_4s" / f"{clip}.mov")
        rows.append((clip, p["end_count"], p["end_count"], 22, 0,
                     p["near_first"], p["near_last"], p["path"],
                     p["level"], p["seed"], p["match"], p["start"], p["end"],
                     len(full), round(len(full) / 10, 1),
                     cell_of(spx, spy), round(spx, 1), round(spy, 1),
                     cell_of(fpx, fpy), round(fpx, 1), round(fpy, 1)))
        print(f"  [compose] {clip}: {p['end_count']} in frame, {p['near_last']} near "
              f"the ball, ball {cell_of(spx, spy)} -> {cell_of(fpx, fpy)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ground_truth.csv").write_text(
        "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    print(f"phase compose: {len(picks)} situations -> {len(picks) * 2} clips in {OUT}")


def phase_all():
    """probe_one is looped until it reports nothing left, so every match gets a fresh
    process (see the module docstring on the ~46-env leak)."""
    print("\n===== phase probe =====", flush=True)
    while subprocess.run([sys.executable, __file__, "probe_one"]).returncode == 0:
        pass
    for mode in ("pick", "vis", "inv", "compose"):
        print(f"\n===== phase {mode} =====", flush=True)
        subprocess.run([sys.executable, __file__, mode], check=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"probe": phase_probe, "probe_one": phase_probe_one, "pick": phase_pick,
     "vis": phase_vis, "inv": phase_inv, "compose": phase_compose,
     "all": phase_all}[mode]()
