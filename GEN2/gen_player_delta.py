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
  match — affordable because the probe is sharded across cores and covers frames 0-700 in
  one go, giving hundreds of candidate windows per match rather than one.

  The published count is this measurement. Clips hide nobody; the probe passes exist only
  to count, and are thrown away.

WHAT ELSE A WINDOW MUST SATISFY  (this is what fixes 1-3)
  * enough players within NEAR_RADIUS of the ball on BOTH the first and last frame that
    it is never sitting alone — see min_near_for(), which scales the bar with how many
    players are on screen at all
  * the ball actually travels rather than sitting at someone's feet
  * the on-screen players are gathered around the play, not strung out to the edges
  * no set piece AWARDED inside the window (one already pending when the window opens is
    fine — that is just play resuming); no goal, no teleport
  * nothing from the opening SKIP_START frames, which is the kickoff

END_COUNT = 6 IS THE HARD ONE, AND IT IS CAMERA GEOMETRY, NOT A BUG
  Measured over 4500 windows in 36 matches: the camera frames exactly 6 players on 0.8% of
  end-frames, and of those 37 windows, 27 had a restart awarded inside and 9 had the ball
  unattended — one was usable. The reason is structural: with the stock tracking camera,
  "few players on screen" almost always means play is jammed into a corner, which is
  precisely when the ball is about to go out. Few-players and stoppage are the same event.

  The counterattack bases (pd19-pd21) exist to break that coupling — a break into open
  ground is the one situation where live, attended play genuinely frames very few bodies.
  If 6 still cannot be filled, widen the seeds on those rather than relaxing the
  near-ball or restart tests: those tests are exactly what the rejected first batch
  failed.

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

  Exit codes carry the distinction the loop needs: 0 = did one match, 3 = shard empty,
  ANYTHING ELSE = the process died and the loop must retry with a fresh one. Using 1 for
  "shard empty" was a real bug — it is indistinguishable from a crash, so every shard
  stopped after its first match and the probe reported success with 13 of 39 maps. See
  run_tonight.sh for the driving loop.

Run:  bash run_tonight.sh                                  # the whole thing, end to end
      python3 gen_player_delta.py probe_one [--shard i/n]  # ONE match, then exits
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

PROBE_END = 700        # frames probed per match. Longer than strictly needed to clear
                       # the kickoff: the 23 env creations per match are fixed cost, so a
                       # longer range buys more candidate windows at the same overhead.
SKIP_START = 150       # kickoff / settling — no window may start before this
DOWN = 2               # probe renders at half size; a body is still tens of pixels
MIN_PIX = 40           # changed (half-size) pixels before a player counts as in frame

NEAR_RADIUS = 0.13     # "near the ball" in pitch units (~13% of the 105 m length)
MIN_TRAVEL = 0.10      # the ball must cover at least this much ground across the window
STRIDE = 2             # dense window scan; the data is cached, so this is nearly free

SETPIECE_MODES = (1, 2, 3, 4, 5, 6)      # anything that is not GM_NORMAL

# Base matches. The camera frames a different NUMBER of players depending on where play
# is: in midfield it sees most of both banks of players, while out on a touchline or in a
# corner of the pitch much of the frame is stand and dead ground, so it sees few. Both
# extremes are needed — 16 comes from midfield, 6 only from wide, deep play.
#
# The first three bases are midfield and were probed first; on those eight matches
# end_count=16 filled immediately (28 candidate windows) while end_count=6 had ZERO and
# end_count=10 only four. Hence the wide/goal-line shapes below, borrowed from the
# flank/box shapes in gen2_lib.SHAPES that the situation sweep showed frame far fewer
# bodies. offsides is off for those: an offside award is a set piece, and every window
# containing one is rejected, so leaving it on just burns candidate windows.
# Wide bases get MORE seeds than midfield ones. Measured on the probed maps: a midfield
# match (g2_pd01_s11) produced NOT ONE window with 10 or fewer players on screen across
# its whole 450 frames, while a wide match (g2_pd10_s11) produced end-counts of 4, 6, 8, 9
# and 10. High counts are abundant and low counts are scarce, so the probe budget goes
# where the scarcity is.
MID_SEEDS = [11, 23, 37]
WIDE_SEEDS = [11, 23, 37, 53, 71]
BAL_SEEDS = [11, 23, 37, 53, 71, 89, 101]

# WIDE BUT BALANCED is the shape that actually yields a 6-player frame worth using.
#
# The mismatch shapes below (pd09-pd14, a strong side camped on a weak one) do put play
# in a corner where the camera frames few bodies — but the weak side just hoofs the ball
# out, and a restart inside the window disqualifies it. Measured over 36 probed matches:
# pd11 2.2 restarts/match, pd12 2.0, pd09 1.8, against 0.7 for balanced midfield pd02.
# That is why 27 of the 37 six-player windows found so far die on "restart awarded
# mid-clip" and only one survived.
#
# pd15-pd18 keep the ball wide but make the two sides even, so wide play is contested and
# stays alive instead of being cleared into touch. They get the most seeds because
# end_count=6 is the one target still short.
BASES = [
    # name,      ball,            push(l, r),     difficulty,    offsides, seeds
    ("g2_pd01", (0.00, 0.00), (0.00, 0.00), (0.80, 0.80), True, MID_SEEDS),   # midfield
    ("g2_pd02", (0.20, 0.12), (0.25, 0.15), (0.80, 0.80), True, MID_SEEDS),
    ("g2_pd03", (-0.15, -0.10), (0.15, 0.30), (0.90, 0.70), True, MID_SEEDS),
    ("g2_pd09", (0.75, 0.34), (0.65, 0.40), (1.00, 0.30), False, WIDE_SEEDS),
    ("g2_pd10", (0.85, -0.32), (0.70, 0.45), (1.00, 0.20), False, WIDE_SEEDS),
    ("g2_pd11", (0.90, 0.10), (0.75, 0.45), (1.00, 0.10), False, WIDE_SEEDS),
    ("g2_pd12", (-0.90, -0.10), (0.45, 0.75), (0.10, 1.00), False, WIDE_SEEDS),
    ("g2_pd13", (-0.75, -0.34), (0.40, 0.65), (0.30, 1.00), False, WIDE_SEEDS),
    ("g2_pd14", (0.62, 0.38), (0.55, 0.35), (0.90, 0.40), False, WIDE_SEEDS),
    # wide AND even — contested wide play that stays in play
    ("g2_pd15", (0.70, 0.36), (0.50, 0.45), (0.80, 0.80), True, BAL_SEEDS),
    ("g2_pd16", (-0.70, -0.36), (0.45, 0.50), (0.80, 0.80), True, BAL_SEEDS),
    ("g2_pd17", (0.80, -0.30), (0.55, 0.50), (0.85, 0.85), True, BAL_SEEDS),
    ("g2_pd18", (0.45, 0.38), (0.40, 0.40), (0.80, 0.80), True, BAL_SEEDS),
    # COUNTERATTACK shapes — the one situation where live play genuinely frames very few
    # players. One side is committed far up the pitch (push 0.70-0.80) against a side
    # sitting deep (push 0.00-0.10); when the deep side wins it back, the ball breaks into
    # open ground with a runner and a chaser and nobody else, while the committed team is
    # stranded behind the camera. That is a handful of bodies on screen WITH the ball
    # attended and the play live — unlike a corner, where few players on screen means the
    # ball is about to go out. Balanced difficulty so turnovers actually happen.
    ("g2_pd19", (0.30, 0.05), (0.78, 0.00), (0.80, 0.80), True, BAL_SEEDS),
    ("g2_pd20", (-0.30, -0.05), (0.00, 0.78), (0.80, 0.80), True, BAL_SEEDS),
    ("g2_pd21", (0.10, 0.22), (0.72, 0.05), (0.85, 0.85), True, BAL_SEEDS),
]


def bases():
    for name, ball, (pl, pr), diff, offs, _seeds in BASES:
        yield name, match_spec(name, ball=ball, offsides=offs, difficulty=diff,
                               push_left=pl, push_right=pr)


def jobs():
    for name, _ball, _push, _diff, _offs, seeds in BASES:
        for seed in seeds:
            yield name, seed


def min_near_for(want):
    """How many bodies must be around the ball, given how many are on screen at all.

    Scaled deliberately. With only 6 players in frame, insisting on 3 of them around the
    ball is a bar that open play rarely clears; two — one on the ball and one closing it
    down — is a genuine duel and reads as football. At 10 or more on screen there is no
    excuse for a loose ball, so the bar goes back up to 3.
    """
    return 2 if want <= 6 else 3


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


EXIT_DONE = 3          # this shard has nothing left — the ONLY reason to stop looping


def phase_probe_one():
    """Probe ONE unprobed match from this shard, then exit.

    Exit codes matter here, because the driving shell loop has to tell two very different
    things apart:
        0            did one match, call me again
        EXIT_DONE(3) shard is empty, stop looping
        anything else — including being killed outright — means the process DIED, and the
                     loop must retry rather than stop.

    An earlier version exited 1 for "nothing left", which is indistinguishable from a
    crash. The engine kills the process partway through (see the module docstring on the
    ~46-env leak), so every shard stopped after its first successful match and the probe
    finished with 13 of 39 maps while reporting success.
    """
    shard_i, shard_n = parse_shard(sys.argv[2:])
    _probe_setup(shard_i)
    todo = _shard_todo(shard_i, shard_n)
    if not todo:
        print(f"shard {shard_i}/{shard_n}: nothing left to probe")
        sys.exit(EXIT_DONE)
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
    # Reject a set piece AWARDED inside the window — that is a restart mid-clip, which
    # re-spots the ball and splits the clip into two unrelated passages. A window that
    # merely OPENS while a restart is already pending is fine: it shows play resuming and
    # running on, which is ordinary football. This is the same rule the corner clips use.
    #
    # The stricter "no non-normal mode anywhere" version had to go: low on-screen counts
    # are structurally correlated with stoppages (when the ball goes out near a touchline
    # the camera sits at the pitch edge, so few players are framed AND play has stopped),
    # and it rejected every single end-count-6 window that existed.
    gm = d["game_mode"][s:e + 1]
    if any(gm[i] in SETPIECE_MODES and gm[i] != gm[i - 1] for i in range(1, len(gm))):
        return None

    need = min_near_for(want)
    near_first = near_ball(ball[0][:2], d["left"][s], d["right"][s])
    near_last = near_ball(ball[-1][:2], d["left"][e], d["right"][e])
    if near_first < need or near_last < need:
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
    short = {w: sum(1 for p in picks if p["end_count"] == w) for w in END_COUNTS}
    missing = {w: PER_COUNT - n for w, n in short.items() if n < PER_COUNT}
    if missing:
        # Loud, because this runs unattended: a short count means the batch is incomplete
        # and someone has to widen the probe, not that the run failed.
        print(f"\n  *** SHORT: {missing} — need more probed matches for these counts.")
        print("  *** Low counts come only from wide/goal-line play; add seeds to "
              "WIDE_SEEDS and re-run the probe. Do NOT relax the near-ball or set-piece "
              "tests: those are what the original clips were rejected for.")


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
