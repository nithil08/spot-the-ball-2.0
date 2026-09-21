# GEN3 — the 24-clip Stanford pilot batch

**BUILT 2026-08-30. REBUILT 2026-09-21** — the whole 24 was re-cut after the slide-cap
bug below; `verify_gen3.py` passes every check, including the new one that catches it.
This file is a record of what was built and how, not a resume point. If you are changing
the batch, read "What was measured" first — several of those findings cost a full engine
run each.

## The slide-cap bug (2026-09-21) — every restart clip opened after its restart

A clip labelled `corner` opened with the ball already bouncing in the six-yard box, the
corner itself never on camera. Measured across the first build: all 5 corners opened
30-41 frames (1.2-1.6 s) after the delivery, 4 of 5 keeper throws 12 frames after the
release, and 2 kick-offs 21-30 frames after the restart — 11 of 15 restart clips.

`_candidates` sets `base = anchor - LEAD[kind]` and then slid the window later by up to
`SLIDE[kind] = 50` frames. A slide can only move the delivery EARLIER in the clip, so
past `LEAD` frames of slide the restart falls out of the front of the window entirely.
The comment on `SLIDE` claimed the opposite ("the delivery stays inside the opening
second") and was never true. Fix: the offset is capped at `anchor - base` as well, so the
last legal start is the delivery frame itself, and `verify_gen3.py` now fails the batch if
any picked restart window has `start > anchor`.

Rebuild cost and consequences, for the next person:

* The corrected windows are a strict SUBSET of the old ones, so every one of them was
  already probed — re-selection needed no engine at all. `_cache/plan.json` was rewritten
  in place from the corrected shortlist, keeping each match's camera offset, because the
  probe is only valid at the offset it ran at.
* Corner matches fell 17 -> 13 (quota is 5) and the 4 that dropped had no legal window
  that also passed `continuous`/`engaged`.
* Corner coherence got worse (mean ~2.1 against ~0.9) and that is correct, not a
  regression: the delivery is a long unowned ball in the air, which is exactly what
  `loose` and `apex` penalise. Coherence ranks windows of the SAME class; it does not
  compare a corner against open play.
* The joint assignment is global, so fixing 5 corners re-cut 20 of the 24 clips. Renders
  are keyed by window, so only the changed ones cost anything.
* The audit rejected three openings the old cut never had to survive — at the arc the
  camera can hold no visible ball for the first frames. The select/render/audit loop
  cleared it in three rounds.
* Everything derived was regenerated: ground truth, scene graphs, classification,
  heatmap, contact sheets and `reproducibility.csv` (24/24 deterministic and
  render-invariant again, 81,861 steps replayed).

## The spec (confirmed with the user, do not reinterpret)

| | |
|---|---|
| 24 clips | 12 levels x 2 |
| Levels | end-frame player count **exactly** 8, 9, 10 ... 19 — one count per level, two clips each |
| "Player" | anyone on the pitch **including goalkeepers**; officials render at 2% and are invisible, so they never count |
| Situations | **5 corner, 5 goalkeeper throw, 5 kick-off, 9 open play**, scattered across the levels |
| Starting team | of the two clips at each count, **one starts with blue (team A) and one with red (team B)** — 12/12 overall |
| Ball | final-frame cell roughly uniform over the 16x6 grid — explicitly NOT the old centre pile-up, but still natural |
| Format | 125 frames @ 25 fps = 5.0 s, 16x6 grid, noname, `full_visibility` + `split_1s_4s` from ONE render pair |
| Quality | the whole point of the batch — physics right, nothing odd, no hidden players |

## What shipped

```
GEN3/clips/full_visibility/clip_01..24.mov     ball visible throughout
GEN3/clips/split_1s_4s/clip_01..24.mov         1 s visible, then 4 s with no ball
GEN3/clips/ground_truth.csv                    one row per situation
GEN3/clips/reproducibility.csv                 scenario sha + play sha per clip
GEN3/clips/scene_graphs/clip_01..24.json       full game state, frame by frame
GEN3/_review/gen3_{corner,gk_throw,kickoff,open}.png   frames 0/62/124 of every clip
```

The two variants of a situation are cut from ONE render pair, so they are the same play
frame for frame and share an identical opening second. `ball_final_cell` is the answer;
`ball_start_cell` is where the red circle sits over the first 8 frames.

`verify_gen3.py`, run against the rendered frames rather than against the plan:

```
counts        8..19, two each, exact
situations    5 corner / 5 gk_throw / 5 kickoff / 9 open, one clip per match
framing       worst clip 81.7% pitch (floor 80%)
vis/inv       largest non-ball difference 96 px (ceiling 400)
ball spread   24 distinct cells of 96, all 6 rows (A4 B4 C3 D5 E5 F3), 15 of 16 columns
```

For scale, GEN1 used 15 of 96 cells and put 20 of its 30 finals in a single row.

Clips are gitignored (`*.mov`), as is `_cache/`. Code, ground truth and `_review/` are
tracked. `_cache/` is ~large and fully regenerable; the sweep alone is 700 matches.

## The pipeline

```bash
cd GEN3
bash run_phase.sh sweep 3        # 700 matches, log only, resumable
python3 gen3.py shortlist        # detect situations + coherence gates
python3 gen3.py plan             # choose matches to probe, assign camera offsets
bash run_phase.sh probe 3        # THE EXPENSIVE ONE — 23 replays per match
bash run_phase.sh ballpix 3      # +1 replay per match, exact ball pixel at each end frame

# selection and the audit loop — repeat until audit is clean (it took two rounds)
python3 gen3.py select           # min-cost-flow joint assignment -> 24 picks
bash run_phase.sh render_vis 3 && bash run_phase.sh render_inv 3
python3 gen3.py audit            # exit 4 = windows rejected, go back to select

python3 gen3.py compose          # clips + ground truth
python3 verify_gen3.py           # spec checks + contact sheets
python3 verify_gen3.py recheck 3 # re-probe 3 clips end to end
```

`run_phase.sh` exists because the engine leaks: a process dies at roughly its 46th
`FootballEnv` with no traceback. Every phase does bounded work and exits; the loop
re-invokes it. Exit codes are load-bearing — `0` = did work, `3` = shard empty,
**anything else = the process died and must be retried**. Using 1 for "nothing left" was
a real bug once: indistinguishable from a crash, so every shard stopped after one match
while reporting success. `audit` uses 4 for "rejections recorded" for the same reason.

Three shards, not six: these are CPU-bound and each holds an OpenGL context, so shard
COUNT is the real throttle and `nice` is only a second line of defence.

## What was measured (do NOT re-derive)

1. **`GM_KICKOFF` never fires.** 0 awards across 120 cached GEN2 matches (330,000 frames),
   while goal kicks and free kicks fire constantly. Kick-offs exist only as *behaviour*:
   the ball respotted on the centre mark with both sides behind halfway. That happens at
   frame 0 and again after every goal (measured: a 1.017 single-frame ball jump, then all
   22 players back in their own halves). `find_kickoffs` detects the behaviour and
   validates it geometrically.

   Corollary: only shapes whose **formation** is a genuine kick-off shape (push = 0) can
   produce kick-offs. A shape pushed 0.55 up the pitch restarts with five players already
   in the opponent's half — measured — and is correctly rejected. Hence `mid_bal`,
   `mid_even`, `mid_even2`.

2. **Corners are the binding constraint: ~0.03 per match.** The 700-match sweep yielded 17
   probeable corner matches against a quota of 5, so the risk flagged at the pause did not
   materialise. This is the price of fixing the "bouncy / lacks logic" rejection — v2
   manufactured set pieces with difficulty gaps of 0.70-0.95 and offsides OFF, and GEN3
   refuses to do that (gap <= 0.40, offsides ON, the GEN1 setting). If a future batch runs
   short of corners the answer is to sweep more matches, never to cripple a side.
   Goalkeeper throws run ~0.09/match, open play ~47/match, kick-offs ~2/match.

3. **Rendering is free; stepping is the whole cost.** `env.render("rgb_array")` ~0.1 ms
   against `env.step` ~8.3 ms. So probing ONE frame costs the same as probing every frame,
   which is what makes exact counts affordable *at the camera offset that ships*. One
   sweep match (2750 frames) = ~39 s wall.

4. **The camera offset changes which players are in frame.** So the count and the ball
   cell are not independent knobs, and a count measured at offset 0 is worthless for a
   clip shipping at offset -12. The probe MUST run at the shipping offset. Since the probe
   yields every frame's count for one price, the offset is a per-MATCH parameter — free,
   because source diversity already limits us to one clip per match.

5. Ball pixels come from a **plate pair**: with all 22 players hidden, a visible-ball plate
   against an invisible-ball plate differs only by the ball — no shirts or socks to confuse
   it. Costs one extra replay on top of the probe's 23. Ground truth is still re-measured
   from the real shipped render in `compose`.

6. **The offset can push the ball clean out of the picture.** ~26.6 px of shot per unit of
   offset X, so at +-24 a centred ball lands at x ~= 0 or ~= 1280 and leaves the frame.
   22 of 1149 candidate end frames had no ball at all, and one match (`wideR_s332`) lost
   all 11 of its. That is a property of the window, not a fault — such a window has no
   answer cell — so `ballpix` drops those end frames rather than failing.

7. **A plate cannot see occlusion.** Everything before `audit` measures the ball on a plate
   with all 22 players hidden, so it is blind to the ball being behind a defender in the
   shipped render. That is what `audit` is for, and it caught a final frame down to 2
   changed pixels. Its ground truth was still correct — render-measured (631.6, 169.0)
   against plate-measured (634.0, 166.2), the same cell — but 2 px is not a ball a viewer
   can read, so the window was rejected.

8. **The spread reweighting has to accumulate.** Recomputing the penalty from the last
   solution alone made it oscillate between two extremes rather than converge: a selection
   in rows B/C/D priced those rows out, the next fled wholesale to A/E/F, and back again.
   Measured: `{B:7, C:8, D:9}` against `{A:6, B:7, E:9, F:2}`. Accumulated multipliers with
   a diminishing step give `{A:3, B:5, C:5, D:4, E:4, F:3}` for 0.047 of mean coherence.
   Collapsing each (match, count) pair to its most coherent window before building the flow
   made this worse still — it hides a match's row-E window behind its row-C one.

9. **Every clip is exactly reproducible, and the render knobs cannot touch the play.**
   Measured over all 24 clips, 81,180 replayed steps (`verify_reproducible.py`). Each clip
   was replayed to its own end frame three times: twice canonically, once at a different
   camera offset with two players hidden. All three logs — ball xyz, all 22 player
   positions, possession, game mode, score, every step — are bit-identical, exact equality
   and not a tolerance. So:

       (scenario source, game_engine_random_seed, physics_steps_per_frame) -> the play

   and `GFOOTBALL_CAM_OFFSET_X/_Y`, `GFOOTBALL_HIDE_SLOTS` and the asset bundle are
   provably render-only. This extends the scenario docstring's determinism claim from the
   120 steps it measured to the full 2450 a clip actually needs.

   `clips/reproducibility.csv` records, per clip, the sha256 of the emitted scenario source
   AND the sha256 of the play. `python3 verify_reproducible.py check` re-derives both in
   one replay per clip — run it after any engine rebuild or edit to `SHAPES`, because a
   changed scenario is otherwise completely invisible.

10. **Counterfactuals must branch at frame 0. This is an engine limit, not a choice.**
    Three routes were measured (`counterfactual_probe.py`):

    * **Roster removal works.** An 11 v 10 scenario builds and runs, the observation
      shrinks to `right_team (n, 10, 2)`, and the counterfactual replays bit-identically.
      It diverges from the factual at step 8, so the clip's own window is a *different
      match*, not "the same clip minus a player".
    * **Rebuilding a scenario from logged positions is far too lossy to restart mid-play.**
      `AddPlayer` takes only (x, y, role) and `SetBallPosition` only (x, y) — no velocity,
      no ball height, no possession, no facing. Restarting clip_09 at frame 2310 from its
      own logged state: ball separation 0.19 world units after 1 s, 0.37 after 2 s (the
      pitch is 2.0 long), and possession agrees with the factual on only 22.4% of frames.
      The logged ball z of 0.317 is simply discarded.
    * **`get_state`/`set_state` restore exactly — but only into an identical env.** The
      C++ serialiser returns a ~100 KB blob carrying the whole simulation; restoring it
      into a fresh env and stepping on reproduces the factual continuation bit-identically
      for all 125 frames. It is also **50x faster** than replaying the prefix (0.5 s
      against 25.4 s to reach frame 2310), which is a real speed-up for any future phase
      that needs many windows from one match.

      It cannot carry an intervention. Both a changed roster (11 v 10) and a merely changed
      env config (adding one controlled agent, same scenario file) are rejected with
      `FATAL ERROR in [football::set_state]: Current environment scenario != scenario in
      the state` — and it is a **fatal abort, signal 11, not a Python exception**, so it
      cannot be caught and probed for. Test it in a throwaway process or it takes the run
      down with it.

    Consequence for the study design: a counterfactual pair is two matches from the same
    seed that differ in their scenario, compared from kick-off. There is no way to hold the
    first 2310 frames fixed and then remove a player.

11. **The starting team is balanced per LEVEL, inside the flow, not by filtering.**
    The first build let possession fall where it liked and six of the twelve levels came
    out with both clips starting on the same side (8, 11, 12, 15, 18, 19 — four of the
    five kick-offs were red). Balancing it afterwards is not possible: swapping one clip
    breaks the count, the situation quota or the one-clip-per-match rule.

    So the constraint went into `gen3_assign` itself. The pair key became
    `(match, count, starting team)` and each count's sink was split into two of capacity
    one, so a level with two same-side starts is not an expensive flow, it is not a
    feasible one. The team belongs on the WINDOW and not the match: one match can supply
    a blue-start window at count 12 and a red-start one at count 14.

    "Starting team" is `gen3_lib.start_possessor`, in three cases, because it is not one
    lookup: whoever owns the ball on the start frame; else the last team to own it (the
    ball is often loose mid-pass); else — only reachable in the opening frames of a match,
    where the kick-off clips live — the first team to take it during the clip.

    **The frame index is log[start], not log[start - 1].** The first cut of
    `start_possessor` read one frame early and mislabelled clip_23, which silently broke
    the level-19 balance. Both the sweep and the render loop observe AFTER stepping and
    the render loop keeps a frame once its index reaches `start`, so loop index i carries
    log[i]: the clip's first frame is log[start] and its last is log[end - 1] — the index
    `_pool` already uses for the end count. The error is invisible on most frames and wrong
    exactly on a possession change at the cut, so check labels against frame 0 of the
    scene graph, never by eye.

    Cost of the rebuild: 14 of 24 windows changed over two selections (11, then 3 more
    after the index fix), the rest reused straight from the render cache. Nothing else
    moved — 5/5/5/9 quota exact, counts exact, 24 distinct matches, 24 distinct cells
    across all six rows, framing floor 81.7%, and mean coherence 0.734 -> 0.743.

12. **The world-space scene graph is free; the pixel-space one is not.**
    `scene_graph.py` exports every clip's full game state per frame — all 22 player
    positions and velocities, ball xyz, possession, game mode, score, roles, kits, nearest
    player to the ball, team centroids — as JSON, in seconds and with no engine run at all.
    It is a slice of the cached sweep log, so it is exact rather than inferred. 24 files,
    ~400 KB each, 9.4 MB for the batch. `--hide-ball-after 1.0` withholds the ball at the
    same cut `split_1s_4s` uses, which is what a ball-prediction eval needs.

    What it cannot contain is player PIXEL positions, and therefore which players are in
    shot. The camera tracks the ball and the repo's only calibration maps the BALL between
    world and pixel space (`solve_offsets`), not arbitrary points. Per-frame per-player
    pixel data needs the 23-pass hide-and-diff probe widened from end frames to all frames:
    ~23 replays per clip, so roughly half an hour across three shards for the batch.
    Rendering is free and stepping is the cost, so that price buys every frame at once.

## Also done

**Per-team goalkeeper kits.** The user asked for keeper kits matching their own team
instead of one shared yellow, "so it is easy to identify instead of hard".

This needed an engine change, because `team.cpp:108` hard-coded ONE global
`goalie_kit.png` for both keepers — no amount of texture editing can differentiate them.
The patch adds `GFOOTBALL_TEAM_GK_KITS`; when set, the keeper's kit comes from his own
team's `kit_url` (`<kit_url>_gk_kit.png`). It is **opt-in on purpose**: `IMG_LoadBmp`
dereferences the decoded surface with no null check, so requesting a texture a bundle does
not carry is a segfault, not an error. `libgame.dylib` rebuilt; backup at
`libgame.dylib.backup_pre_teamgk`. Patch regenerated into `engine_patches/`.

`experiments/apply_gen3_kits.py` builds `gen3` / `gen3_ball_invisible`. Measured by
hide-and-diff (not eyeballed):

| | outfield | keeper |
|---|---|---|
| team A | RGB (72, 94, 250) | (134, 188, 241) pale sky blue |
| team B | RGB (253, 57, 59) | (230, 137, 134) pale coral |

Same hue family as the team, far lighter than the outfield kit. Hue carries team identity,
lightness carries keeper-vs-outfield.

## Still open

* **Officials are invisible** (2% render scale), inherited from the existing benchmark.
  That is deliberate — a visible referee would be a person in frame and would corrupt the
  player-count ground truth — but it means the clips are not a faithful broadcast image in
  that one respect. Worth raising with the user if realism is ever questioned.
* The contact sheets were reviewed and read as genuine football: corners with the box
  crowded, kick-offs with both sides in their own halves and the ball on the centre mark,
  keeper distributions starting at the keeper. A closer look at the moving clips is still
  worth doing before the pilot ships.
