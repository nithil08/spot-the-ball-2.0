# GEN3 — the 24-clip Stanford pilot batch: state, and how to resume

Paused 2026-08-26 partway through the match sweep, at the user's request (session limit).
Nothing was half-written: the goalkeeper work is finished and committed, the pipeline is
written and syntax-clean but has only ever run its first phase.

**Read this before re-deriving anything.** The expensive findings are in
"What was measured", below, and several of them cost a full engine run each.

## The spec (confirmed with the user, do not reinterpret)

| | |
|---|---|
| 24 clips | 12 levels x 2 |
| Levels | end-frame player count **exactly** 8, 9, 10 ... 19 — one count per level, two clips each |
| "Player" | anyone on the pitch **including goalkeepers**; officials render at 2% and are invisible, so they never count |
| Situations | **5 corner, 5 goalkeeper throw, 5 kick-off, 9 open play**, scattered across the levels (the user raised this from "a few" to 15 special) |
| Ball | final-frame cell roughly uniform over the 16x6 grid — explicitly NOT the old centre pile-up, but still natural |
| Format | unchanged: 125 frames @ 25 fps = 5.0 s, 16x6 grid, noname, `full_visibility` + `split_1s_4s` from ONE render pair |
| Quality | the whole point of the batch — physics right, nothing odd, no hidden players |

## Done and committed

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

## Written, never run past phase 1

`gen3_lib.py` (plumbing, shapes, detectors, coherence), `gen3.py` (all phases),
`gen3_assign.py` (min-cost-flow selection), `verify_gen3.py`, `run_phase.sh`.
All four compile; only `sweep` has executed.

**79 of 700 sweep matches are cached** in `_cache/sweep/` (gitignored, regenerable). The
sweep is resumable — existing files are skipped.

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

2. **Corners are the binding constraint: ~0.03 per match.** 4 detected in 120 matches,
   3 of them clearing coherence. This is the price of fixing the "bouncy / lacks logic"
   rejection — v2 manufactured set pieces with difficulty gaps of 0.70-0.95 and offsides
   OFF, and GEN3 refuses to do that (gap <= 0.40, offsides ON, the GEN1 setting). The
   answer is to sweep more matches, never to cripple a side. 700 matches should yield
   ~20 corners; 5 are needed. **This is the riskiest part of the spec.**
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

## Resume

```bash
cd GEN3
bash run_phase.sh sweep 8        # finishes the 700-match sweep (~1.5 h from 79 done)
python3 gen3.py shortlist        # detect situations + coherence gates
python3 gen3.py plan             # choose matches to probe, assign camera offsets
bash run_phase.sh probe 6        # THE EXPENSIVE ONE — 23 replays per match
bash run_phase.sh ballpix 6      # +1 replay per match, exact ball pixel
python3 gen3.py select           # min-cost-flow joint assignment -> 24 picks
bash run_phase.sh render_vis 4 && bash run_phase.sh render_inv 4
python3 gen3.py compose
python3 verify_gen3.py           # spec checks + contact sheets
python3 verify_gen3.py recheck 3 # re-probe 3 clips end to end
```

`run_phase.sh` exists because the engine leaks: a process dies at roughly its 46th
`FootballEnv` with no traceback. Every phase does bounded work and exits; the loop
re-invokes it. Exit codes are load-bearing — `0` = did work, `3` = shard empty,
**anything else = the process died and must be retried**. Using 1 for "nothing left" was a
real bug once: indistinguishable from a crash, so every shard stopped after one match
while reporting success.

## If `select` reports INFEASIBLE

It prints which class or which count could not be filled, rather than silently shipping
21 clips. Expected failure modes, in order of likelihood:

* **corners short** — sweep more seeds (widen `SEEDS` in `gen3.py`). Do not relax the
  coherence gate and do not widen the difficulty gap; that is exactly what v2 was
  rejected for.
* **count 8 or 9 short** — low counts come only from wide/deep/counterattack play, never
  from midfield. Raise `PROBE_BUDGET["open"]` and lean on the `countL`/`countR` shapes,
  or probe a few matches at a second, larger camera offset (a bigger offset frames fewer
  players).
* **count 19 short** — the opposite: midfield shapes at small offsets.

## Still open / not started

* The batch has never been visually reviewed. `verify_gen3.py` writes contact sheets to
  `_review/` for exactly that, and the framing audit (pitch fraction >= 0.80) is
  automated, but a human should still look.
* **Officials remain invisible** (2% render scale), inherited from the existing benchmark.
  That is deliberate — a visible referee would be a person in frame and would corrupt the
  player-count ground truth — but it was never re-confirmed for this batch, and it is the
  one respect in which the clips are not a faithful broadcast image. Worth flagging to the
  user if realism is questioned.
