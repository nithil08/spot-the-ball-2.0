# GEN3_HARD — the 24-clip pilot batch, both sides at hard

Same spec as `../GEN3`, same pipeline, same code. **One thing changed: every shape runs
`left_difficulty = right_difficulty = 0.95`.** That is gfootball's own
`11_vs_11_hard_stochastic` value and what `DIFFICULTY_CLIPS` labels "hard". The point of
the batch is the highest-caliber football the engine will play, with the two sides evenly
matched at that level.

Read `../GEN3/RESUME.md` first. Everything it records about the pipeline, the camera
offset, the counting probe, the flow assignment and the engine's limits still holds — this
file only records what is DIFFERENT and what the difficulty change actually cost.

## What is identical to GEN3

| | |
|---|---|
| 24 clips | 12 levels x 2 |
| Levels | end-frame player count **exactly** 8, 9, 10 ... 19, two clips each |
| Situations | 5 corner, 5 goalkeeper throw, 5 kick-off, 9 open play |
| Starting team | one blue-start and one red-start clip per level, 12/12 overall |
| Ball | final-frame cell spread over the 16x6 grid |
| Format | 125 frames @ 25 fps = 5.0 s, 16x6 grid, noname, `full_visibility` + `split_1s_4s` from ONE render pair |
| Shapes | all 14, same ball positions, same push values, offsides ON |
| Seeds | 300..349 (extended only if the shortlist runs short) |
| Bundles | `gen3` / `gen3_ball_invisible` — kits, GK kits and invisible officials unchanged |

## What changed

```
gen3_lib.SHAPES   every difficulty pair -> HARD = (0.95, 0.95)
scenario names    g3_<shape>  ->  g3h_<shape>
```

The rename matters. GEN3's `clips/reproducibility.csv` records the sha256 of each emitted
scenario source, and `write_scenario` writes into the shared
`~/gfootball_src/gfootball/scenarios/` directory. Writing hard-difficulty scenarios under
the old `g3_` names would have silently invalidated every GEN3 clip. The two batches now
own disjoint scenario files and disjoint caches, and GEN3 is untouched.

## The clips are NOT the same plays

Difficulty is a scenario parameter, so it changes the match from frame 0. Two matches on
the same shape and the same seed at different difficulties diverge within a few steps and
have nothing in common after that. So this batch re-derives everything — the sweep, the
situations, the windows, the camera offsets, the counts — and produces 24 clips that meet
the same spec with different football in them. There is no way to hold the play fixed and
change how good the teams are; see note 10 in `../GEN3/RESUME.md` for why even a mid-play
branch is impossible.

## The risk this batch carries

In GEN3 the four final-third shapes ran asymmetric difficulty — `flankL`, `flankL2`,
`boxL`, `boxR` at (1.00, 0.60) — and that asymmetry is what pinned play in the final third
and produced corners and keeper throws. Symmetric hard removes the lever. The remaining
drivers of where play happens are the ball's starting position and the two push values,
both unchanged.

Corners were already the binding constraint in GEN3 at ~0.03 per match (17 probeable in
700 matches against a quota of 5). If hard-v-hard defends better and that rate drops, the
fix is **more seeds, never a crippled side** — `SEEDS` in `gen3.py`, then re-run `sweep`
(it skips matches already cached) and `shortlist`. `plan` prints an explicit warning when
a class is short of quota.

## The pipeline

Identical to GEN3. From this directory:

```bash
bash run_phase.sh sweep 3        # 700 matches, log only, resumable
python3 gen3.py shortlist
python3 gen3.py plan             # WARNS if a situation class is short
bash run_phase.sh probe 3        # the expensive one — 23 replays per match
bash run_phase.sh ballpix 3

python3 gen3.py select
bash run_phase.sh render_vis 3 && bash run_phase.sh render_inv 3
python3 gen3.py audit            # exit 4 = windows rejected, go back to select

python3 gen3.py compose
python3 verify_gen3.py
python3 verify_gen3.py recheck 3
python3 scene_graph.py           # optional, free
```

Exit codes are load-bearing: `0` = did work, `3` = shard empty, anything else = the
process died and must be retried. `run_phase.sh` handles the retry.

## Status log

* **2026-09-17** — directory created, difficulty set to (0.95, 0.95), scenarios renamed
  `g3h_*`.
* **2026-09-17** — sweep complete, 700/700 matches. Shortlist: **29 corner**, 60 gk_throw,
  123 kickoff, 488 open — every class over quota, no extra seeds needed. Hard-v-hard does
  NOT suppress corners: 0.030/match, identical to GEN3's rate, and 29 probeable matches
  against GEN3's 17. So the asymmetric (1.00, 0.60) shapes were never what produced
  corners — the ball start position and the push values were.
* **2026-09-17** — `plan`: 89 matches to probe (29/18/16/26), no shortfall warning.
* **2026-09-17 17:34** — probe paused at 76/89 by request; resumes from cache.
  `run_overnight.sh` added: drives probe -> ballpix -> select/render/audit loop ->
  compose -> verify -> scene graphs unattended, holds the machine awake with `caffeinate`,
  and unloads its own scheduler when it finishes. Scheduled for 22:00 by the LaunchAgent
  `~/Library/LaunchAgents/com.nithil.gen3hard.plist`.

  The script exports PATH explicitly. launchd hands a job
  `/usr/bin:/bin:/usr/sbin:/sbin`, which does not contain this machine's `python3`
  (`/Library/Frameworks/Python.framework/Versions/3.14/bin`) — without the export the
  whole run dies instantly on "command not found" and looks like it never fired.

* **2026-09-17 22:00 — THE LAUNCHAGENT DOES NOT WORK. DO NOT RETRY IT.**
  The job fired on schedule and died immediately with exit 126:

  ```
  shell-init: error retrieving current directory: getcwd: cannot access parent
              directories: Operation not permitted
  /bin/bash: .../GEN3_HARD/run_overnight.sh: Operation not permitted
  ```

  This is macOS TCC, not a bug in the script and not a PATH problem. The repo lives under
  `~/Desktop`, which is a protected folder, and a process launched by launchd has no
  TCC privileges of its own — so it cannot even `stat` the script, let alone run it.
  `cron` and `at` fail the same way for the same reason. The only fixes are to grant
  `/bin/bash` Full Disk Access in System Settings (broad and interactive) or to move the
  whole repo out of `~/Desktop`.

  **What works instead:** launch `run_overnight.sh` with `nohup` from a terminal that
  already holds Desktop access (Terminal, iTerm, the Claude Code session). TCC
  responsibility is inherited from the spawning app, so the detached run keeps the access
  and survives the parent shell exiting. For a timed start, `resume_at.sh` sleeps to a
  wall-clock time inside that same already-privileged process.

  Agent unloaded; `com.nithil.gen3hard.plist` left on disk as a record.

* **2026-09-17 23:15** — overnight run relaunched by hand from the privileged session,
  resuming the probe at 76/89.

* **2026-09-18 00:16 — `run_overnight.sh` swallowed the audit's exit code.** Its log
  filter (`python3 ... | grep -v`) sat between the stage and its status, and grep exits 0
  whenever it printed anything, so `audit`'s exit 4 arrived as 0. The audit had correctly
  rejected three windows and deleted their renders; the script logged "audit clean" and
  ran compose against six missing files, writing nothing. Fixed with
  `return "${PIPESTATUS[0]}"` in the `py()` helper, plus hard stops on select, compose and
  verify, plus an explicit "are all 48 renders present" check before the audit.
  **Exit codes are how this pipeline reports — never put a filter between a stage and its
  status.**

* **2026-09-18 00:32 — BUILT. `verify_gen3.py` reports ALL CHECKS PASSED.**

  ```
  counts        8..19, two each, exact
  situations    5 corner / 5 gk_throw / 5 kickoff / 9 open, one clip per match
  start team    12 blue / 12 red, one of each at every count
  framing       worst clip 91.9% pitch (floor 80%)
  vis/inv       largest non-ball difference 95 px (ceiling 400)
  ball spread   23 distinct cells of 96, all 6 rows, 14 of 16 columns
  ball present  smallest final-frame ball 15 px
  ```

  **Mean coherence 0.858, against GEN3's 0.743.** Hard-v-hard play scores materially more
  coherent, which is the result the batch was built to get.

  The audit loop took **three** rounds (GEN3 took two), and it did not converge
  monotonically: 21/24 clean, then 23/24, then 22/24, then clean. Every single rejection
  was "no ball for the first N frames" — the ball outside the shot early in the window,
  never occlusion. Hard-v-hard football travels further and faster, so at the off-centre
  camera offsets that produce the spread ball cells the ball leaves the framed area more
  often than it did in GEN3. Budget more audit rounds for any future hard batch; the loop
  is cheap, because renders are keyed by window and only replacements re-render.

  Counts 18 and 19 were the tight ones at selection (only 9 and 2 candidate matches).
  Hard defences keep fewer bodies in frame at the top end, so a future batch wanting
  counts above 19 should expect to sweep more seeds.
