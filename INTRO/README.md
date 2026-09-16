# INTRO — two 20-second clips for the experiment intro

**BUILT 2026-09-16.** Same locked format as the GEN3 pilot, four times as long, and
chosen to be ordinary rather than difficult. Their job is to show a viewer what the
material looks like — kits, grid, camera, the pace of the play — before the graded
5 s trials begin.

```
clips/full_visibility/intro_01.mov   20.0 s, ball visible throughout   <- the intro clip
clips/full_visibility/intro_02.mov
clips/split_1s_19s/intro_01.mov      1 s visible, then 19 s with no ball
clips/split_1s_19s/intro_02.mov
clips/ground_truth.csv               source, start team, start + final ball cell
_review/intro_0*_sheet.png           frames 0/100/200/300/400/499
```

Each pair is cut from ONE render pair, so the visible and hidden variants are the same
play frame for frame. The hidden variant is there in case the intro wants to demonstrate
the task itself; the full-visibility clip is the intro.

## What they are

| | intro_01 | intro_02 |
|---|---|---|
| source | `flankR_s308`, frames 1380-1880 | `boxL_s301`, frames 340-840 |
| length | 500 frames @ 25 fps = 20.0 s | same |
| starts with | blue (team A) in possession | red (team B) in possession |
| ball | C9 -> C10 | C8 -> C10 |
| pitch in shot | 93.9% | 97.1% (floor is 80%) |
| camera offset | 0, 0 | 0, 0 |

Both are unbroken passages of open play: no goal, no restart awarded inside the window,
no engine re-spot, score unchanged end to end. intro_01 builds from midfield into the
attacking third; intro_02 comes back out of one box, through midfield, into the other.

Locked format throughout — 16x6 grid burned in, noname bundles (`gen3` /
`gen3_ball_invisible`), per-team keeper kits, officials at 2% render scale, red circle on
the ball's start cell for the first 8 frames, 1280x480 after the HUD crop.

## Why this is not GEN3 with a bigger number

A graded clip is chosen for its END STATE: an exact player count and a target ball cell,
which is what the 23-pass count probe and the camera-offset solve are for. An intro clip
has no answer to hit, so there is no probe and no offset here — the camera stays at 0,
which is also the widest, cleanest framing the engine gives (hence 94-97% pitch against
GEN3's 81.7% floor).

The one thing that had to be added is a **watchability gate**. Ranking 20 s windows by
coherence alone puts a spell of one team holding the ball in its own half at the very
top — calm, owned, low, close to a player, and it teaches a viewer nothing. So `pick`
also requires possession to change hands at least 3 times, the ball to cover at least a
quarter of the pitch's length, and the ball never to hug a touchline (where the stands
come into shot). `mid_even2_s339` was the top-ranked window and is exactly what that
gate exists to reject: 500 frames, zero possession changes.

## The funnel

Over the 700 cached GEN3 sweep matches, minus the 24 that supply graded clips (an intro
clip must not preview a test item):

```
301,496  20 s windows (stride 5, never opening before frame 25)
232,251  continuous — no goal, no teleport, no mid-window re-spot
100,598  no restart awarded inside the window
 73,073  inside the coherence bands
 34,741  watchable (possession changes, traverse, off the touchline)  -> 549 matches
```

Continuity is the expensive constraint at this length, as expected: a third of all 20 s
windows have a restart awarded part-way through, which at 5 s is rare. There is still no
shortage — 549 matches can supply an intro clip, so a different pair is a re-`pick`
away, not a re-sweep.

## Regenerating

```bash
cd INTRO
python3 intro_clips.py scan          # ~40 s, pure cache read, no engine
python3 intro_clips.py pick          # the two picks -> _cache/picks.json
python3 intro_clips.py render_vis    # ~40 s
python3 intro_clips.py render_inv    # ~40 s, separate process: the bundle is read at import
python3 intro_clips.py compose       # ~20 s -> clips + ground_truth.csv
```

`render_vis` and `render_inv` must be separate processes — the asset bundle is chosen by
an env var read when gfootball is imported, so one process cannot do both passes.

The two raw passes land in `_cache/raw/` as uint8 `.npy`, 884 MB each, because compose
needs the visible and invisible pass of the same clip at once and four of those in a
Python list is how this falls over. 3.4 GB total, gitignored and fully regenerable:
`rm -rf INTRO/_cache/raw` once the clips are written.
