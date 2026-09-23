# GEN4 — 24 clips, two ending with each headcount 8..19

Started 23 Sep 2026. GEN3_HARD's spec, rebuilt so the levels actually hold: two plays
ENDING with 8 people in shot, two ending with 9, up to 19.

The level axis is a setting (`gen4_lib.LEVEL_AT`, "end" or "start"), because the probe
measures both ends of every candidate window at no extra cost. Switching it is a re-run of
`windows` and `select` — a couple of minutes — and never of the probe.

Read `../GEN3/RESUME.md` and `../GEN3_HARD/RESUME.md` first. The engine plumbing, the
bundle rule, the leak workaround, the camera-offset reasoning and the locked 16x6 noname
format are all inherited unchanged, and the **cached sweep logs are reused** — GEN4
replays no match to build its plan.

## The spec

| | |
|---|---|
| 24 clips | 12 levels x 2 |
| Level | **people in shot on the LAST frame**, exactly 8, 9, 10 ... 19, two clips each (`LEVEL_AT`) |
| Situation mix | **not commanded** — reported (see below) |
| Ball | must be in the camera's view on EVERY frame; final-frame cell spread over the 16x6 grid |
| Format | 125 frames @ 25 fps = 5.0 s, 16x6 grid, noname, `full_visibility` + `split_1s_4s` from ONE render pair |
| Football | identical to GEN3_HARD — same 14 shapes, both sides at difficulty 0.95, offsides ON, seeds 300-349 |

## The four things GEN4 does differently, and why each one exists

**1. The levels actually hold.** GEN3_HARD claimed 8..19 twice on the closing frame and
did not deliver it — re-measured, level 15 was empty, 16 held four clips and one clip sat
at 7, because the counts it selected on were wrong. Same spec here, with a count rule that
survives measurement. Both ends are published; the level is the closing one.

**2. No situation quota.** GEN3_HARD held 5/5/5/9 and could do so because its pool was
planned around it; here the level is the binding constraint and the mix is left free, so
the search is never forced to spend a scarce corner on a level open play could have
covered. `verify_gen4.py` prints the mix that fell out. (If the quota matters more than
the levels, that is a different batch and worth saying so before it is built.)

**3. The ball must be in the camera's view on every frame.** GEN3_HARD checked the first
and last frames only and exempted the middle, on the grounds that a ball behind a defender
measures as absent and that is just football. True, but not the only case: its clip_01
shipped with the ball **outside the shot for 65 consecutive frames — 2.6 s** — because the
camera offset that composes the shot off-centre also pushes a ball rolling towards the
touchline off the edge of it. Confirmed by re-rendering that window with all 22 players
hidden: still no ball, so nothing was occluding it.

`audit` now checks every frame and separates the two cases. A frame with no ball in the
shipped pair triggers a PLATE pair for that window (all players hidden, both bundles): if
the ball is visible there it was occlusion and is accepted; if not, the window is rejected
and the out-of-view span is recorded so re-selection avoids it. The plate pair costs two
replays and is only paid on clips that have a gap.

**4. A match is a play, not a name.** `mid_bal`, `mid_even` and `mid_even2` are three names
for ONE scenario spec — same ball, offsides, difficulty and pushes — so one seed replays
bit-identically under each. GEN3_HARD shipped the same kick-off as three clips (06, 15,
19) because its one-clip-per-match rule compared names. `canonical_match` collapses them
before anything else runs: 700 cached sweeps become 600 distinct matches.

## The counting rule, stated once

A person is in shot if at least `MIN_BODY` (20 at full resolution, 6 at down=2) changed
pixels of their **body** are inside the frame. A player outside the frame can still cast a
shadow into it; that is not a person you can see and it is not counted.

GEN3_HARD's probe did neither — it ran at half resolution with a 40-pixel floor (~160 at
full size) and made no body/shadow distinction — and **nine of its 24 published counts
were wrong**, six short by a body clipped by the frame edge and two counting shadows. See
`../GEN3_HARD/_review/CLIP_REVIEW.md`.

## Pipeline

```
python3 gen4.py shortlist          # 600 distinct matches (free, reads GEN3_HARD's sweep)
python3 gen4.py plan               # 60 matches, one camera offset each  [--more N]
bash run_phase.sh probe 3          # 23 replays/match, counts on a 4-frame grid  << the cost
python3 gen4.py windows            # legal windows joined to their counts (free)
bash run_phase.sh ballpix_vis 3    # plate frames at candidate end frames
bash run_phase.sh ballpix_inv 3    # ball pixel, from the plate pair
python3 gen4.py select             # the joint assignment -> 24 picks
bash run_phase.sh render_vis 3 ; bash run_phase.sh render_inv 3
python3 gen4.py audit              # exit 4 = rejections, loop back to select
bash run_phase.sh audit_plate_vis 1 ; bash run_phase.sh audit_plate_inv 1   # only if exit 5
python3 gen4.py compose
python3 verify_gen4.py
```

**The probe is the whole cost** and it is why `plan` takes 60 matches rather than all 600:
23 replays each, ~6 minutes, and everything downstream is cheap. `select` reports which
levels are short; only then is `plan --more N` worth paying for.

Unlike GEN3's probe, this one keeps a **4-frame grid across the whole replay** rather than
a dozen end frames. Replay is the entire cost and kept frames are free, so one probe
answers BOTH ends of every window the match could ever supply — which is what makes the
level axis a setting rather than a rebuild.

The grid is 4 and not 5, and the number is load-bearing: a window is 125 frames, so its
last frame sits 124 after its first. 124 is divisible by 4 and not by 5, so on a 5-grid
every window would have its opening frame measured and its closing frame missed.

## Status

* 23 Sep 2026 — shortlist (600 matches) and plan (60 matches, 12 shapes) done; probe
  running across 3 shards. Level axis set to the CLOSING count.
