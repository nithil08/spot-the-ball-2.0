# GEN4 — 24 clips levelled by the OPENING headcount

Started 23 Sep 2026. GEN3_HARD's football, turned around: the level is the number of
people in shot on the FIRST frame, not the last.

Read `../GEN3/RESUME.md` and `../GEN3_HARD/RESUME.md` first. The engine plumbing, the
bundle rule, the leak workaround, the camera-offset reasoning and the locked 16x6 noname
format are all inherited unchanged, and the **cached sweep logs are reused** — GEN4
replays no match to build its plan.

## The spec

| | |
|---|---|
| 24 clips | 12 levels x 2 |
| Level | **people in shot on the FIRST frame**, exactly 8, 9, 10 ... 19, two clips each |
| Situation mix | **not commanded** — reported (see below) |
| Ball | must be in the camera's view on EVERY frame; final-frame cell spread over the 16x6 grid |
| Format | 125 frames @ 25 fps = 5.0 s, 16x6 grid, noname, `full_visibility` + `split_1s_4s` from ONE render pair |
| Football | identical to GEN3_HARD — same 14 shapes, both sides at difficulty 0.95, offsides ON, seeds 300-349 |

## The four things GEN4 does differently, and why each one exists

**1. The level is the first frame.** Asked for directly: "two plays where it starts with
8 people, and then 9, and then 10". The final count is free and is recorded alongside.

**2. No situation quota.** It cannot be held at 5/5/5/9 on top of the level spec. Measured
on GEN3_HARD's own clips: a corner opens with 3-4 people in shot because the camera has to
frame the corner arc, a kick-off opens with 11-13 because both elevens are at the halfway
line, a keeper clip with 10-14. Only open play spans 8 to 19. Forcing both makes most
levels unreachable, so the level wins and `verify_gen4.py` prints the mix that fell out.

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
bash run_phase.sh probe 3          # 23 replays/match, counts on a 5-frame grid  << the cost
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

Unlike GEN3's probe, this one keeps a **5-frame grid across the whole replay** rather than
a dozen end frames. Replay is the entire cost and kept frames are free, so one probe now
answers both ends of every window the match could ever supply — which is what makes a
level on the opening frame affordable at all.

## Status

* 23 Sep 2026 — shortlist (600 matches) and plan (60 matches, 12 shapes) done; probe
  running across 3 shards.
