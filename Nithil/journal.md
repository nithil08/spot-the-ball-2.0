# Journal

Write one entry every day you work. 5-10 minutes, not more. Be honest —
"I got stuck and gave up after an hour" is a valid entry; "I had a
productive day :)" is not. Skip days you don't work on the project;
**don't backfill**, it defeats the point.

Bring this journal to every weekly sync with Neha.

---

## Daily template

Copy this block at the top of the file for each new day.

```
## YYYY-MM-DD

**What I worked on:**
-

**What was confusing or hard:**
-

**What I learned (technical, conceptual, or about doing research):**
-

**Question I want to ask Neha:**
-

**Tomorrow I'll start with:**
-
```

---

## Example entry (what a good one looks like)

## 2026-06-08

**What I worked on:**
- Followed SETUP.md to install python 3.14 and the venv.
- Got the smoke test to print "wrote 2 description items".
- Opened one of the frames — it looks like a top-down soccer pitch
  with 2 players. Pretty cool that this came from a sim, not a video.

**What was confusing or hard:**
- The `absl` package errored at first ("no attribute FLAGS"). I think
  pip got confused. Deleting the venv and recreating it fixed it. Not
  sure why this worked — copy-pasted the same install command.
- The huge wall of "SDL" warnings looks scary. SETUP.md says ignore
  them, which feels weird.

**What I learned:**
- venvs are isolated python installs. If you don't activate the venv
  (`source .venv/bin/activate`), pip installs go to the wrong place.
- The simulator renders at 1280x720 and outputs .avi files, which then
  get converted to .mov.

**Question I want to ask Neha:**
- Why python 3.14 specifically? Most stuff I see online uses 3.11.
- Why does Apple print so many warnings? Is something actually broken?

**Tomorrow I'll start with:**
- Watch the rest of the existing clips in `experiments/clips/` and
  notice the differences between bundles.
- Start the 3Blue1Brown neural network video on the train.

---

## Entries

(your daily entries go below; newest at the top)
