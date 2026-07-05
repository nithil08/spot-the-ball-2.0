# Ideas

A running list of "what if we tried..." for this project. No formatting
rules. Date each entry. Newest at the top.

Don't filter. Bad ideas adjacent to good ones are how science actually
works. Write them all down.

The goal by end of summer: 30-50 entries here. Maybe 3-5 we actually
try. Maybe 1 ends up in the paper. That's a great ratio.

---

## Categories to think across (use these as prompts)

- **New manipulations.** Some visual or game-state change we haven't
  done. (Examples: silhouettes only, mirror flip, speed up, score
  4-0 vs 0-0 start, swap player models.)
- **New probe questions.** Something we could ask a human or VLM that
  we currently don't. (Examples: "is this player tired?", "is this
  team about to substitute?", "what would the goalkeeper do if the
  shot came from this angle?")
- **New soccer events.** Specific moments worth generating clips of.
  (Examples: counter-attack from defensive corner, breakaway with one
  defender chasing, set-piece routine, goalkeeper rushing off line.)
- **New ground-truth tools.** Ways to measure what's "correct" beyond
  simulation rollouts. (Examples: ask expert soccer fans, ask
  professional players, label intent rather than outcome.)
- **Connections you notice.** "This thing I read in cognitive science
  reminds me of X in our project."

---

## Example entry (so you see the format)

### 2026-06-15

**Idea:** What if we render the same play *with and without* the
referee on screen? The referee's body language (running toward an
incident, raising a flag) is a strong cue humans pick up. Does it
affect VLM judgments of "was this a foul"?

**Why interesting:** referee = a "meta-agent" whose behavior tells
you what the model should infer about the game state. Removing the
referee = removing a perceptual shortcut.

**What it would take:**
- Find the referee asset in the simulator data dir.
- Make a `no_referee` bundle that scales the referee mesh to 0.
- Generate paired clips for any scenario that has fouls.

**Concerns:** referees are pretty rarely visible in normal play. Might
not be a strong enough signal to study. Tabling but keeping noted.

---

## My ideas

(your entries go below)
