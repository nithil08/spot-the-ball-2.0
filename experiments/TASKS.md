# Social-Cognition Probe Tasks for VLMs and Humans

A taxonomy of five probe categories. Each is a different *cognitive demand* on
top of the same underlying soccer simulator. We hold the visual world as
constant as possible across categories so any human–VLM gap reflects the
probe type, not stimulus differences.

Clip length: **10 s** (100 simulator steps at 10 fps).
Resolution: 1280×720.
HUD and player-name overlays: off.
Rendering pipeline: bot-vs-bot rollouts of `gfootball` scenarios, with
asset-swap bundles for occlusion / uniformity variants.

---

## 1. Description  *(calibration tier)*

> *"How many players are in this scene?"*

**Stimulus:** single static frame.
**Manipulation:** vary scene density: 1v1, 3v1, 5v5, 11v11.
**Question variants:**
- How many players are on the **left** team?
- How many players are on the **right** team?
- How many players are in the scene **in total**?

**Answer format:** multiple choice (1 / 2 / 3 / 5 / 7 / 11).
**Ground truth:** dump output reports `n_left` and `n_right` exactly.
**Purpose:** sanity check. Humans and competent VLMs should both be at
ceiling. Items where a VLM fails *here* invalidate any downstream finding
about that model.

---

## 2. Prediction

> *"Where will the ball be in 3 seconds? Will the team score?"*

**Stimulus:** 5-second or 7-second clip from a 10-second scenario.
Clip ends with the visible portion frozen for 1 second.
**Manipulation:** cutoff time (short vs long) — produces a difficulty
curve. Scenario type (3v1, counterattack, pass-and-shoot).
**Question variants:**
- Where will the ball be at the end of the 10 s window?
- Will the attacking team score within the remaining time? (yes/no)
- Which team will have possession at the end?

**Answer format:** 6×4 grid bin for ball location, or click coordinate;
yes/no for scoring; left/right/neither for possession.
**Ground truth:** continue the deterministic rollout to step 100. Ball
xy at step 100 is the answer; final score is the answer for the yes/no.
**Purpose:** measures forward simulation. A model that handles description
but fails here lacks dynamic mental models.

---

## 3. Inference

> *"Reason about what you can't see."*

Three sub-tasks per item (same underlying clip, three rendering variants):

### 3a. Ball hidden
**Stimulus:** 10 s clip rendered with the `ball_tiny` asset bundle —
ball mesh scaled to 5 % so it's invisible at this resolution.
**Question:** where is the ball at the freeze frame?
**Answer format:** click coordinate or 6×4 grid bin.
**Ground truth:** `obs['ball']` from the dump.

### 3b. Player hidden
**Stimulus:** 10 s clip with one player painted over with a green disc
(post-render OpenCV; the engine has no per-player hide).
**Question:** where is the hidden player?
**Answer format:** click coordinate.
**Ground truth:** that player's `(x, y)` from the dump.

### 3c. Region occluded
**Stimulus:** 10 s clip with a horizontal band of pixels blacked out.
**Question:** how many players are inside the black band right now?
**Answer format:** integer (0–11).
**Ground truth:** count players whose pixel-projected y is inside the
band, computed per frame from the dump.

**Purpose:** object permanence and partial observability — the core
extension of "spot the ball" to multi-agent scenes.

---

## 4. Hypothetical *(intervention)*

> *"What would help this team score?"*

**Stimulus:** static tableau (frame ~1.5 s into the play) of a team
approaching the opponent's goal. Four candidate-intervention tableaus
shown side-by-side, labeled A / B / C / D:

- **A.** Add an attacker near the box.
- **B.** Remove a defender.
- **C.** Move the ball 0.15 forward.
- **D.** No change (control).

**Question:** which intervention most increases the team's chance of
scoring?
**Answer format:** multiple choice (A / B / C / D).
**Ground truth:** run each intervention scenario through the rollout
harness (`rollout.py`) for *N* independent seeds and compute empirical
P(goal). The intervention with the highest P(goal) is the correct
answer. ΔP across interventions is the per-item difficulty.
**Purpose:** counterfactual reasoning about *unobserved* configurations.
Requires the agent to mentally simulate plays it never witnessed.

---

## 5. Counterfactual *(responsibility)*

> *"Who is responsible for the goal? What would have prevented it?"*

**Stimulus:** 10 s clip ending in a goal.
**Question variants:**
- Which player was *most* responsible for the goal?
- How responsible was each player? (1–7 Likert per player)
- What single change would have prevented the goal? (MC over candidate
  changes, ranked by ablation-derived ΔP(goal))

**Answer format:** per-player Likert + MC over candidate
counterfactuals.
**Ground truth:** for each player, re-run the scenario with that player
made `lazy` (attackers) or removed entirely (defenders) for *N* seeds.
ΔP(goal) under each ablation = that player's but-for causal contribution.
This produces a *graded* responsibility ranking — not just a binary
right/wrong answer — which lets us correlate human and VLM responsibility
judgments with the mechanical ground truth.
**Purpose:** causal attribution and social-responsibility ascription.
This is the hardest probe and likely the largest human–VLM gap.

---

## Cross-category design properties

- **Same simulator, same camera, same render pipeline.** Differences in
  scores across categories cannot be blamed on stimulus appearance.
- **Paired counterfactuals.** Every item ships with its mechanically-defined
  variants (perturbed positions / removed players / different cutoffs) so
  reviewers can verify ground truth and we can ablate confounds.
- **Mechanical ground truth.** Every category has an objective answer
  derived from simulation, not from human consensus. (For the responsibility
  task we still collect human consensus, but the model is also scored against
  the mechanical answer.)
- **Same scene, multiple probes.** A subset of clips appears in three or
  four categories simultaneously, enabling within-scene analysis of which
  cognitive demand a model fails on.

## Data scale (target)

| Category       | Items | Raters/item | Total ratings |
|----------------|------:|------------:|--------------:|
| Description    |    80 |          50 |         4,000 |
| Prediction     |   100 |          50 |         5,000 |
| Inference      |   120 |          50 |         6,000 |
| Hypothetical   |    80 |          50 |         4,000 |
| Counterfactual |    80 |          30 |         2,400 |
| **Total**      | **460** |           | **21,400**    |

VLM evaluation: 5 models × 460 items × 5 stochastic reps ≈ **11,500 calls**.

## Generation pipeline

| Script | Role |
|---|---|
| `build_bundles.py` | Create asset-swap data dirs (ball_tiny, uniform_jerseys, …) |
| `scenario_factory.py` | Emit parametrized `.py` scenario modules |
| `rollout.py` | Run a scenario N times → empirical P(score) |
| `lib.py` | Env builder, dump reader, pitch↔pixel mapping |
| `gen_description.py` | Static frames at varied densities |
| `gen_prediction.py` | 10 s clips + cutoff variants |
| `gen_inference.py` | Ball-hidden, player-hidden, region-occluded clips |
| `gen_hypothetical.py` | Tableau + 4 intervention variants + rollout GT |
| `gen_counterfactual.py` | Scoring clip + per-player ablation GT |

Each generator writes `stimuli/<category>/<item_id>/{stimulus, meta.json}`
plus an `_index.json` summary.
