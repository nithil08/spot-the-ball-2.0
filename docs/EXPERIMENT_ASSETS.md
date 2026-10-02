# What already exists that experiments can be built on

Written 2 Oct 2026. An inventory of the whole workspace (`~/Desktop/Nithil Research`, the
`nithil08/spot-the-ball-2.0` repo, and the `nithil08/spot-the-ball-video` repo) read with one
question in mind: **if we run a model experiment next week, what do we not have to build?**

Nothing here is a plan. It is a list of reusable parts, each with what it is, where it is, and
what is wrong with it.

---

## 1. Prompts — five that have been run, and what each is for

Every prompt in the project is one of five. They are worth treating as a library, because the
grid legend and the `Reasoning: / Ball position:` contract are what make the answers parseable,
and three separate runners re-implement the same parser.

**P1 — the paper prompt** (`code/spot-the-ball-video/src/lib.py:36`). No grid legend. Used for
the 3s/5s/8s visibility-split study. Ships with its own flaw: it says the ball "has been
digitally removed from every frame", which is false for split clips where it is visible at the
start.

```
You are watching a short clip from a soccer match. The ball has been digitally removed
from every frame. Infer where the ball is located in the final frame of the clip.
Respond in exactly this format:
Reasoning: <one or two sentences>
Ball position: <grid row #, grid column#>
```

**P2 — the same prompt with the grid spelled out** (`experiments/nithil_work/run_spot_ball_adc.py:39`).
This is the one behind the only headline number we have (10% exact, 16x6). It prepends:

```
A yellow grid is overlaid on the video: its ROWS are labelled A-F from top to bottom
(6 rows) and its COLUMNS are labelled 1-16 from left to right (16 columns). A cell is
named <row-letter><column-number>, e.g. D5 is row D, column 5.
```

**P3 — the split-aware prompt** (`experiments/nithil_work/gemini_spot_ball.py:75`, `Q_SPLIT`).
Describes the stimulus honestly — ball visible ~1 s with a red circle, then invisible — and asks
for 2-3 sentences of reasoning rather than one. Generates its grid legend from `(cols, rows)`
so it works for any grid. **This is the one to standardise on.**

**P4 — the perception baseline** (same file, `Q_VISIBLE`). Ball visible throughout, name the
final-frame cell. This is the control that tells you whether a miss is bad inference or bad
perception, and it is the single most valuable prompt in the set because every batch now ships a
`full_visibility` twin of every clip.

**P5 — locate-now + predict-next** (`gen_ball_hidden_grid_video.py:204`, `gen_ball_hidden_4s.py:83`).
Two questions on one stimulus: where is the ball *now*, and where will it be *one second after
the clip ends*. Only ever run on the early 32x12 still-frame items. The extrapolation half has
never been scored against anything, and GEN3's scene graphs now make it scorable for free.

**The five-category question set** (`experiments/gen_{description,prediction,inference,hypothetical,counterfactual}.py`)
is a separate, older family — counting players, "will they score in 5 s", "click the hidden
player", "which intervention most increases their chance of scoring", "which player was most
responsible for the goal" (1-7 Likert). The stimuli behind them are the superseded 32x12
material, but **the question wording and the MC/Likert answer formats are reusable as written**,
and the player-count questions now have exact, pixel-measured ground truth they did not have
when they were written.

### What is missing from the prompt library

* No chain-of-thought / no-CoT pair. PLAN.md Week 9 asked for it; it was never run.
* No prompt that asks for confidence, or for a ranked top-3 of cells. Both would turn a 10%
  exact-match into a distribution you can actually analyse.
* No prompt that asks "how many people are in shot on the final frame" against the counts we
  measured — the hardest ground truth in the project is currently unused by any eval.
* No prompt variant study at all (grid legend vs none, letter-number vs row/col, reasoning
  before answer vs after).

---

## 2. Runners and scoring — three implementations of one thing

| | `run_spot_ball_adc.py` | `gemini_spot_ball.py` | `spot-the-ball-video/src/` |
|---|---|---|---|
| Models | Gemini via Vertex ADC, falls back to AI Studio key | Gemini Files API | Gemini, GPT, LLaMA, Qwen |
| Video | inline bytes (~1 MB clips) | upload, poll ACTIVE, delete | upload (Gemini) / 8 frames (rest) |
| Batches | one, hard-coded | registry, two batches, pluggable GT loader | 3 variants x 30 clips |
| Output | JSONL + markdown table | JSONL + markdown table | CSV, resumable |
| Scoring | exact / within-1 / mean cell distance | same | none — `parse_results.py` only aggregates |

**Keep from each:** the batch registry and pluggable `gt_loader` from `gemini_spot_ball.py`; the
tolerant `parse_cell` from `run_spot_ball_adc.py` (it handles `D5`, `D, 5`, numeric rows, "row D
column 5" — the plain regex in the other two produced *3 unparsed answers out of 30* on the
first visible-ball run); the resume-from-CSV loop and the multi-provider structure from
`spot-the-ball-video`.

`vlm_eval.py` (452 lines) is the only multi-provider harness with real grading: Anthropic,
OpenAI, OpenRouter, Together and local Ollama behind one `call_model`, plus per-category graders.
Its model registry is stale (Qwen2.5-VL, LLaVA-7B, MiniCPM-V) but the provider plumbing is not.

**What has actually been run:** Gemini 2.5 Pro on 30 clips — **3/30 exact (10%), 9/30 within one
cell, mean miss 2.46 cells, 0 unparsed**. Qwen / LLaVA / MiniCPM were run on the old five-category
stimuli in June and never scored into a table. **No model has ever been run on GEN2, GEN3,
GEN3_HARD, INTRO or the DRILLS.** That is the largest single gap in the project.

---

## 3. Stimuli on disk, by how usable they are

| Batch | Clips | Format | Ground truth | Verdict |
|---|---|---|---|---|
| **GEN3** `GEN3/clips` | 24 x 2 | 125f @ 25fps, 1280x480, 16x6, noname | `ground_truth.csv`, `clip_classification.csv`, **24 scene-graph JSONs**, `reproducibility.csv` | **best thing we have** |
| **GEN3_HARD** | 24 x 2 | identical, both sides at difficulty 0.95 | same + `clip_renumbering.csv`, `passes.csv`, `clip_review.csv` | ready; read §4 on passing |
| **GEN2** | 61 x 2 | 50f @ 10fps, 16x6 | `SITUATIONS_GROUND_TRUTH.csv` | headers/corners/GK-throws done, `04_player_delta` never built |
| **30_1s_visible_4s_invisible** | 30 | 50f @ 10fps, 16x6 | `ground_truth.csv` + per-frame headcounts for all 1,500 frames | the one batch with a model score |
| **INTRO** | 2 x 2 | 500f @ 25fps = 20 s | `ground_truth.csv` | task instructions / practice trials |
| **DRILLS** | 70 | 8 set-piece and small-sided drills | `ALL_GROUND_TRUTH.csv` | hand-built scenarios, not sampled match play |
| **DIFFICULTY_CLIPS** | 9 | 5 s, no grid, no GT | none | all 9 skill pairings, seed 42, same window — a stimulus-validity figure |
| **GOALS** | 3 of 621 | 5 s | `goals.csv` | **no shipped clip contains a goal** — `continuous()` rejects them |
| **GEN4** | 0 | — | `plan.json`, `shortlist.json` | planned only; probe never run |

**The paired-variant design is the asset.** Every GEN2/GEN3/GEN3_HARD/INTRO clip exists twice —
`full_visibility` and `split_1s_4s` — cut from **one render pair of the same deterministic
state**. The two are frame-for-frame the same play, pixel-identical for the first second. That
makes P4-vs-P3 a within-stimulus control, not a comparison of two similar clips. No other
benchmark of this kind can do that, and it is free.

---

## 4. Ground truth that is stronger than it looks

* **Ball position is measured, never labelled.** The pixels that differ between the visible and
  invisible render *are* the ball. Sub-pixel accurate, and it generalises: the same hide-and-diff
  trick applied per player slot gives the headcounts.
* **GEN3 scene graphs** (`GEN3/clips/scene_graphs/clip_NN.json`, schema `gen3-scene-graph/1`) —
  per-frame game state for all 125 frames of all 24 clips: every player's world position, the
  ball, possession, kits with measured RGB, pitch geometry, plus `scenario_sha` and `play_sha`.
  Read the `notes` block before using it: coordinates are **world units, not pixels**, player
  pixel positions are *not* derivable (the camera tracks the ball), and who is *in shot* is a
  camera property measured separately.
* **`clip_classification.csv`** — a written English description of every clip plus ~30 numeric
  columns. These are ready-made regressors for a difficulty analysis: `ball_travel_m`,
  `ball_net_m`, `ball_apex_m`, `airborne_fraction`, `possession_changes`,
  `players_near_ball_end`, `crowding_end`, `restart_seconds_into_clip`, zone start/end.
* **Bit-exact reproducibility.** `reproduce_clip.py` rebuilds byte-identical files from
  `(shape, seed, start_frame, end_frame, camera_offset)`. Any clip can be re-rendered under a
  new manipulation and stay the same play.
* **Headcounts.** GEN3_HARD is balanced to exactly 8,9,...,19 people in shot on the final frame,
  two clips each, with a 12/12 blue/red start split. Measured by rendering the final frame 23
  times (all hidden, then one player at a time) and diffing bodies, not shadows, at full
  resolution. Half resolution got **9 of 24 wrong**; colour segmentation does not work at all
  (the hoardings are saturated blue and red).

**Two limitations to design around, both measured rather than assumed:**

1. **GEN3_HARD cannot test passing.** Over the four hidden seconds of all 24 clips there are 4
   passes and 8 turnovers total, and in **16 of 24 the ball never leaves the player holding it**.
   A variable that is zero in two-thirds of the sample cannot be regressed. Difficulty there is
   carried by distance travelled while hidden (rank correlation +0.68 against a constant-velocity
   guess) and speed at disappearance (+0.64).
2. **The centre bias is camera geometry, not scenario design.** The camera tracks the ball, so
   the ball is near frame centre by construction and the final-frame cells cluster in rows C/D.
   `GFOOTBALL_CAM_OFFSET_X/_Y` was added to break it. This bounds how much any grid-cell accuracy
   number means — a model answering "D8" every time is not at chance.

---

## 5. Manipulation knobs — what the engine can already do

Ten asset bundles in `experiments/asset_bundles/`: `default`, `noname`, `ball_tiny`,
`ball_transparent`, `uniform_jerseys`, `gen2`, `gen3`, and `*_ball_invisible` twins. Plus a
patched engine (`engine_patches/gfootball_engine.patch`) exposing:

| Env var | What it does | Experiment it enables |
|---|---|---|
| `GFOOTBALL_HIDE_SLOTS` | render a player at 2% scale while he **keeps playing** | hide-and-diff measurement; "infer the invisible player" |
| `GFOOTBALL_TEAM_GK_KITS` | per-team keeper kits instead of one global texture | goalkeeper identifiability (opt-in: a missing texture segfaults) |
| `GFOOTBALL_CAM_OFFSET_X/_Y` | shift the composed shot | break the centre bias; off-centre framing |
| `GFOOTBALL_FONT` | blank font = no player name labels | the locked noname format |

The ball-invisibility trick is a 5% vertex scale on `generic.ase`, not a pipeline edit — the ball
is still physically there and still in the log, which is why ground truth survives.

**Unbuilt manipulations that are a bundle away:** silhouettes (solid colour per team — strips body
language without merging teams), `no_logos`, `night_mode`, `ball_huge`, mirror flip, playback
speed. `ideas.md` lists these and is otherwise empty — the brainstorm PLAN.md Week 2 asked for
never happened.

**`GFOOTBALL_HIDE_SLOTS` is not a counterfactual knob**, and `GEN3/counterfactual_probe.py`
exists to say so: it changes rendering only, and the observation log stays bit-identical. A real
counterfactual needs the player removed from the *scenario*, which means divergence starts at
frame 0 — the clip's own window becomes a different match. Mid-play snapshot restart is lossy by
construction (the scenario builder takes only position and role: no velocity, no ball height, no
possession, no facing, no animation phase). Read that file before promising anyone a
counterfactual experiment.

---

## 6. Figures and write-up material

* `SPOT_THE_BALL_PRESENTATION/` — the full 4-part package: hidden clips, visible clips, ground
  truth, and **Gemini's reasoning transcribed clip by clip** next to the right answer. The
  qualitative error analysis PLAN.md Week 9 asked for is half-written inside it.
* `_review/` in each GEN batch — ball-final heatmaps, player-count staircases, pass charts,
  per-situation contact sheets, annotated final frames.
* `30_1s_visible_4s_invisible/people_in_frame*.png` — histogram, time course, per-clip heatmap,
  and a sanity image with every counted player boxed.
* `docs/ONBOARDING.{html,pdf}`, `RESEARCH_LOG.md` (26 KB), `LitReview.md`, `READING.md`,
  `AIModels.md` — the last is a current-as-of-writing ranking of candidate models (Gemini 3 Pro
  leading on video, GPT-5.5, Claude Opus 4.x, Qwen3-VL, InternVL3, VideoLLaMA 3,
  LLaVA-OneVision, GLM-4.6V, and Qwen3.5-Omni if audio ever matters).
* The published paper is **arXiv:2511.00261**, Balamurugan et al., "Spot The Ball".

---

## 7. The gaps, ranked by what they cost to close

1. **Run a model on GEN3 / GEN3_HARD.** 96 clips, two prompts (P3 and P4), ground truth and
   scoring already written. Nothing needs building — `gemini_spot_ball.py` needs one new entry in
   its `BATCHES` registry.
2. **Run the `full_visibility` control.** It converts every miss into "perception or inference?"
   and it is the same clips.
3. **One runner instead of three.** Merge the tolerant parser, the batch registry and the
   multi-provider layer. Half a day, and it stops the next batch forking a fourth copy.
4. **More than one model.** Every number in the project is Gemini 2.5 Pro. `vlm_eval.py` already
   speaks to five providers.
5. **Score the headcount question.** Hardest-won ground truth in the repo, never asked of a model.
6. **Prompt variants** — CoT, confidence, top-3, grid-legend ablation. Cheap, and they make a
   10% exact-match interpretable.
7. **Finish GEN4** (the probe is the whole cost, ~6 min/match) **or** build `04_player_delta` for
   GEN2. Both are paused mid-build with their plans on disk.
