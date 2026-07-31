# Spot The Ball — Research Log

**Project:** Visual social inference benchmark using Google Research Football simulator  
**Paper:** Balamurugan et al., "Spot The Ball: A Benchmark for Visual Social Inference" (arXiv:2511.00261)  
**Repo root:** `spot-the-ball-2.0/`

Auto-updated by `experiments/nithil_work/log_entry.py`. Do not edit the Activity Log section by hand — add entries via that module.

---

## Models Researched

| Short name | Full model ID | Type | Categories tested | Notes |
|---|---|---|---|---|
| claude | claude-sonnet-4-6 | Cloud API (Anthropic) | description, prediction, inference, counterfactual, hypothetical | All runs were DRY RUN (no real API calls) |
| qwen (large) | qwen/qwen2.5-vl-72b-instruct | Cloud API (OpenRouter) | description, prediction, inference, counterfactual, hypothetical | All runs were DRY RUN |
| qwen (small) | qwen2.5vl:7b | Local (Ollama) | description, prediction | Live runs |
| llava | llava:7b | Local (Ollama) | description, prediction | Live runs |
| minicpm | minicpm-v | Local (Ollama) | description, prediction | Live runs |
| qwen (7B alt) | Qwen2.5-VL-7B | Local | description | Single run, format unclear |

### Probe categories

| Category | Task description | Stimulus type |
|---|---|---|
| description | Count players on each team | Single static frame |
| prediction | Where will ball be / will team score after clip ends? | 5s or 7s clip (cut from 10s full) |
| inference | Locate hidden ball / hidden player / players in occluded region | Single frame (various occlusions) |
| counterfactual | Which player most responsible for goal? What would have prevented it? | Single frame of goal moment |
| hypothetical | Which intervention most increases scoring chance? | 5 frames (base + 4 interventions) |

---

## Clip Batches

### Batch 1 — `experiments/stimuli/description/` (Static frames, player-count task)

| Item ID | Scenario | Seed | Frame # | Players L/R | Stimulus |
|---|---|---|---|---|---|
| 2d7619893e | custom_1v1_base | 11 | 10 | 2 / 2 | frame.png |
| e56cb1b4e7 | custom_1v1_base | 11 | 30 | 2 / 2 | frame.png |
| 41bab790b0 | custom_1v1_base | 11 | 50 | 2 / 2 | frame.png |
| b86d55e033 | custom_1v1_base | 11 | 70 | 2 / 2 | frame.png |

- Ball: visible | Grid: none | Duration: single frame | Resolution: 1280×720

---

### Batch 2 — `experiments/stimuli/prediction/` (Prediction clips)

| Item ID | Scenario | Seed | Clip length | Cutoff | Full video |
|---|---|---|---|---|---|
| 4af7c88981 | academy_3_vs_1_with_keeper | 3 | 10.0s | 5s | full.mov |
| b02eb15669 | academy_3_vs_1_with_keeper | 3 | 10.0s | 7s | full.mov |
| 772d409dfc | academy_3_vs_1_with_keeper | 11 | 10.0s | 5s | full.mov |
| 99b4d2a334 | academy_3_vs_1_with_keeper | 11 | 10.0s | 7s | full.mov |

- Ball: visible | Grid: none | HUD: off | Clips shown to model: cutoff duration | FPS: 10

---

### Batch 3 — `experiments/stimuli/inference/` (Occlusion inference)

| Item ID | Stimuli | Notes |
|---|---|---|
| 0e84874467 | ball_hidden.mov, player_hidden.mov, region_occluded.mov | 3 versions of same scenario |
| a49c77afbe | ball_hidden.mov, player_hidden.mov, region_occluded.mov | 3 versions of same scenario |

- Ball: hidden (ball_tiny bundle) in ball_hidden.mov | Grid: none | HUD: off

---

### Batch 4 — `experiments/stimuli/counterfactual/` (Goal responsibility)

| Item ID | Stimulus |
|---|---|
| 187086cb44 | clip.mov |
| e4a39b19be | clip.mov |

- Questions: player most responsible, what would have prevented goal, Likert responsibility ratings

---

### Batch 5 — `experiments/stimuli/hypothetical/` (Intervention frames)

| Item ID | Seed | Frame | Best intervention | Stimuli |
|---|---|---|---|---|
| d08a4879e1 | 7 | 15 | D (no change) | frame_base.png + 4 intervention frames |
| 90b9895fe9 | 13 | 15 | D (no change) | frame_base.png + 4 intervention frames |

---

### Batch 6 — `experiments/nithil_work/results/ball_hidden_4s/` (Ball-hidden single frame, early grid)

| Item ID | Scenario | Seed | Frame at | Grid | Models run |
|---|---|---|---|---|---|
| e9d90747ad | academy_counterattack_easy | 23 | 4s | 6×4 (text-described, not drawn) | qwen, llava, minicpm |

- Ball: hidden (ball_tiny bundle) | Grid: described in prompt only (not burned on image) | Stimulus: frame_4s.png + frame_4s_grid_annotated.png
- Note: earlier approach — grid was text-described, not visually burned. Superseded by Batch 7.

---

### Batch 7 — `experiments/nithil_work/results/ball_hidden_grid/` (Ball-hidden video, grid burned in)

| Item ID | Scenario | Seed | Clip shown | Predict at | Grid | Ball cell at freeze | Ball cell at predict |
|---|---|---|---|---|---|---|---|
| 08fcceebae | academy_3_vs_1_with_keeper | 23 | 4.0s | 5.0s | 32×12 yellow, burned on every frame | H28 | G26 |
| 9066812d07 | academy_counterattack_easy | 331 | 4.0s | 5.0s | 32×12 yellow, burned on every frame | E22 | E23 |
| 666bb5de48 | academy_counterattack_hard | 3 | 4.0s | 5.0s | 32×12 yellow, burned on every frame | E24 | I25 |

- Ball: hidden (ball_tiny bundle) in clip.mov | Ball: visible + red marker in debug_full.mov (not shown to model)
- Grid: 32 cols × 12 rows, 40×40px cells, yellow lines alpha=130, alphanumeric labels (cols 1-32, rows A-L)
- Resolution: 1280×480 (HUD cropped) | FPS: 10

---

### Batch 8 — `experiments/20sCLIPS/` (20-second plain clips, no grid)

| File | Duration | Notes |
|---|---|---|
| clip_1.mov | 20.0s | Gameplay, no grid, ball visible |
| clip_2.mov | 20.0s | Gameplay, no grid, ball visible |
| clip_3.mov | 20.0s | Gameplay, no grid, ball visible |
| clip_4.mov | 20.0s | Gameplay, no grid, ball visible |

- Ball: visible | Grid: none | HUD: off | Scenarios/seeds: not recorded

---

### Batch 9 — `experiments/nithil_work/results/plain_grid_clips/` (20-second clips, ball visible, grid burned in) — **Generated 2026-07-02**

| File | Scenario | Seed | Duration | Ball | Grid | Size |
|---|---|---|---|---|---|---|
| clip_1.mov | academy_3_vs_1_with_keeper | 23 | 20.0s | visible | 32×12 yellow, burned on every frame | 2.2 MB |
| clip_2.mov | academy_counterattack_easy | 331 | 20.0s | visible | 32×12 yellow, burned on every frame | 1.9 MB |
| clip_3.mov | academy_counterattack_hard | 3 | 20.0s | visible | 32×12 yellow, burned on every frame | 2.7 MB |
| clip_4.mov | academy_run_pass_and_shoot_with_keeper | 3 | 20.0s | visible | 32×12 yellow, burned on every frame | 2.9 MB |

- Asset bundle: default (full visuals) | FPS: 10 | Steps: 200 | Resolution: 1280×480 (HUD cropped)
- Grid spec: 32 cols × 12 rows, 40×40px cells, yellow (255,255,0) alpha=130, labels cols 1-32 left→right, rows A-L top→bottom
- Episodes that ended early (goal/OOB) auto-reset and continue to fill the full 20s
- Generated by: `experiments/nithil_work/gen_plain_grid_clips.py`

---

### Batch 10 — `experiments/nithil_work/results/5s_noname_clips/` (5-second clips, ball visible, grid burned in, NO player names) — **Generated 2026-07-02**

| File | Scenario | Seed | Duration | Ball | Names | Grid | Size |
|---|---|---|---|---|---|---|---|
| clip_1.mov | academy_3_vs_1_with_keeper | 23 | 5.0s | visible | OFF | 32×12 yellow, burned on every frame | 0.5 MB |
| clip_2.mov | academy_counterattack_easy | 331 | 5.0s | visible | OFF | 32×12 yellow, burned on every frame | 0.6 MB |
| clip_3.mov | academy_counterattack_hard | 3 | 5.0s | visible | OFF | 32×12 yellow, burned on every frame | 0.6 MB |
| clip_4.mov | academy_run_pass_and_shoot_with_keeper | 3 | 5.0s | visible | OFF | 32×12 yellow, burned on every frame | 0.5 MB |

- Names OFF: achieved by setting zero controllable player slots (C++ AI drives all players)
- Title burned on every frame: scenario name, duration, ball status, grid spec
- Generated by: `experiments/nithil_work/gen_5s_noname_clips.py`

---

### Batch 11 — `experiments/nithil_work/results/visibility_variants/` (5-second clips, graduated ball visibility) — **Generated 2026-07-02**

Based on Batch 10 clip_1: `academy_3_vs_1_with_keeper`, seed=23, identical physics across all variants.  
Each variant splices N seconds from the **default bundle** (ball visible) with (5-N) seconds from the **ball_tiny bundle** (ball hidden). Same seed = same player/ball positions, just the ball rendering differs.

| File | Visible portion | Hidden portion | Total | Size |
|---|---|---|---|---|
| 0s_vis_5s_hid.mov | 0 seconds | 5 seconds (full clip) | 5s | 0.4 MB |
| 1s_vis_4s_hid.mov | 1 second | 4 seconds | 5s | 0.4 MB |
| 2s_vis_3s_hid.mov | 2 seconds | 3 seconds | 5s | 0.4 MB |
| 3s_vis_2s_hid.mov | 3 seconds | 2 seconds | 5s | 0.4 MB |
| 4s_vis_1s_hid.mov | 4 seconds | 1 second | 5s | 0.4 MB |

- Grid: 32×12 yellow burned on every frame | Names: OFF | HUD: OFF
- Title burned on every frame: e.g. "VISIBLE: 1s → HIDDEN: 4s  |  academy_3_vs_1_with_keeper  |  seed=23"
- Purpose: test how much visible context a model needs to locate the hidden ball
- Generated by: `experiments/nithil_work/gen_visibility_variants.py`

---

## What We Have Done — Plain English Summary

> **Who this is for:** anyone picking up this project for the first time, or returning after a break.

### The big picture

We are building a research benchmark to test whether AI vision models (and humans) can do "social thinking" — reasoning about where an unseen ball is, predicting what will happen next, and attributing responsibility for goals — using short video clips of a simulated soccer game (Google Research Football simulator, a top-down 2D view).

The simulator lets us control everything: which players are visible, whether the ball is shown, what grid overlay is drawn on the image. This gives us ground truth we can never get from real footage.

---

### What has been generated (all clips are `.mov` H.264 videos)

**Location key:** everything lives inside `spot-the-ball-2.0/experiments/`

#### Group A — Benchmark stimuli (the actual test questions)

| What | Where | # items | Duration per clip | Key feature |
|---|---|---|---|---|
| Player-count frames | `stimuli/description/` | 4 PNG frames | Single frame | How many players on each team? |
| Ball-location prediction clips | `stimuli/prediction/` | 4 clips + 4 full videos | 10s (shows 5s or 7s, asks about rest) | Will the team score / where is ball going? |
| Occlusion inference clips | `stimuli/inference/` | 6 clips (2 scenarios × 3 occlusion types) | Short | Ball hidden, player hidden, region blocked |
| Goal-responsibility clips | `stimuli/counterfactual/` | 2 clips | Short | Who caused the goal? |
| Intervention frames | `stimuli/hypothetical/` | 2 items × 5 frames | Single frames | Which change would help team score? |

#### Group B — Nithil's working experiments

| Batch | What | Where | # clips | Duration | Ball | Grid | Names |
|---|---|---|---|---|---|---|---|
| 6 | Single hidden-ball frame (old method, grid text only) | `nithil_work/results/ball_hidden_4s/` | 1 PNG | 1 frame | Hidden | Text description only | N/A |
| 7 | Hidden-ball video clips with grid burned in | `nithil_work/results/ball_hidden_grid/` | 3 clips | 4s visible | Hidden | 32×12 yellow drawn on video | On (2 players) |
| 8 | 20-second plain gameplay clips, no grid | `20sCLIPS/` | 4 clips | 20s | Visible | None | On |
| 9 | 20-second gameplay clips with grid | `nithil_work/results/plain_grid_clips/` | 4 clips | 20s | Visible | 32×12 yellow | On |
| 10 | 5-second gameplay clips with grid, **no names** | `nithil_work/results/5s_noname_clips/` | 4 clips | 5s | Visible | 32×12 yellow | **OFF** |
| 11 | Visibility-split variants (graduated reveal) | `nithil_work/results/visibility_variants/` | 5 clips | 5s each | Split | 32×12 yellow | **OFF** |

#### Batch 11 in detail — the visibility split experiment

All 5 clips show the exact same 5 seconds of play (3-vs-1 attacking scenario, seed 23). The only difference is how much of the ball you can see before it disappears:

| File | What you see |
|---|---|
| `0s_vis_5s_hid.mov` | Ball is hidden the **entire** clip — you only see players |
| `1s_vis_4s_hid.mov` | Ball is visible for the **first 1 second**, then hidden for 4 |
| `2s_vis_3s_hid.mov` | Ball is visible for the **first 2 seconds**, then hidden for 3 |
| `3s_vis_2s_hid.mov` | Ball is visible for the **first 3 seconds**, then hidden for 2 |
| `4s_vis_1s_hid.mov` | Ball is visible for the **first 4 seconds**, then hidden for only 1 |

The idea: give a model or human more and more context and see if their guess about where the ball is gets better.

---

### Models tested so far

| Model | Where it runs | What it was tested on | Were results real? |
|---|---|---|---|
| Claude Sonnet (claude-sonnet-4-6) | Anthropic cloud API | All 5 benchmark categories | **DRY RUN only** — no real answers yet |
| Qwen 72B (qwen/qwen2.5-vl-72b-instruct) | OpenRouter cloud API | All 5 categories | **DRY RUN only** |
| Qwen 7B (qwen2.5vl:7b) | Local laptop (Ollama) | Description + prediction + ball-hidden frame | Live — real answers, low accuracy |
| LLaVA 7B (llava:7b) | Local laptop (Ollama) | Description + prediction + ball-hidden frame | Live — refused or vague answers |
| MiniCPM-V (minicpm-v) | Local laptop (Ollama) | Description + prediction + ball-hidden frame | Live — refused most questions |

**Next step not done yet:** run the real (non-dry-run) evaluation on the visibility-split clips.

---

## Activity Log

<!-- AUTO-UPDATED — do not edit below this line manually -->
<!-- Entries appended by log_entry.py, newest at bottom -->

<!-- AUTO-UPDATED — do not edit below this line manually -->
<!-- Entries appended by log_entry.py, newest at bottom -->

### 2026-06-29 — Initial benchmark stimulus generation

- Generated description batch: 4 static frames from `custom_1v1_base` scenarios
- Generated prediction batch: 4 clips (2 seeds × 2 cutoff times) from `academy_3_vs_1_with_keeper`
- Generated inference batch: 2 items with ball_hidden / player_hidden / region_occluded variants
- Generated counterfactual batch: 2 goal-moment clips
- Generated hypothetical batch: 2 items × 5 intervention frames each

### 2026-06-30 — Model eval runs (dry run) + local model tests

- Ran DRY RUN eval of `claude-sonnet-4-6` across all 5 categories (26 items) → `results/20260630_085953_claude.jsonl`
- Ran DRY RUN eval of `qwen/qwen2.5-vl-72b-instruct` across all 5 categories → `results/20260630_085955_qwen.jsonl`
- Ran ball_hidden_4s experiment on single frame at 4s, 6×4 text-described grid → `results/ball_hidden_4s/e9d90747ad/`
  - Models: qwen (local 7B), llava:7b, minicpm-v
  - Result: qwen predicted col=3 (off by 1), llava refused, minicpm refused
- Probed episode lengths across 6 scenarios × 4 seeds to find which survive 50+ steps → `results/_probe/`
- Ran multiple local qwen runs with varied prompts (6 qwen jsonl files, 2 llava, 2 minicpm)

### 2026-07-01 — Ball-hidden grid video generation

- Generated ball_hidden_grid batch (Batch 7): 3 clips with 32×12 grid burned in, ball invisible
  - `08fcceebae`: academy_3_vs_1_with_keeper, seed=23
  - `9066812d07`: academy_counterattack_easy, seed=331
  - `666bb5de48`: academy_counterattack_hard, seed=3
- Each item has both `clip.mov` (model input) and `debug_full.mov` (QA only, ball visible + red marker)

### 2026-07-02 — Plain 20-second grid clips (Batch 9)

- Generated 4 × 20-second gameplay clips with ball VISIBLE and 32×12 grid burned in
  - Scenarios: 3v1_with_keeper, counterattack_easy, counterattack_hard, run_pass_shoot_with_keeper
  - Seeds: 23, 331, 3, 3
  - Asset bundle: default (all visuals intact)
  - Episodes that end early auto-reset and continue
- Script: `experiments/nithil_work/gen_plain_grid_clips.py`
- Output: `experiments/nithil_work/results/plain_grid_clips/`

### 2026-07-02 — Generated 5s no-name grid clips (gen_5s_noname_clips.py)

- clip_1: academy_3_vs_1_with_keeper, seed=23, 5s, ball visible, 32x12 grid, no player names, 0.5 MB
- clip_2: academy_counterattack_easy, seed=331, 5s, ball visible, 32x12 grid, no player names, 0.6 MB
- clip_3: academy_counterattack_hard, seed=3, 5s, ball visible, 32x12 grid, no player names, 0.6 MB
- clip_4: academy_run_pass_and_shoot_with_keeper, seed=3, 5s, ball visible, 32x12 grid, no player names, 0.5 MB
- Output: experiments/nithil_work/results/5s_noname_clips/
- Bundle: default | Steps: 50 | FPS: 10 | Grid: 32x12 | Names: OFF | HUD: OFF

### 2026-07-02 — Generated visibility-split variants (gen_visibility_variants.py)

- Source: academy_3_vs_1_with_keeper, seed=23, 5s total, 32x12 grid, no player names
- 0s_vis_5s_hid.mov: 0s ball visible then 5s ball hidden, 0.4 MB
- 1s_vis_4s_hid.mov: 1s ball visible then 4s ball hidden, 0.4 MB
- 2s_vis_3s_hid.mov: 2s ball visible then 3s ball hidden, 0.4 MB
- 3s_vis_2s_hid.mov: 3s ball visible then 2s ball hidden, 0.4 MB
- 4s_vis_1s_hid.mov: 4s ball visible then 1s ball hidden, 0.4 MB
- Output: experiments/nithil_work/results/visibility_variants/
- Visible = default bundle | Hidden = ball_tiny bundle | Same seed = identical physics

### 2026-07-02 — Generated 5s no-name grid clips (gen_5s_noname_clips.py)

- clip_1: academy_3_vs_1_with_keeper, seed=23, 5s, ball visible, 32x12 grid, no player names, 0.5 MB
- clip_2: academy_counterattack_easy, seed=331, 5s, ball visible, 32x12 grid, no player names, 0.6 MB
- clip_3: academy_counterattack_hard, seed=3, 5s, ball visible, 32x12 grid, no player names, 0.6 MB
- clip_4: academy_run_pass_and_shoot_with_keeper, seed=3, 5s, ball visible, 32x12 grid, no player names, 0.5 MB
- Output: experiments/nithil_work/results/5s_noname_clips/
- Bundle: default | Steps: 50 | FPS: 10 | Grid: 32x12 | Names: OFF | HUD: OFF

### 2026-07-02 — Generated visibility-split variants (gen_visibility_variants.py)

- Source: academy_3_vs_1_with_keeper, seed=23, 5s total, 32x12 grid, no player names
- 0s_vis_5s_hid.mov: 0s ball visible then 5s ball hidden, 0.4 MB
- 1s_vis_4s_hid.mov: 1s ball visible then 4s ball hidden, 0.4 MB
- 2s_vis_3s_hid.mov: 2s ball visible then 3s ball hidden, 0.4 MB
- 3s_vis_2s_hid.mov: 3s ball visible then 2s ball hidden, 0.4 MB
- 4s_vis_1s_hid.mov: 4s ball visible then 1s ball hidden, 0.4 MB
- Output: experiments/nithil_work/results/visibility_variants/
- Visible = default bundle | Hidden = ball_tiny bundle | Same seed = identical physics

### 2026-07-02 — Generated 5s no-name grid clips (gen_5s_noname_clips.py)

- clip_1: academy_3_vs_1_with_keeper, seed=23, 5s, ball visible, 32x12 grid, no player names, 0.5 MB
- clip_2: academy_counterattack_easy, seed=331, 5s, ball visible, 32x12 grid, no player names, 0.6 MB
- clip_3: academy_counterattack_hard, seed=3, 5s, ball visible, 32x12 grid, no player names, 0.6 MB
- clip_4: academy_run_pass_and_shoot_with_keeper, seed=3, 5s, ball visible, 32x12 grid, no player names, 0.5 MB
- Output: experiments/nithil_work/results/5s_noname_clips/
- Bundle: default | Steps: 50 | FPS: 10 | Grid: 32x12 | Names: OFF | HUD: OFF

### 2026-07-02 — Generated visibility-split variants (gen_visibility_variants.py)

- Source: academy_3_vs_1_with_keeper, seed=23, 5s total, 32x12 grid, no player names
- 0s_vis_5s_hid.mov: 0s ball visible then 5s ball hidden, 0.4 MB
- 1s_vis_4s_hid.mov: 1s ball visible then 4s ball hidden, 0.4 MB
- 2s_vis_3s_hid.mov: 2s ball visible then 3s ball hidden, 0.4 MB
- 3s_vis_2s_hid.mov: 3s ball visible then 2s ball hidden, 0.4 MB
- 4s_vis_1s_hid.mov: 4s ball visible then 1s ball hidden, 0.4 MB
- Output: experiments/nithil_work/results/visibility_variants/
- Visible = noname bundle (normal ball, no names) | Hidden = noname_ball_invisible (5% ball, no names)

### 2026-07-02 — Generated 5s no-name grid clips (gen_5s_noname_clips.py)

- clip_1: academy_3_vs_1_with_keeper, seed=23, 5s, ball visible, 32x12 grid, no player names, 0.4 MB
- clip_2: academy_counterattack_easy, seed=331, 5s, ball visible, 32x12 grid, no player names, 0.6 MB
- clip_3: academy_counterattack_hard, seed=3, 5s, ball visible, 32x12 grid, no player names, 0.6 MB
- clip_4: academy_run_pass_and_shoot_with_keeper, seed=3, 5s, ball visible, 32x12 grid, no player names, 0.5 MB
- Output: experiments/nithil_work/results/5s_noname_clips/
- Bundle: default | Steps: 50 | FPS: 10 | Grid: 32x12 | Names: OFF | HUD: OFF

### 2026-07-02 — Generated visibility-split variants (gen_visibility_variants.py)

- Source: academy_3_vs_1_with_keeper, seed=23, 5s total, 32x12 grid, no player names
- 0s_vis_5s_hid.mov: 0s ball visible then 5s ball hidden, 0.4 MB
- 1s_vis_4s_hid.mov: 1s ball visible then 4s ball hidden, 0.4 MB
- 2s_vis_3s_hid.mov: 2s ball visible then 3s ball hidden, 0.4 MB
- 3s_vis_2s_hid.mov: 3s ball visible then 2s ball hidden, 0.4 MB
- 4s_vis_1s_hid.mov: 4s ball visible then 1s ball hidden, 0.4 MB
- Output: experiments/nithil_work/results/visibility_variants/
- Visible = noname bundle (normal ball, no names) | Hidden = noname_ball_invisible (5% ball, no names)

### 2026-07-03 — Generated 30 × 10s no-name grid clips (gen_10s_noname_clips.py)

- 6 scenarios × 5 seeds = 30 clips total
- clip_01: academy_3_vs_1_with_keeper, seed=3, 10s, ball visible, no names, 32x12 grid, 1.0 MB
- clip_02: academy_3_vs_1_with_keeper, seed=7, 10s, ball visible, no names, 32x12 grid, 1.0 MB
- clip_03: academy_3_vs_1_with_keeper, seed=11, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_04: academy_3_vs_1_with_keeper, seed=23, 10s, ball visible, no names, 32x12 grid, 0.9 MB
- clip_05: academy_3_vs_1_with_keeper, seed=42, 10s, ball visible, no names, 32x12 grid, 0.9 MB
- clip_06: academy_counterattack_easy, seed=3, 10s, ball visible, no names, 32x12 grid, 1.3 MB
- clip_07: academy_counterattack_easy, seed=7, 10s, ball visible, no names, 32x12 grid, 1.3 MB
- clip_08: academy_counterattack_easy, seed=11, 10s, ball visible, no names, 32x12 grid, 1.4 MB
- clip_09: academy_counterattack_easy, seed=23, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_10: academy_counterattack_easy, seed=42, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_11: academy_counterattack_hard, seed=3, 10s, ball visible, no names, 32x12 grid, 1.3 MB
- clip_12: academy_counterattack_hard, seed=7, 10s, ball visible, no names, 32x12 grid, 1.3 MB
- clip_13: academy_counterattack_hard, seed=11, 10s, ball visible, no names, 32x12 grid, 1.3 MB
- clip_14: academy_counterattack_hard, seed=23, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_15: academy_counterattack_hard, seed=42, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_16: academy_run_pass_and_shoot_with_keeper, seed=3, 10s, ball visible, no names, 32x12 grid, 1.0 MB
- clip_17: academy_run_pass_and_shoot_with_keeper, seed=7, 10s, ball visible, no names, 32x12 grid, 1.0 MB
- clip_18: academy_run_pass_and_shoot_with_keeper, seed=11, 10s, ball visible, no names, 32x12 grid, 1.0 MB
- clip_19: academy_run_pass_and_shoot_with_keeper, seed=23, 10s, ball visible, no names, 32x12 grid, 1.0 MB
- clip_20: academy_run_pass_and_shoot_with_keeper, seed=42, 10s, ball visible, no names, 32x12 grid, 1.1 MB
- clip_21: academy_pass_and_shoot_with_keeper, seed=3, 10s, ball visible, no names, 32x12 grid, 1.1 MB
- clip_22: academy_pass_and_shoot_with_keeper, seed=7, 10s, ball visible, no names, 32x12 grid, 1.0 MB
- clip_23: academy_pass_and_shoot_with_keeper, seed=11, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_24: academy_pass_and_shoot_with_keeper, seed=23, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_25: academy_pass_and_shoot_with_keeper, seed=42, 10s, ball visible, no names, 32x12 grid, 1.1 MB
- clip_26: academy_run_to_score_with_keeper, seed=3, 10s, ball visible, no names, 32x12 grid, 1.3 MB
- clip_27: academy_run_to_score_with_keeper, seed=7, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_28: academy_run_to_score_with_keeper, seed=11, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- clip_29: academy_run_to_score_with_keeper, seed=23, 10s, ball visible, no names, 32x12 grid, 1.1 MB
- clip_30: academy_run_to_score_with_keeper, seed=42, 10s, ball visible, no names, 32x12 grid, 1.2 MB
- Output: experiments/nithil_work/results/10s_ball_visible_no_names/
- Bundle: noname | GFOOTBALL_FONT: blank font (all chars -> space glyph) | Steps: 100 | FPS: 10

### 2026-07-03 — Generated 90 × 10s visibility-split clips (gen_10s_visibility_splits.py)

- 30 base clips × 3 variants each (3s/5s/8s visible, rest hidden)
- Variants: 3s_vis_7s_hid, 5s_vis_5s_hid, 8s_vis_2s_hid per clip
- Purpose: test whether more time watching ball improves model predictions
- Output: experiments/nithil_work/results/10s_visibility_splits/clip_01/ ... clip_30/
- Bundles: noname (visible) + noname_ball_invisible (hidden, 5% ball scale)

### 2026-07-08 — Generated 2 natural 5s passing clips × 3 grid variants (gen_natural_passing_clips.py)

- Brief: continuous play, NO goals — passing / moving / chasing a moving ball (not everyone dribbling); natural framing (do NOT force all 22 on frame); ball visible throughout; no names
- Fix vs prior match_clips batch: dropped wide full_field + zoomed half_field crops (looked OD); used only the standard ball-tracking camera for natural framing
- Match: 11_vs_11_stochastic, seed 42 (verified goal-free across 450 steps, score 0-0)
- clip1: warmup=30 (steps 30-80), settled midfield possession/passing
- clip2: warmup=110 (steps 110-160), ball travels laterally, some chasing
- Grid variants per clip (labelled, burned): 32x12, 16x6, 28x8, 8x3 (yellow alphanumeric)
- 10 clips total (clipN_base + clipN_grid_{32x12,16x6,28x8,8x3}), all 1280x480, 50 frames @ 10 fps = 5.0s
- Note: green-jersey players are the goalkeepers (one per side)
- Output: experiments/nithil_work/results/natural_passing_clips/
- Bundle: noname (ball visible, no player-name captions) | HUD: cropped
