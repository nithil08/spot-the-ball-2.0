# 10-Week Plan

This is a guide, not a contract. If something takes you longer, that's
fine. If you finish early and want stretch goals, they're at the bottom
of each week.

**Daily journal.** Every day you work, write a short entry in
`Nithil/journal.md` — the file has a template and an example. 5-10
minutes, four short questions. The daily habit is more valuable than
any of the weekly "deliverables" below; if you only do one of the two,
do the journal.

**Weekly sync with Neha:** ~30 min, set a recurring time. Bring your
journal. The weekly deliverables are what you *show off* at sync —
your finished artifact for the week. The journal is the record of how
you got there.

---

## Week 1: Get set up, see the system

**Read / watch (≈2 hrs):**
- This folder's `README.md` and `SETUP.md`.
- 3Blue1Brown, "But what *is* a neural network?" (YouTube, 19 min).
- Neha's prior paper on "spot the ball" — she'll share the PDF. Read the
  intro and figures; skip the methods for now.

**Do:**
- Complete the SETUP guide. Make a video clip play on your laptop.
- Open `experiments/TASKS.md` and read it cover to cover. Then write a
  ~200-word summary in your journal explaining the 5 task categories in
  your own words. This is your "do you actually get it" check.
- Watch all four videos in `experiments/clips/compare_5v5/` and
  `experiments/clips/*/`. Notice what changes between the bundles.

**Deliverable:** journal entry + a single text message to Neha that
says "everything works, I'm ready" — or describes exactly where you got
stuck.

---

## Week 2: The science behind the project

**Read / watch (≈4 hrs):**
- 3Blue1Brown, "Transformers, the tech behind LLMs" (YouTube, 27 min).
  Watch it twice if it doesn't stick the first time. Most people need
  two passes.
- Andrej Karpathy, "State of GPT" (YouTube, ~40 min). The first 15
  minutes are the most important.
- Skim the gfootball paper:
  https://arxiv.org/abs/1907.11180 — just the abstract, intro, and
  Figure 1.
- Read about "theory of mind" — a one-pager intro. Search "theory of
  mind cognitive science" and pick any explainer that looks readable.

**Do:**
- Run all 5 generators end-to-end at `--n=2` each. Watch every output
  video. Make a table in your journal: which categories look good, which
  look weird, what specifically seems off.
- Browse `stimuli/inference/*/player_hidden.mov`. Note: the green dot
  that's supposed to hide a player is currently in the wrong spot. This
  is a known bug — you'll fix it in Week 4.
- **Soccer events brainstorm (~2 hrs).** Watch a Premier League goals
  compilation on YouTube (any season). For ~10 goals, log in
  `ideas.md`: (a) what specific event led to the goal? (b) what would
  a human naturally predict 3 seconds before it? (c) do you think our
  simulator could reproduce a scene like that? This seeds which kinds
  of stimuli we prioritize in Week 5.

**Deliverable:** journal entry + at least 10 entries in `ideas.md`.
Also try to explain in 3-4 sentences "what is a Transformer doing
when it processes an image" — even if you're unsure, write your best
guess. Neha will give feedback.

---

## Week 3: Learn enough Python to be dangerous

You don't need to be a Python expert. You do need to read existing code
without panicking and write small scripts that loop, read files, and
print things.

**Read / watch (≈5 hrs):**
- "Automate the Boring Stuff with Python," chapters 1-6. Free online at
  https://automatetheboringstuff.com. This is the standard
  no-CS-background intro.
- Specifically: chapters on lists, dictionaries, functions, files. Skip
  regular expressions for now.

**Do:**
- Open `experiments/lib.py` and read it line by line. Where you don't
  understand something, paste it into Claude/ChatGPT and ask "explain
  this line." Don't move on until each function makes sense.
- Open `experiments/gen_description.py` and do the same.
- **Mini-project:** create `experiments/nithil_work/gallery.py`. It
  should walk through every folder under `experiments/stimuli/`, find
  every `meta.json`, and print one line per item: `<id> <category>
  <ground truth summary>`. About 30 lines of code. Ask Claude for help
  scaffolding it but make sure you understand every line.

**Deliverable:** working `gallery.py` + journal entry.

---

## Week 4: Your first real fix

**Context:** in the `gen_inference.py` script, we draw a green circle on
top of one player to hide them. The circle is in the wrong place because
the function that converts "soccer field coordinates" to "pixel
coordinates" was eyeballed. You're going to fix it.

**Read:**
- `experiments/lib.py`, the function `pitch_to_pixel`. Understand what
  it does (math: linear scaling from one range to another).
- Look at any `meta.json` from an `inference` item — note the
  `hidden_player.xy` field. These are the *true* coordinates of the
  hidden player on the soccer field.

**Do:**
- Build a calibration tool: a Python script that loads one rendered
  frame and overlays a dot at every player's predicted pixel
  location. You'll *see* how far off the prediction is.
- Adjust the constants in `pitch_to_pixel` (in `lib.py`) until dots line
  up with players. This is iterative — change a number, re-run, look.
- Re-run `gen_inference.py --n=1` and verify the green circle is now on
  top of the right player.

**Stretch:** if it's still wrong after linear adjustment, the camera has
*perspective* (the field is tilted slightly). Look up "homography" and
ask Claude how a 2D homography would let you map field → screen more
accurately. Don't implement it unless linear isn't enough.

**Deliverable:** updated `lib.py`, before/after frame showing the fix.
Journal entry.

---

## Week 5: Scale up generation

**Read:**
- `experiments/TASKS.md` again, this time with comprehension. Notice the
  difference between "stimuli" (the videos) and "ground truth" (the
  answer).

**Do:**
- Generate the first real batch:
  - 30 description items (`--n=30`)
  - 30 prediction items
  - 20 inference items
  - 15 hypothetical items (this one's slow — let it run overnight)
  - 15 counterfactual items (also slow)
- For each item, watch the clip. Decide if the stimulus is "good" or
  "broken" (something looks weird, ball goes through a wall, scene is
  ambiguous). Record judgments in a CSV you build:
  `experiments/nithil_work/qc_round1.csv` with columns
  `id, category, status, notes`.
- Report at the end: out of N items, how many were good? What were the
  most common problems?

**Deliverable:** CSV + 1-page summary in journal.

---

## Week 6: Improve the pipeline based on what you saw

Based on Week 5, you'll have a list of recurring problems. Pick the two
most common and fix them. Examples of what might come up:

- Scenarios end too early (the play stops after 3 s instead of 10 s).
- The HUD strip Neha cropped doesn't fully cover the radar on some
  resolutions.
- Some scenarios put the ball in a position where nothing interesting
  happens.

For each fix:
- Write the smallest possible change.
- Re-generate 10 items.
- Confirm the bug is gone.

**Then — pick one idea from your `ideas.md` and try to build it.**
This is the part where you get intellectual ownership of the project.
Bring 2-3 candidate ideas to your weekly sync; Neha will help you pick
one that's tractable. The goal isn't a fully working production
feature — it's a scrappy prototype + a written-up answer to "did this
manipulation reveal something interesting?"

Good first-time ideas usually fit this shape:
- A new asset bundle (mirror of `ball_tiny`/`uniform_jerseys`).
- A new post-processing effect on existing clips (e.g., motion blur,
  silhouette).
- A new scenario type that captures a soccer event from your Week 2
  list (counter-attack, breakaway, etc).

If you're stuck, the safe options are:
- `ball_huge` — opposite of `ball_tiny`. Just for fun.
- `no_logos` — overwrite team logo BMPs with blank ones to remove
  another visual cue.
- `night_mode` — overwrite the grass texture with a darker version.
- `silhouettes` — replace player skin/jersey with a single solid
  color per team. Strips body language without merging teams.

**Deliverable:** fixed pipeline + one new manipulation you built
yourself + 1-page write-up in your journal: what manipulation did you
build, why did you think it'd be interesting, what did you actually
find when you watched the result?

---

## Week 7: Meet the AI models

**Read / watch (≈4 hrs):**
- Karpathy, "Let's build GPT from scratch" (YouTube, ~2 hr). You can
  skip the math-heavy parts; the goal is to feel what an LLM is doing
  internally, not to derive it.
- Anthropic's docs intro:
  https://docs.anthropic.com — the "messages" API quick start.

**Do:**
- Get a Claude API key from Neha (she'll set you up with a small
  budget). Add it to a `.env` file (Neha will show you).
- Write `experiments/nithil_work/vlm_baseline.py`. It should:
  1. Pick one description item.
  2. Load its `frame.png`.
  3. Send it to Claude with the question: "How many players are on the
     left team?"
  4. Print Claude's response.
- About 40 lines of code. Anthropic's docs have a copy-paste example.
- Once it works for one item, loop it over 10 items and record the
  responses in a CSV.

**Deliverable:** working script + CSV of 10 model responses + comparison
to ground truth in your journal.

---

## Week 8: Scale the eval

**Do:**
- Generalize last week's script to all 5 categories. The trick is each
  category has different stimuli and different questions. You'll need
  small if/else for each.
- Run the eval on the items you generated in Week 5 (~110 items total).
- Compute accuracy per category. Make a simple bar chart with
  `matplotlib`.

**Stretch:** add a second model — GPT-4o or Gemini. Compare. You'll
need an OpenAI or Google API key; ask Neha.

**Deliverable:** bar chart + 1-page interpretation in journal: which
categories does the model handle? Where does it fail? Were you
surprised?

---

## Week 9: Analysis and reflection

**Read:**
- Pick *one* paper from `READING.md`'s "Cognitive science" section and
  read it carefully. Take notes.

**Do:**
- Look at the items the model got wrong. For 10 of them, write what you
  think the model was confused by.
- Cross-reference: are the model's mistakes also hard for you? Or does
  it fail on things you find obvious?
- If you have time, run the eval with "chain-of-thought" prompting (add
  "Think step by step." to the prompt). Does it change anything?

**Deliverable:** 2-page write-up in journal:
- What did I do this summer?
- What surprised me about VLMs?
- What surprised me about doing research?
- What would I want to do next?

This becomes your "I did research" portfolio piece for college apps.

---

## Week 10: Hand-off and wrap-up

**Do:**
- Clean up `nithil_work/`. Delete experiments that went nowhere. Add a
  `README.md` at the top that explains what's in there.
- Write a 1-page summary for Neha covering: the items you generated,
  the eval you ran, the bugs you found, the open questions.
- Co-author paragraph in the methods section of the paper, where
  appropriate.

**Deliverable:** clean folder, summary doc. You're done.

---

## What "done" actually means

You don't need to finish every week perfectly. You'll get more out of
deep work on weeks 4, 7, and 8 than skimming all 10. If something is
taking longer than expected and you're learning, that's the right call.
Tell Neha and adjust.

The thing you should never skip: the weekly journal entry. It's how you
notice your own learning.
