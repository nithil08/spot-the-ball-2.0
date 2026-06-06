# Read this first.

Hi! Welcome to the project. This folder is yours — it's your onramp into a
real research project. Don't be intimidated by the rest of the repo; you
won't need most of it right away.

## The 30-second pitch

You watch a soccer clip. Two players run toward the ball, a third sprints
into open space, the goalie steps off the line. Without thinking, you know
who's on which team, who's about to pass to whom, who messed up if the goal
goes in. Your brain just *does* this — it builds a little model of what's
happening and runs it forward.

AI doesn't yet. Big "vision-language models" (VLMs) — the things behind
ChatGPT-with-images, Gemini, Claude — can describe a soccer field, but they
struggle with the kind of reasoning that you do effortlessly: predicting
the next move, inferring hidden players, figuring out who's responsible
for a goal.

We're building an experiment to measure exactly *where* this gap is. The
idea: take a soccer simulator (the same engine Google used to train
soccer-playing AIs), generate thousands of short video clips, and ask the
same questions to humans (via online studies) and to the latest VLMs. The
pattern of where AI matches humans and where it falls apart tells us
something fundamental about what AI is missing.

This is a paper Neha is writing this year aimed at a top conference. Your
work helps build the dataset that the whole paper rests on.

## About the simulator

We use a piece of software called **gfootball** — short for "Google
Research Football." It's a real-time soccer simulator that Google built
in 2019 to train AI agents to play soccer. They wanted an environment
where an AI could practice playing matches millions of times faster than
real life. The simulator handles all the parts of soccer you'd
expect — physics of the ball, players running and tackling, passing,
shooting, goals, fouls, offsides, the works.

We're not using it for what it was designed for (training AI to play).
We're using it as a **controlled video generator**. Here's why that
matters:

Real soccer footage is *messy*. Commentary tells you the score, the
broadcast camera cuts between angles, you can sometimes read jersey
numbers and team logos, weather and lighting vary, players have
distinctive hairstyles and body types. If you tested an AI on real
soccer clips, you'd never know whether it was "reasoning about the
play" or just "reading the score graphic in the corner." Real video is
*correlated with everything*.

A simulator removes all that. We can take the *same* play and render
it twelve ways:
- With the ball hidden, to test whether the AI can infer where it is.
- With one defender removed, to test the AI's intuition for "what
  would have happened if..."
- With both teams in identical jerseys, to test whether team
  identification was actually doing the work.
- From a different camera angle. With different player textures.
  With one player frozen mid-stride. With the goalkeeper having a
  different policy.

Each variation gives us *exactly the same scenario* with one thing
changed. That's the only way to reliably attribute an AI's failure to a
specific cognitive ability (or lack of one). You can't do this with
real video.

What we ultimately have control over (this is worth knowing):
- **Player positions, count, roles** (goalkeeper, defender, etc).
- **Ball position and starting velocity.**
- **Per-player behavior policies** — every player can be controlled
  by a "smart bot," frozen ("lazy"), or run by a human/AI.
- **Visual assets** — player models, jerseys, ball textures, pitch
  appearance.
- **Game state** — score, time remaining, offside rules, etc.
- **Random seed** — same seed → same exact play, frame-by-frame
  identical. Critical for counterfactuals.

What we *don't* easily control: camera angle (the engine has a fixed
top-down view), referee decisions (some are scripted in C++), crowd
sounds (not relevant for us).

## What you'll do

Three things, in increasing depth:

1. **Make stimulus videos** using the simulator. There are 5 categories of
   videos and roughly 460 we eventually want. You'll learn the pipeline,
   generate them, watch them, throw out broken ones.
2. **Run AI models on the videos.** Write a little script that sends each
   clip to Claude / Gemini / GPT / other open source models and records what they say. This is the
   part where you'll learn how AI APIs actually work.
3. **Analyze the results.** Make plots showing where AI matches humans and
   where it doesn't. This is the part that actually goes in the paper.

Along the way you'll learn:
- Python (enough to write small scripts and read existing ones)
- How to use a research codebase (most of it is already written; you'll
  read and run, not write from scratch)
- Roughly how Transformers and VLMs work (the architecture behind every
  modern AI model)
- What "doing research" actually looks like day-to-day (it's messy)

## The bigger question (and why you matter beyond running scripts)

The five task categories we already have (description, prediction,
inference, hypothetical, counterfactual) are a *starting point*, not a
finished design. The whole project gets stronger if you can think of
manipulations or probes we haven't thought of.

That's not flattery — it's a real ask. Spending six weeks watching
soccer clips gives you an intuition for the domain that I (Neha) might
have lost from staring at code. You'll notice things I won't. So I want
you to keep a running list of ideas in a file called
`Nithil/ideas.md`, and we'll talk through them at every weekly sync.
Most won't pan out. One or two might end up in the paper.

To prime the pump, here are open questions worth thinking about:

### 1. What else could we manipulate?

We currently do: shrink the ball, swap jerseys, occlude a region,
remove a defender, freeze a player. What other manipulations would
test something interesting?

Some directions to brainstorm:
- **Visual:** silhouettes only, motion blur, low resolution, swap
  player models entirely, hide jersey numbers, mismatched skin tones.
- **Temporal:** speed up, slow down, reverse, skip frames, freeze for
  N seconds mid-clip.
- **Spatial:** mirror flip the whole field, zoom in on one player,
  black out everything outside a circle around the ball.
- **Game state:** start with score 0-0 vs 4-0 (do humans/AI bring
  different expectations?), start with a player on a yellow card.
- **Multi-clip:** show two clips and ask "are these from the same
  team's play style?"

For each idea, ask yourself: *what cognitive ability does this probe
test?* If the answer isn't clear, the manipulation probably isn't
useful.

### 2. What soccer events are worth probing specifically?

Soccer has natural units of action — a counter-attack, a corner kick,
a one-on-one with the goalkeeper, a defensive recovery, a build-up
through the midfield. Each of these tests a different kind of
reasoning.

I want you to spend ~2 hours during Week 2 watching real soccer
highlights on YouTube (a Premier League goals compilation is great) and
make a list in `ideas.md` of:
1. What happened just before each goal? (Specific moments.)
2. What would a human viewer naturally predict 3 seconds before
   the goal? (Possession, shot direction, who'd score.)
3. Which of these events does our simulator already produce well?
   Which doesn't it produce at all?

This list directly informs Week 5 — when we scale up stimulus
generation, we can prioritize the events your list flags as rich and
under-represented.

### 3. What about the goalkeeper?

The goalkeeper is special. They have a different role, different
expected behaviors, different position priors. *Should* our experiments
treat goalkeepers as a distinct category? Should we ask things like
"will the goalkeeper save this shot?" as its own category? Or is that
just a special case of prediction?

I don't have a settled answer. Think about it as you watch clips.
Write down your intuitions.

### 4. The meta-question

The point of this benchmark isn't to make AIs look bad. It's to find
out *which specific cognitive abilities* are missing — so future
researchers (maybe you?) can design models that have them.

If you come away from a week of watching clips with a story for *why*
a certain VLM failure happens — not "it's just bad," but "it can't do
X, and X requires Y" — that story is the most valuable thing you can
contribute. Concrete hypotheses about *what's missing inside the
model* are what science papers actually argue.

I'll grade your ideas leniently. Quantity first; we'll sharpen the
best ones together.

## How to use this folder

Read these in order:

1. **`SETUP.md`** — get the project running on your laptop. Day 1 stuff.
   Sanity-check: can you generate a video and watch it? (you've already done this part!! good work)
2. **`PLAN.md`** — your 10-week structured plan. Each week has a reading
   list, tasks, and a deliverable. **Follow this loosely, not rigidly.**
3. **`READING.md`** — books, videos, papers worth reading at your own pace
   to build context. Don't binge it; pick up one a week.
4. **`ideas.md`** — your running list of "what if we tried..." thoughts.
   See the "bigger question" section above for the ask. No formatting
   rules, just dated bullet points.
5. **`journal.md`** — your daily log. See the template inside.

When you're working, put your code under `experiments/nithil_work/`.
Don't edit the existing files in `experiments/` directly until you're
comfortable — you can break things and that's fine in your sandbox.

## How to ask for help

You're going to get stuck. That's not a failure mode; it's the work. When
you do:

1. **Try for 30 minutes first.** Re-read the error message. Google the
   exact error. Look at the file the traceback points to. Most of the
   time the answer is right there.
2. **Write down what you tried.** This makes asking 10x more useful — for
   both of you.
3. **Then ping me (Neha).** A good question looks like: *"I ran X, expected
   Y, got Z. I tried A and B. I don't understand why \[specific thing\]."*

If you're stuck on conceptual stuff (what's a Transformer, what does
"counterfactual" mean here, why does the paper care about this), ask
anytime. Concepts are the point; tool-fiddling is the means.

## A note on AI tools

Use Claude / ChatGPT freely. Use them for:
- Explaining unfamiliar code line by line
- Debugging error messages
- Suggesting how to write a small script
- Teaching you Python / Transformers concepts

Don't use them for:
- Writing your weekly reflection (those are *your* thinking; that's the
  whole point of doing this)
- Generating your reading-list notes wholesale
- Doing the eval/analysis work without understanding what's happening —
  if you can't explain your plot, it doesn't count

This is the same advice professional researchers follow.

## One more thing

You don't need to "be a coder" before you start. You'll learn by doing —
by running things, breaking them, fixing them, repeating. The point of
this summer isn't to become a senior engineer; it's to find out whether
this kind of work feels exciting to you. By August you should be able to
answer that for yourself.

Let's go.
