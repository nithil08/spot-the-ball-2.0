# Reading & watching list

You're not expected to read all of this. It's a curated menu — when you
want to go deeper on something, this is where to look. The PLAN.md
schedule already pulls the must-read items in at the right time.

Each item has a one-line "why" and a difficulty tag: 🟢 accessible,
🟡 some background helpful, 🔴 hard.

## Watch first (start here)

- 🟢 **3Blue1Brown — "But what *is* a neural network?"** (YouTube, 19 min).
  *Why:* the cleanest visual explanation of what a neural network
  computes. Watch even if you've seen it before.
- 🟢 **3Blue1Brown — "Transformers, the tech behind LLMs"** (YouTube, 27 min).
  *Why:* same series, on the architecture behind every modern AI model.
  Will give you 80% of the intuition you need.
- 🟡 **Andrej Karpathy — "State of GPT"** (YouTube, 40 min).
  *Why:* a senior researcher's tour of how modern LLMs are actually
  built — pretraining, fine-tuning, RLHF, the works. First 15 min are
  most important.
- 🟡 **Andrej Karpathy — "Let's build GPT from scratch"** (YouTube, 2 hr).
  *Why:* if you want to *feel* what a Transformer is doing, this is the
  one. Skip the heaviest math; the code-along is the point.

## How AI vision-language models work

- 🟡 **Sebastian Raschka — "Understanding Multimodal LLMs"** (blog).
  Search the title. *Why:* a clean breakdown of how images get fed into
  a Transformer, which is the core mechanism behind every VLM we're
  testing.
- 🟢 **OpenAI's GPT-4V system card** (PDF, skim).
  *Why:* shows what the model is and isn't claimed to do; useful for
  knowing what to expect when you run evals.
- 🔴 **"Attention Is All You Need"** (Vaswani et al, 2017).
  *Why:* the original Transformer paper. Famously dense. Read the
  abstract and Figure 1, skip the rest unless you want to torture
  yourself.

## Soccer simulator background

- 🟢 **gfootball paper** (Kurach et al, 2020), arXiv:1907.11180.
  *Why:* the abstract + intro tells you what the simulator does and why
  it exists. The paper is mostly about training AIs to *play* soccer,
  but the engine is what we use as a stimulus generator.
- 🟢 **The gfootball GitHub README**.
  *Why:* user-facing tour of what the engine can do; the screenshots
  alone give you intuition.

## Cognitive science (the *why* of the project)

- 🟢 **Spelke & Kinzler, "Core knowledge" (2007)**.
  *Why:* foundational piece on what cognitive abilities humans seem to
  come with built-in. Background for why "intuitive physics" and
  "agents" are research targets.
- 🟡 **Smith, Battaglia, Vul, "Sources of uncertainty in intuitive
  physics" (2013)**.
  *Why:* a clean example of how scientists probe internal "mental
  simulators" — analogous to what we're doing for *social* mental
  simulators.
- 🟡 **Baker, Saxe, Tenenbaum, "Bayesian theory of mind" (2011)**.
  *Why:* the mathematical formalization of "I infer what you want from
  watching what you do." Directly relevant to the inference and
  responsibility tasks in our paper.

## What "research" feels like

- 🟢 **Karpathy — "A Recipe for Training Neural Networks"** (blog).
  *Why:* less about training and more about how to debug research
  systematically. Read this once and re-read it whenever you feel lost.
- 🟢 **Patrick Winston — "How to Speak"** (MIT OCW lecture, 1 hr).
  *Why:* the best lecture ever recorded on how to present research.
  Useful when you write your final summary in Week 10.

## Spot-the-ball (Neha's prior work)

Neha will share the PDF directly. *Why:* this paper sets up the
intellectual frame for ours — the idea that VLM failures on "what's in
the picture" reveal something about what they're modeling internally.
You should know this paper cold by Week 3.

## Python (only if you need it)

- 🟢 **Automate the Boring Stuff with Python**, free at
  https://automatetheboringstuff.com.
  *Why:* the standard "I've never coded" → "I can write a useful
  script" tour. Chapters 1-6 cover everything you'll need this summer.

## A reading rhythm that works

- **Tuesday afternoon:** ~1 hour of focused reading or watching.
- **Friday afternoon:** ~30 min skimming something new, just to expand
  your taste.
- **Weekend:** zero. Rest matters.

If you finish a tagged item, jot 3-5 bullets in your journal: what was
new, what was familiar, what you'd want to follow up on. This habit
will make you a much better reader by August.
