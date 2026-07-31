"""build_presentation.py — assemble a presentation-ready folder for the spot-the-ball study.

Creates  <research root>/SPOT_THE_BALL_PRESENTATION/  with:
  README.md                     plain-language explanation of the whole experiment
  1_ball_hidden_clips/          the 30 clips shown to Gemini (red circle on frame 0,
                                ball then invisible for 4 s)   -> clipNN_hidden.mov
  2_ball_visible_clips/         the same 30 plays with the ball VISIBLE the whole time
                                (the visual "answer")          -> clipNN_visible.mov
  3_ground_truth/               final-frame ground-truth cell per clip + crosshair images
  4_gemini_predictions/         Gemini's answers, scores, and full reasoning per clip
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BATCH = HERE / "results" / "30_1s_visible_4s_invisible"
FR = BATCH / "_frames"
GT_CSV = BATCH / "ground_truth.csv"
JSONL = sorted((BATCH / "eval_runs").glob("run_gemini-2.5-pro_*.jsonl"),
               key=lambda p: p.stat().st_mtime)[-1]
RESEARCH_ROOT = HERE.parents[3]                       # .../Nithil Research
OUT = RESEARCH_ROOT / "SPOT_THE_BALL_PRESENTATION"

W, H, COLS, ROWS, FPS = 1280, 480, 16, 6, 10
CELL_W, CELL_H = W / COLS, H / ROWS
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

sys.path.insert(0, str(HERE.parent))
from lib import frames_to_mov  # noqa: E402


def build_grid():
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    line = (255, 255, 0, 130)
    for c in range(COLS + 1):
        dr.line([(int(round(c * CELL_W)), 0), (int(round(c * CELL_W)), H)], fill=line, width=1)
    for r in range(ROWS + 1):
        dr.line([(0, int(round(r * CELL_H))), (W, int(round(r * CELL_H)))], fill=line, width=1)
    font = ImageFont.truetype(FONT_PATH, 13)
    for c in range(COLS):
        dr.text((int(round(c * CELL_W)) + 2, 0), str(c + 1), fill=(255, 255, 0, 255), font=font)
    for r in range(ROWS):
        dr.text((1, int(round(r * CELL_H)) + 1), chr(ord("A") + r), fill=(255, 255, 0, 255), font=font)
    return ov


def burn(frame, overlay):
    base = Image.fromarray(frame).convert("RGBA")
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def make_visible_clips(dst, overlay):
    for vp in sorted(FR.glob("clip*_vis.npz"), key=lambda p: int(p.name[4:6])):
        cid = int(vp.name[4:6])
        frames = [burn(f, overlay) for f in np.load(vp)["frames"]]   # all 50, ball visible
        frames_to_mov(frames, dst / f"clip{cid:02d}_visible.mov", fps=FPS, crop_hud=False)
        print(f"  visible clip{cid:02d}")


def copy_hidden_clips(dst):
    for cid in range(1, 31):
        src = BATCH / f"clip{cid:02d}_1s_visible_then_4s_invisible.mov"
        shutil.copy(src, dst / f"clip{cid:02d}_hidden.mov")


def copy_ground_truth(dst):
    for name in ("ground_truth.csv", "ground_truth.md", "ground_truth.json"):
        shutil.copy(BATCH / name, dst / name)
    dbg = dst / "ball_location_images"
    dbg.mkdir(exist_ok=True)
    for p in sorted((BATCH / "_gt_debug").glob("clip*.png")):
        shutil.copy(p, dbg / p.name)


def clean_answer(cell, row):
    return f"{row}, col {cell[1:]}" if cell else "—"


def build_predictions(dst):
    gt = {}
    with GT_CSV.open() as f:
        for r in csv.DictReader(f):
            gt[int(r["clip"])] = r["ball_cell"]
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["clip"])
    n = len(rows)
    exact = sum(1 for r in rows if r["grade"].get("correct"))
    within1 = sum(1 for r in rows if r["grade"].get("row_err") is not None
                  and r["grade"]["row_err"] <= 1 and r["grade"]["col_err"] <= 1)
    dists = [r["grade"]["cell_dist"] for r in rows if r["grade"].get("cell_dist") is not None]
    mean_d = round(sum(dists) / len(dists), 2) if dists else None

    md = ["# Gemini 2.5 Pro — predictions & reasoning", "",
          "Model: **gemini-2.5-pro** (Google Vertex AI). One video per clip; the ball is",
          "visible for the first second (red circle marks its start) then invisible for 4 s.",
          "Gemini must name the grid cell holding the ball in the **final frame**.", "",
          "## Scoreboard",
          f"- **Exact cell: {exact}/{n} ({round(100*exact/n)}%)**",
          f"- Within 1 cell (neighbouring): {within1}/{n} ({round(100*within1/n)}%)",
          f"- Average miss distance: {mean_d} cells",
          f"- Unreadable answers: {sum(1 for r in rows if r['pred_cell'] is None)}/{n}",
          "", "## Results table",
          "| Clip | True cell | Gemini's cell | Correct? | Miss (cells) |",
          "|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        ok = "✅" if r["grade"].get("correct") else "❌"
        md.append(f"| {r['clip']:02d} | {r['gt_cell']} | {r['pred_cell'] or '—'} | {ok} "
                  f"| {r['grade'].get('cell_dist')} |")
    md += ["", "## Gemini's reasoning, clip by clip", ""]
    for r in rows:
        ok = "correct ✅" if r["grade"].get("correct") else "wrong ❌"
        md.append(f"### Clip {r['clip']:02d} — true **{r['gt_cell']}**, "
                  f"Gemini said **{r['pred_cell'] or '—'}** ({ok})")
        raw = r["raw"].strip()
        reason = raw.split("Reasoning:", 1)[-1].split("Ball position:")[0].strip()
        answer = ("Ball position:" + raw.split("Ball position:", 1)[-1].strip()) \
            if "Ball position:" in raw else raw
        md.append(f"> {reason}")
        md.append(f"> ")
        md.append(f"> **{answer}**")
        md.append("")
    (dst / "gemini_predictions_and_reasoning.md").write_text("\n".join(md) + "\n")

    # also a flat CSV
    with (dst / "gemini_predictions.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip", "true_cell", "gemini_cell", "correct", "miss_cells", "reasoning"])
        for r in rows:
            reason = r["raw"].split("Reasoning:", 1)[-1].split("Ball position:")[0].strip()
            wr.writerow([r["clip"], r["gt_cell"], r["pred_cell"] or "",
                         bool(r["grade"].get("correct")), r["grade"].get("cell_dist"),
                         reason.replace("\n", " ")])
    shutil.copy(JSONL, dst / "raw_model_output.jsonl")
    return {"n": n, "exact": exact, "within1": within1, "mean_d": mean_d}


def write_readme(stats):
    txt = f"""# Spot the Invisible Ball — can Gemini infer where a hidden soccer ball is?

## The one-sentence version
We show Gemini 2.5 Pro a soccer clip where the ball disappears, and we test whether it can
figure out **where the ball ends up** — then we check its answer against the true location.

## How each clip works (5 seconds long)
1. **Seconds 0–1: the ball is visible.** A **red circle** is drawn around the ball on the
   very first frame so you know exactly where the play starts.
2. **Seconds 1–5: the ball is invisible.** It's digitally removed. Players keep moving and
   passing as normal, but you can no longer see the ball.
3. Gemini watches the whole clip and must answer: **which grid cell holds the ball in the
   final frame?**

A yellow grid is drawn on every clip: **rows A–F** (top to bottom) and **columns 1–16**
(left to right). An answer is a cell like `D5` = row D, column 5.

## How we know the right answer (ground truth)
For every clip we render the exact same play **twice** — once with the ball and once
without. Comparing the two frames pinpoints the ball's exact pixel, with zero guesswork.
That pixel is converted to a grid cell = the ground truth. (See `3_ground_truth/`, and the
`ball_location_images/` folder shows a red crosshair on the ball in each final frame.)

## What's in this folder
- **`1_ball_hidden_clips/`** — the 30 clips exactly as Gemini saw them (red circle on the
  first frame, ball invisible after 1 second). `clipNN_hidden.mov`.
- **`2_ball_visible_clips/`** — the same 30 plays with the ball **visible the whole time**,
  so you can watch where it actually goes. This is the visual answer key. `clipNN_visible.mov`.
- **`3_ground_truth/`** — the true final-frame cell for each clip (`ground_truth.md`/`.csv`)
  plus `ball_location_images/` marking the ball on each final frame.
- **`4_gemini_predictions/`** — Gemini's answer for every clip, whether it was right, how
  far off it was, and **its full written reasoning** (`gemini_predictions_and_reasoning.md`).

## The result
Gemini 2.5 Pro on all {stats['n']} clips:
- **Exact cell: {stats['exact']}/{stats['n']} ({round(100*stats['exact']/stats['n'])}%)**
- Within one cell of correct: {stats['within1']}/{stats['n']} ({round(100*stats['within1']/stats['n'])}%)
- Average miss: {stats['mean_d']} cells

**Takeaway:** tracking a ball you can no longer see is hard. Gemini usually gets the rough
area right (it lands in or next to the correct cell about a third of the time) but rarely
pins the exact cell. It tends to guess the vertical middle of the frame and to overshoot how
far the ball travelled. Full per-clip reasoning is in `4_gemini_predictions/`.
"""
    (OUT / "README.md").write_text(txt)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    d1 = OUT / "1_ball_hidden_clips"
    d2 = OUT / "2_ball_visible_clips"
    d3 = OUT / "3_ground_truth"
    d4 = OUT / "4_gemini_predictions"
    for d in (d1, d2, d3, d4):
        d.mkdir(parents=True, exist_ok=True)
    overlay = build_grid()
    print("building visible-ball clips...")
    make_visible_clips(d2, overlay)
    print("copying hidden clips + ground truth...")
    copy_hidden_clips(d1)
    copy_ground_truth(d3)
    print("building predictions + reasoning...")
    stats = build_predictions(d4)
    write_readme(stats)
    print(f"\nDONE -> {OUT}")


if __name__ == "__main__":
    main()
