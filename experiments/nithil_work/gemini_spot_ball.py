"""gemini_spot_ball.py — Test Gemini on the "spot the ball" grid benchmark.

Sends each clip (full .mov video) to a Gemini model and asks which grid cell holds
the ball in the FINAL frame, then scores the answer against the exact pixel-diff
ground truth.

Two batches are supported:

  30_1s_visible_4s_invisible  (DEFAULT) — the real inference task.
      16x6 grid (rows A-F, cols 1-16). Ball is VISIBLE for the first ~1 s with a
      RED CIRCLE around its start, then INVISIBLE for the remaining 4 s. The model
      must infer where the (now invisible) ball is in the final frame. Ground truth
      = the final ball pixel from natural_30_5s_grid/ground_truth.csv (same seeds /
      windows / deterministic render), remapped into this 16x6 grid.

  natural_30_5s_grid — perception baseline. 32x12 grid, ball visible throughout;
      ground truth cell read straight from its own ground_truth.csv.

Usage:
    export GEMINI_API_KEY=...            # or put it in the repo-root .env
    python3 gemini_spot_ball.py                    # 30 clips, split batch, 2.5-pro
    python3 gemini_spot_ball.py --n 3              # smoke test on 3 clips
    python3 gemini_spot_ball.py --batch natural_30_5s_grid
    python3 gemini_spot_ball.py --model gemini-2.5-flash

Outputs (in results/<batch>/):
    gemini_<model>_<timestamp>.jsonl    one row per clip (raw answer + grade)
    gemini_<model>_<timestamp>.md       human-readable table + summary
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── paths / .env ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                       # spot-the-ball-2.0/
RESULTS = HERE / "results"

ENV_FILE = REPO / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

W, H = 1280, 480          # cropped render size shared by all batches


# ── grid helpers ────────────────────────────────────────────────────────────────
def cell_of(px, py, cols, rows):
    """Pixel -> (label, row_letter, col_number) for a cols x rows grid."""
    c = min(max(int(px // (W / cols)), 0), cols - 1)
    r = min(max(int(py // (H / rows)), 0), rows - 1)
    return f"{chr(ord('A') + r)}{c + 1}", chr(ord('A') + r), c + 1


def grid_legend(cols, rows):
    last = chr(ord("A") + rows - 1)
    return (
        f"A yellow coordinate grid is overlaid on the video. Its ROWS are labelled "
        f"A-{last} from top to bottom ({rows} rows) and its COLUMNS are labelled "
        f"1-{cols} from left to right ({cols} columns). A cell is named "
        f"<row-letter><column-number>, e.g. B2 is row B, column 2.\n\n"
    )


Q_SPLIT = (
    "This is a short clip from a soccer match with the grid described above. "
    "For the first second the ball is VISIBLE and a RED CIRCLE is drawn around it; "
    "after that the ball is removed from the video and stays INVISIBLE for the rest "
    "of the clip. Watch how play develops and infer which single grid cell contains "
    "the ball in the FINAL frame of the clip. Respond in EXACTLY this format and "
    "nothing else:\n"
    "Reasoning: <2-3 sentences explaining how you tracked the ball and inferred its "
    "final location>\n"
    "Ball position: <row-letter><column-number>"
)
Q_VISIBLE = (
    "This is a short clip from a soccer match with the grid described above. The ball "
    "is visible throughout. Look at the FINAL frame and determine which single grid "
    "cell the soccer ball is in. Respond in EXACTLY this format and nothing else:\n"
    "Reasoning: <one or two sentences>\n"
    "Ball position: <row-letter><column-number>"
)


# ── ground-truth loaders (clip number -> {cell,row,col}) ────────────────────────
def gt_own_csv(batch_dir, cols, rows):
    """Final-frame cell already stored in this batch's ground_truth.csv."""
    gt = {}
    with (batch_dir / "ground_truth.csv").open() as f:
        for r in csv.DictReader(f):
            gt[int(r["clip"])] = {
                "cell": r["ball_cell"].strip().upper(),
                "row": r["grid_row"].strip().upper(),
                "col": int(r["grid_col"]),
            }
    return gt


def gt_final_remapped(batch_dir, cols, rows):
    """Final ball pixel from natural_30_5s_grid, remapped into this grid."""
    src = RESULTS / "natural_30_5s_grid" / "ground_truth.csv"
    gt = {}
    with src.open() as f:
        for r in csv.DictReader(f):
            cell, row, col = cell_of(float(r["px"]), float(r["py"]), cols, rows)
            gt[int(r["clip"])] = {"cell": cell, "row": row, "col": col}
    return gt


# ── batch registry ──────────────────────────────────────────────────────────────
BATCHES = {
    "30_1s_visible_4s_invisible": {
        "cols": 16, "rows": 6,
        "glob": "clip*_1s_visible_then_4s_invisible.mov",
        "question": Q_SPLIT,
        "gt_loader": gt_final_remapped,
    },
    "natural_30_5s_grid": {
        "cols": 32, "rows": 12,
        "glob": "clip_*.mov",
        "question": Q_VISIBLE,
        "gt_loader": gt_own_csv,
    },
}


# ── answer parsing / scoring ────────────────────────────────────────────────────
CELL_RE = re.compile(r"([A-Za-z])\s*[-, ]?\s*(\d{1,2})")


def parse_cell(text, cols, rows):
    """Pull a valid cell out of the model answer; prefer the 'Ball position:' line."""
    m = re.search(r"Ball position:\s*(.+)", text, re.IGNORECASE)
    space = m.group(1) if m else text
    for cm in CELL_RE.finditer(space):
        row = cm.group(1).upper()
        col = int(cm.group(2))
        if 0 <= ord(row) - ord("A") < rows and 1 <= col <= cols:
            return f"{row}{col}", row, col
    return None, None, None


def grade(pred_row, pred_col, gt):
    if pred_row is None or gt is None:
        return {"correct": None, "row_err": None, "col_err": None, "cell_dist": None}
    row_err = abs((ord(pred_row) - ord("A")) - (ord(gt["row"]) - ord("A")))
    col_err = abs(pred_col - gt["col"])
    return {
        "correct": (pred_row == gt["row"] and pred_col == gt["col"]),
        "row_err": row_err,
        "col_err": col_err,
        "cell_dist": round((row_err**2 + col_err**2) ** 0.5, 2),
    }


# ── Gemini call ─────────────────────────────────────────────────────────────────
def get_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("ERROR: set GEMINI_API_KEY (env var or repo-root .env).")
    return genai.Client(api_key=key)


def ask_gemini(client, model, clip, prompt):
    """Upload the clip via the Files API, wait until ACTIVE, then query."""
    up = client.files.upload(file=str(clip))
    for _ in range(60):
        if up.state.name == "ACTIVE":
            break
        if up.state.name == "FAILED":
            raise RuntimeError(f"file processing FAILED for {clip.name}")
        time.sleep(2)
        up = client.files.get(name=up.name)
    else:
        raise RuntimeError(f"file not ACTIVE after 120s for {clip.name}")

    resp = client.models.generate_content(model=model, contents=[up, prompt])
    try:
        client.files.delete(name=up.name)
    except Exception:
        pass
    return (resp.text or "").strip()


def clip_num(path):
    return int(re.search(r"clip_?(\d+)", path.name).group(1))


# ── main ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="30_1s_visible_4s_invisible", choices=list(BATCHES))
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--n", type=int, default=None, help="only first N clips (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; print the prompt")
    args = ap.parse_args()

    cfg = BATCHES[args.batch]
    cols, rows = cfg["cols"], cfg["rows"]
    prompt = grid_legend(cols, rows) + cfg["question"]

    batch_dir = RESULTS / args.batch
    if not batch_dir.exists():
        sys.exit(f"ERROR: batch dir not found: {batch_dir}")
    gt = cfg["gt_loader"](batch_dir, cols, rows)
    clips = sorted(batch_dir.glob(cfg["glob"]), key=clip_num)
    if args.n:
        clips = clips[: args.n]

    if args.dry_run:
        print(f"[dry-run] batch={args.batch} grid={cols}x{rows} model={args.model} "
              f"clips={len(clips)}")
        print(f"sample gt (clip {clips and clip_num(clips[0])}): "
              f"{gt.get(clip_num(clips[0])) if clips else None}")
        print("--- prompt ---\n" + prompt)
        return

    client = get_client()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace("/", "_")
    jsonl_path = batch_dir / f"gemini_{tag}_{ts}.jsonl"
    md_path = batch_dir / f"gemini_{tag}_{ts}.md"

    print(f"=== Gemini spot-the-ball: {args.model} on {args.batch} "
          f"(grid {cols}x{rows}, {len(clips)} clips) ===\nOutput: {jsonl_path}\n")

    rows_out = []
    with jsonl_path.open("w") as fout:
        for clip in clips:
            num = clip_num(clip)
            g = gt.get(num)
            print(f"clip_{num:02d} (gt={g['cell'] if g else '?'}) ...", end=" ", flush=True)
            try:
                raw = ask_gemini(client, args.model, clip, prompt)
            except Exception as e:
                raw = f"ERROR: {e}"
            cell, prow, pcol = parse_cell(raw, cols, rows)
            grade_d = grade(prow, pcol, g)
            mark = ("✓" if grade_d.get("correct") else
                    ("✗" if grade_d.get("correct") is False else "?"))
            print(f"pred={cell}  {mark}  dist={grade_d.get('cell_dist')}")
            row = {
                "clip": num, "file": clip.name, "model": args.model, "batch": args.batch,
                "gt_cell": g["cell"] if g else None,
                "pred_cell": cell, "grade": grade_d, "raw": raw,
            }
            rows_out.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    write_report(md_path, args, cols, rows, rows_out)
    print_summary(rows_out)
    print(f"\nWrote {jsonl_path}\n      {md_path}")


def summarize(rows):
    graded = [r for r in rows if r["grade"].get("correct") is not None]
    n = len(graded)
    exact = sum(1 for r in graded if r["grade"]["correct"])
    dists = [r["grade"]["cell_dist"] for r in graded if r["grade"].get("cell_dist") is not None]
    within1 = sum(1 for r in graded
                  if r["grade"]["row_err"] <= 1 and r["grade"]["col_err"] <= 1)
    mean_dist = round(sum(dists) / len(dists), 2) if dists else None
    return {"n": n, "exact": exact, "within1": within1, "mean_dist": mean_dist,
            "unparsed": sum(1 for r in rows if r["pred_cell"] is None)}


def print_summary(rows):
    s = summarize(rows)
    print("\n=== SUMMARY ===")
    if s["n"]:
        print(f"  exact cell:      {s['exact']}/{s['n']}  ({100*s['exact']/s['n']:.0f}%)")
        print(f"  within 1 cell:   {s['within1']}/{s['n']}  ({100*s['within1']/s['n']:.0f}%)")
        print(f"  mean cell dist:  {s['mean_dist']}")
    print(f"  unparsed answers: {s['unparsed']}")


def write_report(md_path, args, cols, rows, rows_out):
    s = summarize(rows_out)
    lines = [
        f"# Gemini spot-the-ball — {args.model} on {args.batch} (grid {cols}x{rows})",
        "",
        f"- exact cell: **{s['exact']}/{s['n']}**"
        + (f" ({100*s['exact']/s['n']:.0f}%)" if s["n"] else ""),
        f"- within 1 cell: {s['within1']}/{s['n']}",
        f"- mean cell distance: {s['mean_dist']}",
        f"- unparsed: {s['unparsed']}",
        "",
        "| clip | ground truth | predicted | correct | cell dist |",
        "|---|---|---|---|---|",
    ]
    for r in rows_out:
        g = r["grade"]
        ok = "✓" if g.get("correct") else ("✗" if g.get("correct") is False else "?")
        lines.append(f"| {r['clip']:02d} | {r['gt_cell']} | {r['pred_cell']} "
                     f"| {ok} | {g.get('cell_dist')} |")
    md_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
