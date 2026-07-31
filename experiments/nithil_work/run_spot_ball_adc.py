"""run_spot_ball_adc.py — run all 30 split clips through Gemini and score vs final-frame GT.

Uses the EXACT prompt below (verbatim). Each clip is sent as inline video bytes (clips are
~1 MB, well under the request limit). The 16x6 grid is burned into the video, so the model
reads the grid visually.

Auth (in priority order):
  * Vertex AI via ADC — set GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION,
    default us-central1). Requires ADC:  bash <(curl -sSL \
    https://storage.googleapis.com/cloud-samples-data/adc/setup_adc.sh)
  * AI Studio — else falls back to GEMINI_API_KEY / GOOGLE_API_KEY (repo-root .env).

Usage:
    export GOOGLE_CLOUD_PROJECT=your-project        # Vertex/ADC
    python3 run_spot_ball_adc.py                      # all 30 clips, gemini-2.5-pro
    python3 run_spot_ball_adc.py --n 3                # smoke test on 3 clips
    python3 run_spot_ball_adc.py --model gemini-2.5-flash

Outputs -> results/30_1s_visible_4s_invisible/eval_runs/:
    run_<model>_<ts>.jsonl   one row per clip (raw answer + grade)
    run_<model>_<ts>.md      readable table + summary
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BATCH = HERE / "results" / "30_1s_visible_4s_invisible"
RUNS = BATCH / "eval_runs"
GLOB = "clip*_1s_visible_then_4s_invisible.mov"
COLS, ROWS = 16, 6

PROMPT = (
    "A yellow grid is overlaid on the video: its ROWS are labelled A-F from top to "
    "bottom (6 rows) and its COLUMNS are labelled 1-16 from left to right (16 columns). "
    "A cell is named <row-letter><column-number>, e.g. D5 is row D, column 5.\n\n"
    "You are watching a short clip from a soccer match. The ball has been digitally "
    "removed from every frame. Infer where the ball is located in the final frame of "
    "the clip. Respond in exactly this format:\n"
    "Reasoning: <one or two sentences>\n"
    "Ball position: <row-letter><column-number>"
)

# ── .env (for GEMINI_API_KEY fallback) ───────────────────────────────────────────
ENV_FILE = REPO / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ── ground truth ─────────────────────────────────────────────────────────────────
def load_gt():
    gt = {}
    with (BATCH / "ground_truth.csv").open() as f:
        for r in csv.DictReader(f):
            gt[int(r["clip"])] = {"cell": r["ball_cell"].strip().upper(),
                                  "row": r["grid_row"].strip().upper(),
                                  "col": int(r["grid_col"])}
    return gt


# ── answer parsing ───────────────────────────────────────────────────────────────
def parse_cell(text):
    """Pull (cell,row_letter,col) from the model answer. Accepts 'D5', 'D, 5', '4, 5'
    (numeric row 1-6 -> letter), 'row D column 5', etc."""
    m = re.search(r"Ball position:\s*(.+)", text, re.IGNORECASE)
    seg = m.group(1) if m else text          # first line after the label only
    # 1) letter row (A-F) followed by the column number: 'D5', 'D, 5', 'row D column 5'
    lm = re.search(r"([A-Fa-f])\D{0,12}?(\d{1,2})", seg)
    if lm:
        row, col = lm.group(1).upper(), int(lm.group(2))
        if 1 <= col <= COLS:
            return f"{row}{col}", row, col
    # 2) numeric row (1-6) + column: '4, 5'
    nums = [int(x) for x in re.findall(r"\d{1,2}", seg)]
    if len(nums) >= 2 and 1 <= nums[0] <= ROWS and 1 <= nums[1] <= COLS:
        row = chr(ord("A") + nums[0] - 1)
        return f"{row}{nums[1]}", row, nums[1]
    return None, None, None


def grade(prow, pcol, gt):
    if prow is None or gt is None:
        return {"correct": None, "row_err": None, "col_err": None, "cell_dist": None}
    re_ = abs((ord(prow) - ord("A")) - (ord(gt["row"]) - ord("A")))
    ce_ = abs(pcol - gt["col"])
    return {"correct": (prow == gt["row"] and pcol == gt["col"]),
            "row_err": re_, "col_err": ce_,
            "cell_dist": round((re_**2 + ce_**2) ** 0.5, 2)}


# ── Gemini client (Vertex/ADC or AI Studio) ──────────────────────────────────────
def get_client():
    from google import genai
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        loc = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        print(f"[auth] Vertex AI via ADC — project={project} location={loc}")
        return genai.Client(vertexai=True, project=project, location=loc)
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("ERROR: set GOOGLE_CLOUD_PROJECT (Vertex/ADC) or GEMINI_API_KEY (AI Studio).")
    print("[auth] AI Studio via GEMINI_API_KEY")
    return genai.Client(api_key=key)


def ask(client, model, clip):
    from google.genai import types
    part = types.Part.from_bytes(data=clip.read_bytes(), mime_type="video/quicktime")
    resp = client.models.generate_content(model=model, contents=[part, PROMPT])
    return (resp.text or "").strip()


def clip_num(p):
    return int(re.search(r"clip_?(\d+)", p.name).group(1))


# ── summary / report ─────────────────────────────────────────────────────────────
def summarize(rows):
    graded = [r for r in rows if r["grade"].get("correct") is not None]
    n = len(graded)
    exact = sum(1 for r in graded if r["grade"]["correct"])
    within1 = sum(1 for r in graded if r["grade"]["row_err"] <= 1 and r["grade"]["col_err"] <= 1)
    dists = [r["grade"]["cell_dist"] for r in graded if r["grade"].get("cell_dist") is not None]
    return {"n": n, "exact": exact, "within1": within1,
            "mean_dist": round(sum(dists) / len(dists), 2) if dists else None,
            "unparsed": sum(1 for r in rows if r["pred_cell"] is None)}


def write_report(md_path, model, rows):
    s = summarize(rows)
    lines = [f"# Spot-the-ball (final frame) — {model} on 30_1s_visible_4s_invisible (16x6)", "",
             f"- exact cell: **{s['exact']}/{s['n']}**"
             + (f" ({100*s['exact']/s['n']:.0f}%)" if s["n"] else ""),
             f"- within 1 cell: {s['within1']}/{s['n']}",
             f"- mean cell distance: {s['mean_dist']}",
             f"- unparsed: {s['unparsed']}", "",
             "| clip | ground truth | predicted | correct | cell dist |",
             "|---|---|---|---|---|"]
    for r in rows:
        g = r["grade"]
        ok = "✓" if g.get("correct") else ("✗" if g.get("correct") is False else "?")
        lines.append(f"| {r['clip']:02d} | {r['gt_cell']} | {r['pred_cell']} | {ok} | {g.get('cell_dist')} |")
    md_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--n", type=int, default=None, help="only first N clips (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gt = load_gt()
    clips = sorted(BATCH.glob(GLOB), key=clip_num)
    if args.n:
        clips = clips[: args.n]
    if args.dry_run:
        print(f"[dry-run] {len(clips)} clips, model={args.model}\n--- prompt ---\n{PROMPT}")
        print(f"sample gt clip1: {gt.get(1)}")
        return

    RUNS.mkdir(parents=True, exist_ok=True)
    client = get_client()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace("/", "_")
    jsonl_path = RUNS / f"run_{tag}_{ts}.jsonl"
    md_path = RUNS / f"run_{tag}_{ts}.md"
    print(f"=== {args.model} on {len(clips)} clips ===\nOutput: {jsonl_path}\n")

    rows = []
    with jsonl_path.open("w") as fout:
        for clip in clips:
            num = clip_num(clip)
            g = gt.get(num)
            print(f"clip{num:02d} (gt={g['cell'] if g else '?'}) ...", end=" ", flush=True)
            try:
                raw = ask(client, args.model, clip)
            except Exception as e:
                raw = f"ERROR: {e}"
            cell, prow, pcol = parse_cell(raw)
            gd = grade(prow, pcol, g)
            mark = "✓" if gd.get("correct") else ("✗" if gd.get("correct") is False else "?")
            print(f"pred={cell}  {mark}  dist={gd.get('cell_dist')}")
            row = {"clip": num, "file": clip.name, "model": args.model,
                   "gt_cell": g["cell"] if g else None, "pred_cell": cell,
                   "grade": gd, "raw": raw}
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    write_report(md_path, args.model, rows)
    s = summarize(rows)
    print("\n=== SUMMARY ===")
    if s["n"]:
        print(f"  exact cell:    {s['exact']}/{s['n']} ({100*s['exact']/s['n']:.0f}%)")
        print(f"  within 1 cell: {s['within1']}/{s['n']}")
        print(f"  mean dist:     {s['mean_dist']}")
    print(f"  unparsed:      {s['unparsed']}")
    print(f"\nWrote {jsonl_path}\n      {md_path}")


if __name__ == "__main__":
    main()
