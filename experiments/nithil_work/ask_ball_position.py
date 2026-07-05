"""Ask the "where is the ball in the final frame" question against the 6
UPDATED scenarios' full_clip_without_ball.mp4, through all 3 local VLMs.

Samples 6 evenly-spaced frames per clip (always including the true final
frame, since the question is specifically about it) and sends them as a
multi-image prompt to each Ollama model, mirroring vlm_eval.py's call_ollama
convention (same 3 registered models: qwen2.5vl:7b, llava:7b, minicpm-v).

Writes a markdown table to results/ball_position_answers.md.
"""

import json
import re
import subprocess
from pathlib import Path

import requests

UPDATED = Path(r"C:\Users\nithi\OneDrive\Desktop\Nithil Research\UPDATED")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
FRAME_DIR = Path(__file__).parent / "_ball_position_frames"

MODELS = {
    "Qwen2.5-VL-7B": "qwen2.5vl:7b",
    "LLaVA-7B": "llava:7b",
    "MiniCPM-V": "minicpm-v",
}

GRID_LEGEND = (
    "The grid overlaid on the video has rows labeled A-L (top to bottom) and "
    "columns labeled 1-32 (left to right).\n\n"
)
QUESTION = (
    "You are watching a short clip from a soccer match. The ball has been "
    "digitally removed from every frame. Infer where the ball is located in "
    "the final frame of the clip. Respond in exactly this format:\n"
    "Reasoning: <one or two sentences>\n"
    "Ball position: <grid row #, grid colum#>"
)
PROMPT = GRID_LEGEND + QUESTION

N_FRAMES = 1  # just the final frame -- 6-image calls to these local 7B models timed out (>300s)


def extract_frames(clip: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of",
         "default=nokey=1:noprint_wrappers=1", str(clip)],
        capture_output=True, text=True,
    )
    total = int(nb.stdout.strip())
    idxs = sorted(set(round(i * (total - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)))
    paths = []
    for i in idxs:
        png = out_dir / f"f{i:03d}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
             "-vf", f"select=eq(n\\,{i})", "-vframes", "1", str(png)],
            check=True,
        )
        paths.append(png)
    return paths


def call_ollama(model_id: str, prompt: str, images: list[Path]) -> str:
    import base64
    imgs_b64 = [base64.b64encode(p.read_bytes()).decode() for p in images]
    payload = {
        "model": model_id,
        "prompt": prompt,
        "images": imgs_b64,
        "stream": False,
        "options": {"num_predict": 256},
    }
    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def parse_answer(text: str) -> tuple:
    reasoning_m = re.search(r"Reasoning:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    position_m = re.search(r"Ball position:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    reasoning = reasoning_m.group(1).strip() if reasoning_m else text.replace("\n", " ")[:200]
    position = position_m.group(1).strip() if position_m else "(not in expected format)"
    return reasoning, position


def main():
    rows = []
    for i in range(1, 7):
        sdir = UPDATED / f"scenario_{i}"
        clip = sdir / "full_clip_without_ball.mp4"
        if not clip.exists():
            print(f"scenario_{i}: MISSING {clip}")
            continue
        frames = extract_frames(clip, FRAME_DIR / f"scenario_{i}")
        print(f"scenario_{i}: sampled {len(frames)} frames")

        for display_name, model_id in MODELS.items():
            print(f"  {display_name} ...", end=" ", flush=True)
            try:
                raw = call_ollama(model_id, PROMPT, frames)
            except Exception as e:
                raw = f"ERROR: {e}"
            reasoning, position = parse_answer(raw)
            print(f"-> {position}")
            rows.append({
                "scenario": f"scenario_{i}",
                "model": display_name,
                "raw": raw,
                "reasoning": reasoning,
                "position": position,
            })
            with (OUT_DIR / "ball_position_raw.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1]) + "\n")

    # markdown table for copy-paste
    lines = ["| Scenario | Model | Reasoning | Ball position |",
             "|---|---|---|---|"]
    for r in rows:
        reasoning_escaped = r["reasoning"].replace("|", "\\|")
        position_escaped = r["position"].replace("|", "\\|")
        lines.append(f"| {r['scenario']} | {r['model']} | {reasoning_escaped} | {position_escaped} |")
    table = "\n".join(lines)
    (OUT_DIR / "ball_position_answers.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\nwrote {OUT_DIR / 'ball_position_answers.md'}")


if __name__ == "__main__":
    main()
