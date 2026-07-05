"""Send the ball-hidden 4s freeze-frame to qwen/llava/minicpm and grade
against the mechanical ground truth from gen_ball_hidden_4s.py.

Usage: python run_ball_hidden_4s.py <item_id>
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_eval import MODELS, call_model

RESULTS = Path(__file__).parent / "results" / "ball_hidden_4s"


def parse_bin(answer: str):
    m = re.search(r"col\D{0,5}(\d)\D{0,10}row\D{0,5}(\d)", answer, re.IGNORECASE | re.DOTALL)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    return None


def main():
    item_id = sys.argv[1] if len(sys.argv) > 1 else sorted(
        p.name for p in RESULTS.iterdir() if p.is_dir())[-1]
    item_dir = RESULTS / item_id
    meta = json.loads((item_dir / "meta.json").read_text())
    frame = item_dir / meta["stimulus"]
    gt = meta["ground_truth"]

    out_path = item_dir / "model_answers.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for key, (display, provider, model_id) in MODELS.items():
            print(f"\n=== {display} ===")
            for qkey in ("locate_now", "predict_next"):
                prompt = meta["questions"][qkey]
                try:
                    answer = call_model(provider, model_id, prompt, [frame])
                except Exception as e:
                    answer = f"ERROR: {e}"
                pred_bin = parse_bin(answer)
                expected_bin = gt["ball_bin_at_4s"] if qkey == "locate_now" else gt["ball_bin_at_5s"]
                correct = pred_bin == expected_bin if pred_bin else None
                row = {
                    "model": key, "question": qkey, "prompt": prompt,
                    "answer": answer, "predicted_bin": pred_bin,
                    "expected_bin": expected_bin, "correct": correct,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                print(f"  [{qkey}] pred={pred_bin} exp={expected_bin} correct={correct}")
                print(f"    answer: {answer[:200]}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
