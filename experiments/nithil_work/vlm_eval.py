"""VLM evaluation script for the gfootball social-cognition benchmark.

Runs description, prediction, inference, hypothetical, and counterfactual
items through a set of models and records responses + ground-truth grades.

Usage:
    python vlm_eval.py --models claude qwen --categories description prediction
    python vlm_eval.py --models all --categories all --n 5

Outputs a JSONL file per run: results/<timestamp>_<model>.jsonl
Each line: {item_id, category, question, model_answer, ground_truth, correct}

API keys are read from a .env file in the repo root:
    ANTHROPIC_API_KEY=...
    OPENAI_API_KEY=...          # also used for OpenRouter
    OPENROUTER_API_KEY=...      # preferred for Qwen/open models
    TOGETHER_API_KEY=...        # alternative for open models
"""

import base64
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# ── load .env ──────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
ENV_FILE = REPO / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

STIMULI = REPO / "experiments/stimuli"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

# ── model registry ─────────────────────────────────────────────────────────────
# Each entry: (display_name, provider, model_id)
# 3 open-source vision models, all run locally via Ollama (no API keys needed).
MODELS = {
    "qwen":      ("Qwen2.5-VL-7B",  "ollama", "qwen2.5vl:7b"),
    "llava":     ("LLaVA-7B",       "ollama", "llava:7b"),
    "minicpm":   ("MiniCPM-V",      "ollama", "minicpm-v"),
}

CATEGORIES = ["description", "prediction", "inference", "hypothetical", "counterfactual"]


# ── image helpers ──────────────────────────────────────────────────────────────

def img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def load_png_as_b64(png_path: Path) -> str:
    return img_to_b64(png_path)


# ── providers ─────────────────────────────────────────────────────────────────

def call_anthropic(model_id: str, prompt: str, images: list[Path]) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content = []
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": load_png_as_b64(img),
            }
        })
    content.append({"type": "text", "text": prompt})
    msg = client.messages.create(
        model=model_id,
        max_tokens=256,
        messages=[{"role": "user", "content": content}],
    )
    return msg.content[0].text.strip()


def call_openai_compat(model_id: str, prompt: str, images: list[Path],
                       base_url: str = None, api_key_env: str = "OPENAI_API_KEY") -> str:
    import openai
    kwargs = {"api_key": os.environ[api_key_env]}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    content = []
    for img in images:
        b64 = load_png_as_b64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=256,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content.strip()


def call_ollama(model_id: str, prompt: str, images: list[Path]) -> str:
    """Call a local Ollama vision model via its REST API (no auth needed)."""
    import requests
    imgs_b64 = [load_png_as_b64(p) for p in images]
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


def call_model(provider: str, model_id: str, prompt: str, images: list[Path]) -> str:
    if provider == "anthropic":
        return call_anthropic(model_id, prompt, images)
    elif provider == "openai":
        return call_openai_compat(model_id, prompt, images,
                                  api_key_env="OPENAI_API_KEY")
    elif provider == "openrouter":
        return call_openai_compat(model_id, prompt, images,
                                  base_url="https://openrouter.ai/api/v1",
                                  api_key_env="OPENROUTER_API_KEY")
    elif provider == "together":
        return call_openai_compat(model_id, prompt, images,
                                  base_url="https://api.together.xyz/v1",
                                  api_key_env="TOGETHER_API_KEY")
    elif provider == "ollama":
        return call_ollama(model_id, prompt, images)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── probe builders ─────────────────────────────────────────────────────────────

def build_description_probes(item_dir: Path, meta: dict) -> list[dict]:
    img = item_dir / "frame.png"
    probes = []
    for q in meta["question_variants"]:
        probes.append({
            "question": q,
            "images": [img],
            "answer_format": meta["answer_format"],
            "ground_truth": meta["ground_truth"],
        })
    return probes


def build_prediction_probes(item_dir: Path, meta: dict) -> list[dict]:
    clip_frame = item_dir / "clip_frame.png"
    # Use the first frame of clip.mov as static stimulus for image VLMs
    # (extract it if not already done)
    if not clip_frame.exists():
        import subprocess
        clip_mov = item_dir / "clip.mov"
        if clip_mov.exists():
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(clip_mov), "-vframes", "1", str(clip_frame)
            ], check=False)
    if not clip_frame.exists():
        return []
    probes = []
    for q in meta["question_variants"]:
        probes.append({
            "question": q,
            "images": [clip_frame],
            "answer_format": meta["answer_format"],
            "ground_truth": meta["ground_truth"],
        })
    return probes


def build_inference_probes(item_dir: Path, meta: dict) -> list[dict]:
    probes = []
    for subtask, question in meta["questions"].items():
        # Extract last frame of each .mov as static stimulus
        mov = item_dir / f"{subtask}.mov"
        png = item_dir / f"{subtask}_frame.png"
        if not png.exists() and mov.exists():
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-sseof", "-0.5", "-i", str(mov),
                "-vframes", "1", str(png)
            ], check=False)
        if not png.exists():
            continue
        probes.append({
            "question": question,
            "images": [png],
            "answer_format": meta["answer_format"],
            "ground_truth": meta["ground_truth"],
            "subtask": subtask,
        })
    return probes


def build_hypothetical_probes(item_dir: Path, meta: dict) -> list[dict]:
    imgs = [item_dir / f"frame_{tag}.png"
            for tag in ("base", "a_add_attacker", "b_remove_defender",
                        "c_ball_forward", "d_no_change")
            if (item_dir / f"frame_{tag}.png").exists()]
    if not imgs:
        return []
    q = meta["questions"][0]
    return [{
        "question": q + "\nA=add attacker  B=remove defender  C=ball forward  D=no change\nANSWER: ",
        "images": imgs,
        "answer_format": meta["answer_format"],
        "ground_truth": meta["ground_truth"],
    }]


def build_counterfactual_probes(item_dir: Path, meta: dict) -> list[dict]:
    clip_frame = item_dir / "clip_frame.png"
    if not clip_frame.exists():
        import subprocess
        clip_mov = item_dir / "clip.mov"
        if clip_mov.exists():
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-sseof", "-0.5", "-i", str(clip_mov),
                "-vframes", "1", str(clip_frame)
            ], check=False)
    if not clip_frame.exists():
        return []
    probes = []
    for q in meta["questions"]:
        probes.append({
            "question": q + "\nANSWER: ",
            "images": [clip_frame],
            "answer_format": meta["answer_format"],
            "ground_truth": meta["ground_truth"],
        })
    return probes


PROBE_BUILDERS = {
    "description":    build_description_probes,
    "prediction":     build_prediction_probes,
    "inference":      build_inference_probes,
    "hypothetical":   build_hypothetical_probes,
    "counterfactual": build_counterfactual_probes,
}


# ── grading ───────────────────────────────────────────────────────────────────

def grade_description(answer: str, gt: dict, question: str) -> dict:
    """Extract number from answer, compare to ground truth."""
    import re
    nums = re.findall(r"\b(\d+)\b", answer)
    pred = int(nums[0]) if nums else None
    if "left" in question.lower():
        correct_val = gt["n_left"]
    elif "right" in question.lower():
        correct_val = gt["n_right"]
    else:
        correct_val = gt["n_total"]
    correct = (pred == correct_val) if pred is not None else False
    return {"predicted": pred, "expected": correct_val, "correct": correct}


def grade_prediction(answer: str, gt: dict, question: str) -> dict:
    if "score" in question.lower():
        pred_yes = any(w in answer.lower() for w in ["yes", "will score", "goal"])
        expected = gt["left_scored"] or gt["right_scored"]
        return {"predicted": pred_yes, "expected": expected,
                "correct": pred_yes == expected}
    return {"predicted": answer[:80], "expected": str(gt.get("ball_end_bin_xy")),
            "correct": None}


def grade_inference(answer: str, gt: dict, question: str, subtask: str = "") -> dict:
    if subtask == "region_occluded":
        import re
        nums = re.findall(r"\b(\d+)\b", answer)
        pred = int(nums[0]) if nums else None
        expected = gt.get("region_player_count_final")
        return {"predicted": pred, "expected": expected,
                "correct": pred == expected if pred is not None else False}
    return {"predicted": answer[:80], "expected": "coordinate/bin",
            "correct": None}


def grade_hypothetical(answer: str, gt: dict, question: str) -> dict:
    import re
    mc = re.search(r"\b([ABCD])\b", answer.upper())
    pred = mc.group(1) if mc else None
    expected_label = gt.get("best_intervention", "")
    expected = {"a_add_attacker": "A", "b_remove_defender": "B",
                "c_ball_forward": "C", "d_no_change": "D"}.get(expected_label)
    return {"predicted": pred, "expected": expected,
            "correct": pred == expected if (pred and expected) else None}


def grade_counterfactual(answer: str, gt: dict, question: str) -> dict:
    return {"predicted": answer[:120], "expected": gt.get("attacker_responsibility_rank"),
            "correct": None}


GRADERS = {
    "description":    grade_description,
    "prediction":     grade_prediction,
    "inference":      grade_inference,
    "hypothetical":   grade_hypothetical,
    "counterfactual": grade_counterfactual,
}


# ── main eval loop ─────────────────────────────────────────────────────────────

def run_eval(model_key: str, categories: list[str], n_per_cat: int, dry_run: bool):
    display_name, provider, model_id = MODELS[model_key]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS / f"{ts}_{model_key}.jsonl"
    print(f"\n=== {display_name} ({provider}) ===")
    print(f"Output: {out_path}\n")

    with open(out_path, "w", encoding="utf-8") as fout:
        for cat in categories:
            cat_dir = STIMULI / cat
            if not cat_dir.exists():
                print(f"  [{cat}] no stimuli dir, skipping")
                continue
            index_file = cat_dir / "_index.json"
            if not index_file.exists():
                print(f"  [{cat}] no _index.json, skipping")
                continue
            index = json.loads(index_file.read_text())[:n_per_cat]
            builder = PROBE_BUILDERS[cat]
            grader = GRADERS[cat]

            for item in index:
                item_id = item["id"]
                item_dir = cat_dir / item_id
                meta_file = item_dir / "meta.json"
                if not meta_file.exists():
                    continue
                meta = json.loads(meta_file.read_text())
                probes = builder(item_dir, meta)

                for probe in probes:
                    q = probe["question"]
                    imgs = [p for p in probe["images"] if p.exists()]
                    if not imgs:
                        print(f"    [{cat}/{item_id}] no image found, skipping")
                        continue

                    if dry_run:
                        answer = f"[DRY RUN — would send {len(imgs)} image(s) + prompt]"
                    else:
                        try:
                            answer = call_model(provider, model_id, q, imgs)
                            time.sleep(0.5)  # gentle rate limit
                        except Exception as e:
                            answer = f"ERROR: {e}"

                    grade_kwargs = {}
                    if cat == "inference":
                        grade_kwargs["subtask"] = probe.get("subtask", "")
                    grade = grader(answer, probe["ground_truth"], q, **grade_kwargs)

                    row = {
                        "model": model_key,
                        "model_id": model_id,
                        "category": cat,
                        "item_id": item_id,
                        "question": q[:200],
                        "answer": answer,
                        "grade": grade,
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()

                    status = "CORRECT" if grade.get("correct") else (
                        "WRONG" if grade.get("correct") is False else "UNGRADED")
                    print(f"  [{cat}/{item_id}] {status}  pred={grade.get('predicted')}  "
                          f"exp={grade.get('expected')}")

    return out_path


def print_summary(jsonl_path: Path):
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    by_cat = {}
    for r in rows:
        cat = r["category"]
        g = r["grade"]
        by_cat.setdefault(cat, {"correct": 0, "wrong": 0, "ungraded": 0})
        if g.get("correct") is True:
            by_cat[cat]["correct"] += 1
        elif g.get("correct") is False:
            by_cat[cat]["wrong"] += 1
        else:
            by_cat[cat]["ungraded"] += 1

    print("\n=== SUMMARY ===")
    for cat, counts in by_cat.items():
        total = counts["correct"] + counts["wrong"]
        acc = f"{counts['correct']}/{total}" if total else "n/a"
        print(f"  {cat:15s}  acc={acc:6s}  ungraded={counts['ungraded']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["claude"],
                    help=f"Model keys or 'all'. Available: {list(MODELS)}")
    ap.add_argument("--categories", nargs="+", default=["description"],
                    help=f"Categories or 'all'. Available: {CATEGORIES}")
    ap.add_argument("--n", type=int, default=4,
                    help="Max items per category per model")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip real API calls, print what would be sent")
    args = ap.parse_args()

    model_keys = list(MODELS.keys()) if "all" in args.models else args.models
    cats = CATEGORIES if "all" in args.categories else args.categories

    for bad in model_keys:
        if bad not in MODELS:
            print(f"Unknown model key: {bad}. Available: {list(MODELS)}")
            sys.exit(1)
    for bad in cats:
        if bad not in CATEGORIES:
            print(f"Unknown category: {bad}. Available: {CATEGORIES}")
            sys.exit(1)

    for mk in model_keys:
        out = run_eval(mk, cats, n_per_cat=args.n, dry_run=args.dry_run)
        print_summary(out)


if __name__ == "__main__":
    main()
