"""verify_batch.py — check the finished GEN2 batch against the spec before shipping it.

Everything here is a property the batch is REQUIRED to have, checked against the files on
disk rather than against the pipeline's own logs — a generator that silently wrote 19 clips
still prints a happy summary, so the count has to be taken from the filesystem.

Checks:
  * the expected number of situations per class, each present in BOTH visibility folders
  * every clip is exactly 5.0 s / 50 frames at 10 fps
  * ground_truth.csv covers every clip on disk, and every row has a clip on disk
  * player-delta end counts are 5 each of 6 / 10 / 12 / 16, and match the CSV
  * player-delta hides NOBODY — all 22 players in play, players_hidden == 0. The original
    clips hit their count by hiding players and were rejected for it, so this guards
    against the approach creeping back in.
  * player-delta clips are not one-sided (both teams on screen at the end)
  * player-delta source matches are distinct — 20 clips from 20 different matches

Run:  python3 verify_batch.py
Exit code is non-zero if anything fails, so it can gate a commit.
"""
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from gen2_lib import ALL_SLOTS, CACHE, CLIP_FRAMES, FPS, HERE

# Situations per class. The three match-situation classes are 10 each (the batch was cut
# from 20 to 10 on request); player-delta stays at 20, because it has to cover four
# separate end-counts at 5 apiece. Each situation ships in BOTH visibility variants, so
# the clip count per class is twice the number here.
WANT_PER_CLASS = {
    "01_headers": 10,
    "02_corner_kicks": 10,
    "03_goalkeeper_throws": 10,
    "04_player_delta": 20,
}
CLASSES = list(WANT_PER_CLASS)
VARIANTS = ["full_visibility", "split_1s_4s"]
END_COUNTS = {6: 5, 10: 5, 12: 5, 16: 5}

fails, warns = [], []


def fail(msg):
    fails.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def probe_frames(path):
    """Frame count and duration straight from the container."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None, None
    s = json.loads(out.stdout)["streams"][0]
    return int(s.get("nb_read_frames", 0)), float(s.get("duration", 0) or 0)


def check_counts_and_pairing():
    print("\n[1] clip counts and visibility pairing")
    for cls in CLASSES:
        d = HERE / cls
        if not d.is_dir():
            fail(f"{cls}: folder missing")
            continue
        names = {}
        for v in VARIANTS:
            names[v] = {p.stem for p in (d / v).glob("*.mov")} if (d / v).is_dir() else set()
        want = WANT_PER_CLASS[cls]
        n_full, n_split = len(names["full_visibility"]), len(names["split_1s_4s"])
        if n_full != want or n_split != want:
            fail(f"{cls}: {n_full} full_visibility + {n_split} split_1s_4s "
                 f"(want {want} each)")
        else:
            ok(f"{cls}: {n_full} + {n_split} = {n_full + n_split} clips")
        only_full = names["full_visibility"] - names["split_1s_4s"]
        only_split = names["split_1s_4s"] - names["full_visibility"]
        if only_full or only_split:
            fail(f"{cls}: unpaired clips — full-only {sorted(only_full)}, "
                 f"split-only {sorted(only_split)}")


def check_durations():
    print(f"\n[2] every clip is exactly {CLIP_FRAMES} frames ({CLIP_FRAMES / FPS:.1f}s)")
    bad, total = [], 0
    for cls in CLASSES:
        for v in VARIANTS:
            for p in sorted((HERE / cls / v).glob("*.mov")):
                total += 1
                n, dur = probe_frames(p)
                if n is None:
                    bad.append(f"{p.relative_to(HERE)}: unreadable")
                elif n != CLIP_FRAMES:
                    bad.append(f"{p.relative_to(HERE)}: {n} frames")
    if bad:
        for b in bad[:10]:
            fail(b)
        if len(bad) > 10:
            fail(f"...and {len(bad) - 10} more")
    elif total:
        ok(f"all {total} clips are {CLIP_FRAMES} frames")


def _rows(path):
    if not path.exists():
        return None
    with path.open() as fh:
        return list(csv.DictReader(fh))


def check_ground_truth():
    print("\n[3] ground truth covers every clip, and vice versa")
    for cls in CLASSES:
        rows = _rows(HERE / cls / "ground_truth.csv")
        if rows is None:
            fail(f"{cls}: ground_truth.csv missing")
            continue
        gt = {r["clip"] for r in rows}
        disk = {p.stem for p in (HERE / cls / "full_visibility").glob("*.mov")}
        if gt != disk:
            fail(f"{cls}: GT/disk mismatch — in GT only {sorted(gt - disk)}, "
                 f"on disk only {sorted(disk - gt)}")
            continue
        blank = [r["clip"] for r in rows
                 if not r.get("ball_final_cell") or not r.get("ball_start_cell")]
        if blank:
            fail(f"{cls}: rows with no ball cell: {blank}")
        else:
            ok(f"{cls}: {len(rows)} rows, all with start and final cells")


def check_player_delta():
    print("\n[4] player-delta end counts, team balance, source diversity")
    rows = _rows(HERE / "04_player_delta" / "ground_truth.csv")
    if rows is None:
        fail("04_player_delta/ground_truth.csv missing")
        return
    got = Counter(int(r["end_count"]) for r in rows)
    if got != Counter(END_COUNTS):
        fail(f"end-count spread is {dict(got)}, want {END_COUNTS}")
    else:
        ok(f"end counts: {dict(sorted(got.items()))}")

    mismatched = [r["clip"] for r in rows
                  if int(r["players_in_frame_last"]) != int(r["end_count"])]
    if mismatched:
        fail(f"players_in_frame_last != end_count for {mismatched}")
    else:
        ok("every clip ends with exactly its target number of players")

    # Nobody may be hidden. This is the whole point of the rebuild: the original clips
    # hit their count by hiding players and were rejected for it, so a non-zero here means
    # the hiding approach has crept back in.
    hidden = [r["clip"] for r in rows if int(r.get("players_hidden", 0)) != 0]
    if hidden:
        fail(f"players were hidden in {hidden} — the count must come from framing")
    elif all(int(r["players_in_play"]) == 22 for r in rows):
        ok("all 22 players in play and none hidden, in every clip")
    else:
        fail("players_in_play is not 22 in every clip")

    picks_path = CACHE / "player_delta" / "picks_natural.json"
    counts_dir = CACHE / "player_delta" / "counts"
    if not (picks_path.exists() and counts_dir.is_dir()):
        warns.append("player-delta cache missing; skipped balance/diversity checks")
        print("  warn  cache missing, skipped balance and diversity checks")
        return

    picks = json.loads(picks_path.read_text())
    sources = Counter(p["match"] for p in picks)
    if len(sources) != len(picks):
        dupes = {k: n for k, n in sources.items() if n > 1}
        fail(f"source matches reused (clips are near-duplicates): {dupes}")
    else:
        ok(f"{len(picks)} clips from {len(sources)} distinct matches")

    # Team balance is not enforced by the picker any more — nobody is hidden, so the
    # frame contains whoever the camera saw. It is still worth reporting: a clip with one
    # team alone would not read as football, and would mean the window choice is bad.
    import numpy as np
    one_sided = []
    for p in picks:
        f = counts_dir / f"{p['match']}.npz"
        if not f.exists():
            continue
        on = np.load(f)["on"][:, p["end"] - 1]
        left = int(sum(1 for i, s in enumerate(ALL_SLOTS) if on[i] and s.startswith("L")))
        right = int(sum(1 for i, s in enumerate(ALL_SLOTS) if on[i] and s.startswith("R")))
        if min(left, right) == 0:
            one_sided.append(f"{p['match']}@end{p['end_count']} ({left}L/{right}R)")
    if one_sided:
        fail(f"one team missing at the final frame: {one_sided}")
    else:
        ok("both teams on screen at the final frame in all clips")


def main():
    print(f"verifying GEN2 batch in {HERE}")
    check_counts_and_pairing()
    check_durations()
    check_ground_truth()
    check_player_delta()
    total = sum(len(list((HERE / c / v).glob('*.mov'))) for c in CLASSES for v in VARIANTS)
    want_total = sum(WANT_PER_CLASS.values()) * 2
    print(f"\n{'=' * 60}")
    print(f"{total} clips on disk (want {want_total})")
    for w in warns:
        print(f"WARN: {w}")
    if fails:
        print(f"FAILED: {len(fails)} problem(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
