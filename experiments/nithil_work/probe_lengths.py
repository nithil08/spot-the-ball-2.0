import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib
from lib import use_bundle, make_env, run_clip

LEVELS = ["academy_counterattack_easy", "academy_counterattack_hard",
          "academy_pass_and_shoot_with_keeper", "5_vs_5", "1_vs_1_easy",
          "academy_run_to_score_with_keeper"]
SEEDS = [3, 11, 23, 47]

use_bundle("ball_tiny")
for lvl in LEVELS:
    for s in SEEDS:
        work = Path(f"results/_probe/{lvl}_{s}")
        env = make_env(lvl, s, work, write_video=False, hud=False)
        info = run_clip(env, steps=60, dump_name="probe")
        env.close()
        print(f"{lvl:45s} seed={s:4d} steps_run={info['steps_run']:3d} done={info['done']}")
