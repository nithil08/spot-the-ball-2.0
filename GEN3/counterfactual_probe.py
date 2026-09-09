"""counterfactual_probe.py — can we actually play out "what if this player were not
there", and from where?

THE DISTINCTION THIS EXISTS TO MAKE
  GFOOTBALL_HIDE_SLOTS looks like a counterfactual knob and is not one. It sets a
  player's render scale to zero: he vanishes from the picture and goes on playing —
  marking, intercepting, scoring. It is exactly the right tool for the player-count
  benchmark (the play must NOT change when you hide someone, or the hidden-and-diff
  count would be measuring a different match) and exactly the wrong tool for a
  counterfactual. verify_reproducible.py's pass C is the proof: hiding two players
  leaves the observation log bit-identical.

  A real counterfactual has to remove the player from the SCENARIO, so the engine builds
  a team without him.

THE THREE QUESTIONS
  1. ROSTER    Does the engine accept an asymmetric team (11 v 10)? If not, the whole
               approach is dead and we need a different intervention.
  2. DETERMINISM  Is the counterfactual run itself reproducible? A counterfactual you
               cannot replay is not evidence of anything.
  3. ENTRY POINT  A counterfactual necessarily starts where the intervention happens.
               Removing a player from the scenario removes him from kick-off, so the
               divergence begins at frame 0 — the clip's own window (say frames
               2310-2435) is then a different match entirely, not "the same clip minus a
               player".

               The alternative is to restart mid-play from a SNAPSHOT: read every
               position out of the factual log at the clip's start frame and emit a new
               scenario there. That is what `snapshot` measures, and it is the load-
               bearing result, because the scenario builder takes only AddPlayer(x, y,
               role) and SetBallPosition(x, y). There is no velocity, no ball height, no
               possession, no player facing, no animation phase. So the question is not
               whether the snapshot restart is lossy — it must be — but HOW lossy, i.e.
               how long a snapshot-restarted match stays close to the factual one.

USAGE
  python3 counterfactual_probe.py roster       # 11v10 accepted? deterministic?
  python3 counterfactual_probe.py snapshot     # mid-play restart fidelity
  python3 counterfactual_probe.py all
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))

import gen3_lib as G  # noqa: E402

# clip_09: countL, seed 349, window 2310-2435, a corner at 12 players in frame.
CLIP = "clip_09"
SHAPE, SEED = "countL", 349
START, END = 2310, 2435
DROP = 7                 # right-team outfield slot to remove; 0 is the keeper
PROBE_STEPS = 300        # enough to characterise divergence without a full replay


def _spec_without(shape, team, idx):
    """The shape's scenario with one player removed from one side."""
    from scenario_factory import ScenarioSpec
    base = G.shape_spec(shape)
    left = list(base.left)
    right = list(base.right)
    if team == "right":
        right.pop(idx)
    else:
        left.pop(idx)
    return ScenarioSpec(
        name=f"cf_{shape}_{team}{idx}", ball=base.ball, left=left, right=right,
        game_duration=base.game_duration, deterministic=base.deterministic,
        offsides=base.offsides, end_on_score=base.end_on_score,
        end_on_out=base.end_on_out,
        end_on_possession_change=base.end_on_possession_change,
        left_difficulty=base.left_difficulty, right_difficulty=base.right_difficulty,
    )


def cmd_roster(argv):
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(G.BUNDLE_VIS)
    write_scenario(G.shape_spec(SHAPE), force=True)
    cf = _spec_without(SHAPE, "right", DROP)
    write_scenario(cf, force=True)
    print(f"counterfactual scenario: {cf.name}  "
          f"left={len(cf.left)} right={len(cf.right)}")

    fact = G.run_log(f"g3_{SHAPE}", SEED, PROBE_STEPS)
    print(f"factual      left_team {fact['left'].shape}  right_team {fact['right'].shape}")

    a = G.run_log(cf.name, SEED, PROBE_STEPS)
    print(f"counterfactual left_team {a['left'].shape}  right_team {a['right'].shape}")

    b = G.run_log(cf.name, SEED, PROBE_STEPS)
    same = all(np.array_equal(a[k], b[k]) for k in
               ("ball", "left", "right", "owned_team", "owned_player", "game_mode", "score"))
    print(f"counterfactual replays bit-identically: {same}")

    d = np.abs(fact["ball"][:, :2] - a["ball"][:, :2]).sum(axis=1)
    first = int(np.argmax(d > 1e-6)) if (d > 1e-6).any() else None
    print(f"ball diverges from factual at step: {first}")
    print(f"mean |ball| separation over {PROBE_STEPS} steps: {d.mean():.4f}")
    return 0


def _snapshot_spec(log, frame, drop=None):
    """A scenario that starts from the factual state at `frame`.

    Positions only — that is the whole point of the measurement. Roles are taken from the
    shape so each slot keeps its job. Coordinates are clamped to the pitch because the
    factual log can put a player a hair outside it and the scenario validator refuses
    that; the clamp is reported so it is never silently absorbed.
    """
    from scenario_factory import PlayerSpec, ScenarioSpec
    base = G.shape_spec(SHAPE)
    clamped = [0]

    def mk(pos, proto):
        x = float(np.clip(pos[0], -1.0, 1.0))
        y = float(np.clip(pos[1], -0.42, 0.42))
        if x != float(pos[0]) or y != float(pos[1]):
            clamped[0] += 1
        return PlayerSpec(x, y, proto.role)

    left = [mk(log["left"][frame][i], base.left[i]) for i in range(len(base.left))]
    right = [mk(log["right"][frame][i], base.right[i]) for i in range(len(base.right))]
    if drop is not None:
        right.pop(drop)
    ball = log["ball"][frame]
    spec = ScenarioSpec(
        name=f"snap_{SHAPE}_{frame}" + (f"_d{drop}" if drop is not None else ""),
        ball=(float(np.clip(ball[0], -1.0, 1.0)), float(np.clip(ball[1], -0.42, 0.42))),
        left=left, right=right,
        game_duration=base.game_duration, deterministic=base.deterministic,
        offsides=base.offsides, end_on_score=base.end_on_score,
        end_on_out=base.end_on_out,
        end_on_possession_change=base.end_on_possession_change,
        left_difficulty=base.left_difficulty, right_difficulty=base.right_difficulty,
    )
    return spec, clamped[0], ball


def cmd_snapshot(argv):
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(G.BUNDLE_VIS)

    sweep = G.SWEEP / f"{G.tag(SHAPE, SEED)}.npz"
    if not sweep.exists():
        print(f"no cached sweep log at {sweep}")
        return 1
    log = dict(np.load(sweep))
    print(f"factual log: {log['ball'].shape[0]} frames (from the sweep cache)")

    spec, clamped, ball0 = _snapshot_spec(log, START)
    write_scenario(spec, force=True)
    print(f"snapshot at frame {START}: {spec.name}  clamped {clamped} coords")
    print(f"  ball logged (x,y,z) = {np.round(ball0, 4).tolist()}  "
          f"-> scenario sets (x,y) only, z={float(ball0[2]):.3f} is LOST")

    n = END - START
    snap = G.run_log(spec.name, SEED, n)
    fact_ball = log["ball"][START:START + n, :2]
    snap_ball = snap["ball"][:n, :2]
    d = np.linalg.norm(fact_ball - snap_ball, axis=1)

    print("\nball separation, snapshot-restart vs factual (world units, pitch is 2.0 long):")
    for f in (0, 5, 10, 25, 50, 75, 100, n - 1):
        if f < len(d):
            print(f"  frame {START + f:>5} (+{f:>3}, {f / G.FPS:4.1f}s):  {d[f]:.4f}")
    print(f"\n  mean {d.mean():.4f}   max {d.max():.4f}")

    fl = log["left"][START:START + n]
    sl = snap["left"][:n]
    pd = np.linalg.norm(fl - sl, axis=2).mean(axis=1)
    print(f"  mean left-team player displacement: frame 0 {pd[0]:.4f}, "
          f"+25 {pd[min(25, n - 1)]:.4f}, +{n - 1} {pd[-1]:.4f}")

    om = (snap["owned_team"][:n] == log["owned_team"][START:START + n]).mean()
    print(f"  possession agreement over the window: {om * 100:.1f}%")
    return 0


def cmd_state(argv):
    """Does the engine's own state save/restore give EXACT mid-play branching?

    `snapshot` establishes that rebuilding a scenario from logged positions loses the
    play within half a second, because AddPlayer carries no velocity, no ball height and
    no possession. But gfootball has a real serialiser — FootballEnvCore.get_state pickles
    the Python-side state and hands it to the C++ engine's own get_state, which returns a
    blob of the ENTIRE simulation. If restoring that blob into a fresh env and stepping on
    reproduces the factual continuation bit-for-bit, then mid-play branching is exact and
    the snapshot limitation is an artefact of the wrong tool, not a property of the engine.

    Two things are measured, because they are different claims:
      restore-exactness  restore at START, step to END, compare against the factual tail
      cost              a restore costs one deserialise instead of START replayed steps
    """
    import time

    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(G.BUNDLE_VIS)
    write_scenario(G.shape_spec(SHAPE), force=True)
    level = f"g3_{SHAPE}"

    def obs_of(env):
        o = env.observation()
        o = o[0] if isinstance(o, list) else o
        return (np.asarray(o["ball"], dtype=np.float32).copy(),
                np.asarray(o["left_team"], dtype=np.float32).copy(),
                np.asarray(o["right_team"], dtype=np.float32).copy(),
                int(o["ball_owned_team"]), int(o["ball_owned_player"]))

    def step_log(env, n):
        out = []
        for _ in range(n):
            obs, _r, done, _i = env.step([])
            o = obs[0] if isinstance(obs, list) else obs
            out.append((np.asarray(o["ball"], dtype=np.float32).copy(),
                        np.asarray(o["left_team"], dtype=np.float32).copy(),
                        np.asarray(o["right_team"], dtype=np.float32).copy(),
                        int(o["ball_owned_team"]), int(o["ball_owned_player"])))
            if done:
                break
        return out

    G._prep("", (0.0, 0.0))
    t0 = time.time()
    env = G._env(level, SEED, G.CACHE / "_work_cf")
    env.render("rgb_array")
    env.reset()
    step_log(env, START)
    t_replay = time.time() - t0
    blob = env.get_state()
    print(f"state blob: {len(blob):,} bytes, captured at frame {START} "
          f"after {t_replay:.1f}s of replay")
    fact = step_log(env, END - START)
    env.close()

    G._prep("", (0.0, 0.0))
    t1 = time.time()
    env2 = G._env(level, SEED, G.CACHE / "_work_cf2")
    env2.render("rgb_array")
    env2.reset()
    env2.set_state(blob)
    t_restore = time.time() - t1
    at_restore = obs_of(env2)
    cf = step_log(env2, END - START)
    env2.close()

    # The sweep log appends AFTER each step, so log[i] is the state following step i+1.
    # Capturing the blob after START steps therefore lines up with log[START - 1], not
    # log[START]. Getting this index wrong reads as a restore failure when the restore is
    # in fact perfect, so it is spelled out rather than left to the reader.
    sweep_ball = np.load(G.SWEEP / f"{G.tag(SHAPE, SEED)}.npz")["ball"]
    ok_ball = np.array_equal(at_restore[0],
                             np.asarray(sweep_ball[START - 1], dtype=np.float32))
    print(f"restored observation matches the factual state at frame {START - 1}: {ok_ball}")

    n = min(len(fact), len(cf))
    exact = all(np.array_equal(fact[i][j], cf[i][j])
                for i in range(n) for j in (0, 1, 2)) and \
        all(fact[i][3:] == cf[i][3:] for i in range(n))
    print(f"continuation bit-identical over {n} frames: {exact}")
    if not exact:
        d = [float(np.abs(fact[i][0][:2] - cf[i][0][:2]).sum()) for i in range(n)]
        first = next((i for i, v in enumerate(d) if v > 1e-6), None)
        print(f"  first ball divergence at +{first}, mean {np.mean(d):.5f}")
    print(f"\ncost: replay to frame {START} = {t_replay:.1f}s, "
          f"restore from blob = {t_restore:.1f}s  "
          f"({t_replay / max(t_restore, 1e-6):.0f}x faster)")
    return 0


def cmd_graft(argv):
    """The combination that would make a mid-CLIP counterfactual possible: capture the
    factual state at the clip's start frame, then restore it into an env whose scenario
    is missing a player.

    If this works, "the same clip, minus this player" is directly constructible: identical
    football up to frame START, then a branch in which the player is gone. If it does not,
    the intervention can only happen at frame 0 and the counterfactual is a different
    match that happens to share a seed.

    Run in its own process on purpose. The engine deserialises this blob in C++ with no
    roster check visible on the Python side, so a size mismatch is as likely to abort the
    process as to raise.
    """
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(G.BUNDLE_VIS)
    write_scenario(G.shape_spec(SHAPE), force=True)
    cf = _spec_without(SHAPE, "right", DROP)
    write_scenario(cf, force=True)

    G._prep("", (0.0, 0.0))
    env = G._env(f"g3_{SHAPE}", SEED, G.CACHE / "_work_cf")
    env.render("rgb_array")
    env.reset()
    for _ in range(START):
        env.step([])
    blob = env.get_state()
    env.close()
    print(f"captured {len(blob):,} bytes from the 11 v 11 factual at frame {START}")

    G._prep("", (0.0, 0.0))
    env2 = G._env(cf.name, SEED, G.CACHE / "_work_cf2")
    env2.render("rgb_array")
    env2.reset()
    print(f"counterfactual env built: {cf.name} (11 v 10). attempting set_state...",
          flush=True)
    try:
        env2.set_state(blob)
    except Exception as e:
        print(f"REFUSED: {type(e).__name__}: {e}")
        return 0
    obs = env2.observation()
    o = obs[0] if isinstance(obs, list) else obs
    print(f"ACCEPTED: left_team {np.asarray(o['left_team']).shape} "
          f"right_team {np.asarray(o['right_team']).shape}")
    for _ in range(25):
        env2.step([])
    obs = env2.observation()
    o = obs[0] if isinstance(obs, list) else obs
    print(f"stepped 25 frames after the graft; ball {np.round(o['ball'][:2], 4).tolist()}")
    env2.close()
    return 0


CMDS = {"roster": cmd_roster, "snapshot": cmd_snapshot, "state": cmd_state,
        "graft": cmd_graft}


def cmd_all(argv):
    for name in ("roster", "snapshot", "state", "graft"):
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        CMDS[name](argv)
    return 0


CMDS["all"] = cmd_all

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(CMDS[sys.argv[1]](sys.argv[2:]))
