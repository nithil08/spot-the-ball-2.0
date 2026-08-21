"""calibrate_inframe.py — predict "is this player on camera?" from world coordinates.

WHY THIS EXISTS
  GEN2's player-delta clips must hit an exact number of players in frame on the final
  frame, and — per the brief — WITHOUT hiding anybody. Every one of the 22 players is
  rendered; the count has to come from picking windows the camera naturally frames that
  way. To pick such windows we must know the on-screen count for any frame of any match.

  Measuring it by rendering costs 23 passes per match (a plate with everyone hidden, plus
  one solo pass per player). That is ~50 s per match and only ever covered 120 frames of
  25 matches. The sweep cache holds 602 matches x 1100 frames of world coordinates and no
  pixels, and that is the pool the window search needs to run over.

  So: fit the camera's field of view once, against the rendered ground truth we already
  paid for, then apply it to the whole cache for free.

THE MODEL
  The engine's camera tracks the ball, lagging it slightly, so a player is on screen when
  it sits inside a fixed box around the SMOOTHED ball position. Camera centre is an
  exponential moving average of the ball (alpha fitted); the box is fitted separately for
  each edge because the render is a 1280x480 crop out of 1280x720 that takes 60 px off the
  top and 180 off the bottom, making the vertical field of view asymmetric.

  Fitted against the solo-probe labels in _cache/player_delta/onscreen.json — tens of
  thousands of (player, frame) pairs whose true on-screen status was measured in pixels.

Run:  python3 calibrate_inframe.py            # fit, report accuracy, save the model
Out:  _cache/inframe_model.json
"""
import json
import sys

import numpy as np

from gen2_lib import CACHE, run_log

MAP = CACHE / "player_delta" / "onscreen.json"
POS = CACHE / "player_delta" / "positions.npz"
MODEL = CACHE / "inframe_model.json"


def gather_positions():
    """Replay each probed match to recover ball and player world coordinates.

    The probe saved only the on-screen map, not the positions it corresponded to. The
    matches are deterministic, so replaying the same (level, seed) reproduces exactly the
    play that was probed.
    """
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle("gen2")
    import gen_player_delta as PD

    maps = json.loads(MAP.read_text())
    usable = {k: v for k, v in maps.items() if v}
    for _n, spec in PD.bases():
        write_scenario(spec, force=True)

    out = {}
    for key, rec in usable.items():
        n_frames = len(rec["on"][0])
        log = run_log(rec["level"], rec["seed"], n_frames)
        out[f"{key}/ball"] = log["ball"][:n_frames]
        out[f"{key}/left"] = log["left"][:n_frames]
        out[f"{key}/right"] = log["right"][:n_frames]
        print(f"  positions {key}: {n_frames} frames", flush=True)
    np.savez_compressed(POS, **out)
    print(f"saved -> {POS}")


def build_dataset():
    maps = json.loads(MAP.read_text())
    usable = {k: v for k, v in maps.items() if v}
    z = np.load(POS)
    balls, players, labels = [], [], []
    for key, rec in usable.items():
        on = np.array(rec["on"], dtype=bool)            # (22, T)
        ball = z[f"{key}/ball"]
        pl = np.concatenate([z[f"{key}/left"], z[f"{key}/right"]], axis=1)  # (T, 22, 2)
        T = min(on.shape[1], len(ball), len(pl))
        balls.append(ball[:T])
        players.append(pl[:T])
        labels.append(on[:, :T].T)                      # (T, 22)
    return balls, players, labels


def camera_centre(ball, alpha):
    """Exponential moving average of the ball — the camera's lagging track."""
    c = np.empty_like(ball[:, :2])
    acc = ball[0, :2].copy()
    for t in range(len(ball)):
        acc = alpha * ball[t, :2] + (1 - alpha) * acc
        c[t] = acc
    return c


def evaluate(balls, players, labels, alpha, xr, ylo, yhi):
    tp = fp = fn = tn = 0
    for ball, pl, lab in zip(balls, players, labels):
        c = camera_centre(ball, alpha)
        dx = pl[:, :, 0] - c[:, None, 0]
        dy = pl[:, :, 1] - c[:, None, 1]
        pred = (np.abs(dx) <= xr) & (dy >= ylo) & (dy <= yhi)
        tp += int((pred & lab).sum())
        fp += int((pred & ~lab).sum())
        fn += int((~pred & lab).sum())
        tn += int((~pred & ~lab).sum())
    total = tp + fp + fn + tn
    return (tp + tn) / total, tp, fp, fn, tn


def fit():
    balls, players, labels = build_dataset()
    n = sum(l.size for l in labels)
    print(f"fitting on {n} (player, frame) labelled pairs from {len(labels)} matches")

    best = None
    # Coarse sweep, then refine around the winner. The ranges bracket the pitch: x is
    # +-1 over 105 m and y is +-0.42 over 68 m, so a broadcast frame is a fraction of that.
    for alpha in (0.15, 0.25, 0.35, 0.5, 0.7, 1.0):
        for xr in np.arange(0.16, 0.46, 0.02):
            for ylo in np.arange(-0.30, -0.04, 0.02):
                for yhi in np.arange(0.04, 0.30, 0.02):
                    acc, *_ = evaluate(balls, players, labels, alpha, xr, ylo, yhi)
                    if best is None or acc > best[0]:
                        best = (acc, alpha, float(xr), float(ylo), float(yhi))
    acc, alpha, xr, ylo, yhi = best
    a2, tp, fp, fn, tn = evaluate(balls, players, labels, alpha, xr, ylo, yhi)
    print(f"\nbest: alpha={alpha}  |dx|<={xr:.2f}  {ylo:.2f}<=dy<={yhi:.2f}")
    print(f"accuracy {acc*100:.2f}%   tp={tp} tn={tn} fp={fp} fn={fn}")

    # What actually matters is the COUNT per frame, not each player individually —
    # errors that cancel out across a frame do not move the count.
    errs = []
    for ball, pl, lab in zip(balls, players, labels):
        c = camera_centre(ball, alpha)
        dx = pl[:, :, 0] - c[:, None, 0]
        dy = pl[:, :, 1] - c[:, None, 1]
        pred = (np.abs(dx) <= xr) & (dy >= ylo) & (dy <= yhi)
        errs.append(pred.sum(axis=1).astype(int) - lab.sum(axis=1).astype(int))
    errs = np.concatenate(errs)
    exact = float((errs == 0).mean())
    within1 = float((np.abs(errs) <= 1).mean())
    print(f"per-frame COUNT: exact {exact*100:.1f}%, within +-1 {within1*100:.1f}%, "
          f"bias {errs.mean():+.2f}, sd {errs.std():.2f}")

    MODEL.write_text(json.dumps({
        "alpha": alpha, "x_range": xr, "y_lo": ylo, "y_hi": yhi,
        "player_accuracy": acc, "count_exact": exact, "count_within1": within1,
        "count_bias": float(errs.mean()), "count_sd": float(errs.std()),
        "n_pairs": int(n),
    }, indent=2))
    print(f"\n-> {MODEL}")
    return exact


def in_frame_mask(ball, left, right, model):
    """Apply the fitted model. ball (T,3); left/right (T,11,2) -> (T,22) bool."""
    pl = np.concatenate([left, right], axis=1)
    c = camera_centre(ball, model["alpha"])
    dx = pl[:, :, 0] - c[:, None, 0]
    dy = pl[:, :, 1] - c[:, None, 1]
    return ((np.abs(dx) <= model["x_range"])
            & (dy >= model["y_lo"]) & (dy <= model["y_hi"]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "positions":
        gather_positions()
    else:
        if not POS.exists():
            gather_positions()
        fit()
