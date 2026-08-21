"""apply_gen2_kits.py — the GEN2 kit palette: team A BLUE, team B RED, goalkeepers YELLOW.

Standing instruction (2026-08-21): "one team will be blue and other team will be red and
the goalie will be a distinct color something like yellow or green."

WHY THE COLOURS ARE WRITTEN BACKWARDS
  The engine loads the kit .bmp and interprets its bytes as **BGR**, not RGB. This is not
  a guess — it was measured: painting fcbarcelona_kit_02.bmp pure RGB red (255,0,0) and
  realmadrid_kit_02.bmp pure RGB green (0,255,0) rendered a BLUE team and a GREEN team.
  Green is the fixed point of an R<->B swap, so only the red one flipped, which pins the
  channel order exactly. It also explains the stock look: the Barcelona texture is yellow
  (238,240,55) and rendered CYAN (55,240,238); the Madrid texture is navy (2,12,73) and
  rendered DARK RED (73,12,2). Both are the byte-reverse.

  So: pick the colour you want ON SCREEN, then write it reversed into the file.

WHICH FILE IS WHICH
  team.cpp:104  outfield -> "<kit_url>_kit_0<N>.png"   (kit_url from teamdata.cpp:139/147:
                fcbarcelona = LEFT team, realmadrid = RIGHT team; the bundles ship kit 02)
  team.cpp:108  goalkeeper -> "media/objects/players/textures/goalie_kit.png"
                — ONE global texture, so BOTH keepers wear it. That is what makes yellow
                read as "goalkeeper" rather than "a third team".

SHADING
  The kit texture is painted flat in the target hue but scaled by the original pixel's
  relative luminance, so shirt/shorts/socks stay separable and the folds survive. Scene
  lighting is applied by the renderer on top of this.

Yellow is chosen for the keepers over green: green collides with the pitch (~RGB 48,108,84)
and with the legacy green referee kit. See also apply_bw_referee_kit.py — officials are
rendered at 2% scale (effectively invisible), so they do not compete for colour space.

Run:  python3 apply_gen2_kits.py
Out:  asset_bundles/gen2/  and  asset_bundles/gen2_ball_invisible/
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BUNDLES = HERE / "asset_bundles"

# (source bundle, gen2 bundle). The invisible-ball twin must differ from its visible
# partner ONLY in the ball, or the visible-vs-invisible pixel diff stops being the ball.
PAIRS = [("noname", "gen2"), ("noname_ball_invisible", "gen2_ball_invisible")]

TEAM_LEFT = "databases/default/images_teams/primeradivision/fcbarcelona_kit_02.bmp"
TEAM_RIGHT = "databases/default/images_teams/primeradivision/realmadrid_kit_02.bmp"
GOALIE = "media/objects/players/textures/goalie_kit.bmp"

# Colours as they should appear ON SCREEN (RGB).
BLUE = (28, 72, 224)
RED = (214, 32, 32)
YELLOW = (252, 226, 38)

KITS = [(TEAM_LEFT, BLUE, "left team"),
        (TEAM_RIGHT, RED, "right team"),
        (GOALIE, YELLOW, "goalkeepers")]


def recolour(path: Path, screen_rgb):
    """Repaint every non-background texel in `screen_rgb`, keeping relative luminance.

    The stock texture is kept once as *_gen2_backup.bmp and is always the source, so this
    is idempotent — re-running never compounds the recolour.
    """
    backup = path.with_name(path.stem + "_gen2_backup.bmp")
    src = backup if backup.exists() else path
    im = np.array(Image.open(src).convert("RGB"))
    if not backup.exists():
        Image.fromarray(im).save(backup)

    # Exact black is the texture's unused UV background; leave it alone.
    kit = im.sum(axis=2) > 0
    lum = im[..., 0] * 0.299 + im[..., 1] * 0.587 + im[..., 2] * 0.114
    # Reference = a bright-but-not-outlier kit texel, so most of the kit lands near the
    # target colour and only genuinely dark texels (shorts, shadowed folds) go darker.
    ref = np.percentile(lum[kit], 80) or 1.0
    scale = np.clip(lum / ref, 0.0, 1.25)[..., None]

    out = np.zeros_like(im)
    target = np.array(screen_rgb, dtype=float)
    painted = np.clip(target[None, None, :] * scale, 0, 255).astype(np.uint8)
    out[kit] = painted[kit]
    # Engine reads the file as BGR -> reverse so `screen_rgb` is what actually renders.
    Image.fromarray(out[..., ::-1]).save(path)
    return int(kit.sum())


def build():
    made = []
    for src_name, dst_name in PAIRS:
        src = BUNDLES / src_name / "data"
        if not src.exists():
            sys.exit(f"source bundle missing: {src}  (run build_noname_bundles.py first)")
        dst = BUNDLES / dst_name / "data"
        if dst.parent.exists():
            shutil.rmtree(dst.parent)
        shutil.copytree(src, dst, symlinks=False)
        print(f"[{dst_name}] copied from {src_name}")
        for rel, colour, what in KITS:
            n = recolour(dst / rel, colour)
            print(f"    {what:12s} -> RGB{colour}  ({n} texels)")
        made.append(dst_name)
    print("\nbundles:", ", ".join(made))


if __name__ == "__main__":
    build()
