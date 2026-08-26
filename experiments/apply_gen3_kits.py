"""apply_gen3_kits.py — the GEN3 palette: team A BLUE, team B RED, and each KEEPER in
his own team's hue.

WHAT CHANGED FROM GEN2, AND WHY
  GEN2 put both keepers in one yellow kit. That makes "goalkeeper" obvious but "which
  side is he on" a puzzle, and the user asked for the opposite trade: "for the color of
  the goalie jerseys make it similar to whichever team they are on so it is easy to
  identify instead of hard."

  So each keeper now wears a pale version of his own team's colour — sky blue behind the
  blue team, coral behind the red team. Same hue family as the outfield kit, so team
  membership reads instantly; much lighter, so the keeper is still separable from the ten
  outfielders in front of him.

WHY THIS NEEDED AN ENGINE PATCH
  team.cpp:108 hard-coded ONE global texture for both keepers:

      kitFilename = "media/objects/players/textures/goalie_kit.png";

  There is no per-team goalkeeper kit in stock gfootball, so no amount of texture editing
  can give the two keepers different kits. The patch (see engine_patches/) makes the
  keeper's kit derive from his own team's kit_url when GFOOTBALL_TEAM_GK_KITS is set:

      kitFilename = GetTeamData()->GetKitUrl() + "_gk_kit.png";

  It is opt-in because IMG_LoadBmp dereferences the decoded surface without a null check,
  so requesting a file a bundle does not carry is a segfault rather than an error. Only
  the gen3 bundles carry the two "<kit_url>_gk_kit.bmp" files, and gen3_lib sets the env
  var on every engine pass.

  The global goalie_kit.bmp is still repainted (to the blue keeper's colours) so that an
  unset env var degrades to something sane instead of to stock white.

WHY THE COLOURS ARE WRITTEN BACKWARDS
  The engine reads the kit .bmp as **BGR**, not RGB — measured, not guessed: painting one
  kit pure RGB red and another pure RGB green rendered a BLUE team and a GREEN team, and
  green is the fixed point of an R<->B swap, which pins the channel order exactly. So:
  pick the colour you want ON SCREEN, then write it reversed into the file.

  ".png" in the engine source is a convention only; IMG_LoadBmp rewrites the extension to
  ".bmp" before touching the disk, which is why these are all .bmp files.

Run:  python3 apply_gen3_kits.py
Out:  asset_bundles/gen3/  and  asset_bundles/gen3_ball_invisible/
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BUNDLES = HERE / "asset_bundles"

# (source bundle, gen3 bundle). The invisible-ball twin must differ from its visible
# partner ONLY in the ball, or the visible-vs-invisible pixel diff stops being the ball.
PAIRS = [("noname", "gen3"), ("noname_ball_invisible", "gen3_ball_invisible")]

TEAMS = "databases/default/images_teams/primeradivision"
KIT_LEFT = f"{TEAMS}/fcbarcelona_kit_02.bmp"          # left team  (teamdata.cpp:139)
KIT_RIGHT = f"{TEAMS}/realmadrid_kit_02.bmp"          # right team (teamdata.cpp:147)
GK_LEFT = f"{TEAMS}/fcbarcelona_gk_kit.bmp"           # new: read via the engine patch
GK_RIGHT = f"{TEAMS}/realmadrid_gk_kit.bmp"
GOALIE_GLOBAL = "media/objects/players/textures/goalie_kit.bmp"

# Colours as they should appear ON SCREEN (RGB).
#
# The two keeper colours are deliberately the same hue as their outfield kit and far
# lighter, rather than a different hue. Hue carries team identity at a glance; lightness
# carries the goalkeeper/outfield distinction, and it survives being read at the small
# on-screen size a keeper occupies in a tracking shot. Both are well clear of the teal
# pitch (~RGB 48,108,84), which has almost no red.
BLUE = (28, 72, 224)          # team A outfield
RED = (214, 32, 32)           # team B outfield
GK_BLUE = (125, 195, 255)     # team A keeper — pale sky blue
GK_RED = (255, 145, 130)      # team B keeper — pale coral

# (path in bundle, colour, source texture, label). `source` is the stock texture whose UV
# layout the file must keep: the keeper kits are painted from the GOALKEEPER texture, not
# from a team shirt, so sleeves/gloves/collar land where the goalie mesh expects them.
KITS = [
    (KIT_LEFT,       BLUE,    None,          "team A outfield"),
    (KIT_RIGHT,      RED,     None,          "team B outfield"),
    (GK_LEFT,        GK_BLUE, GOALIE_GLOBAL, "team A keeper"),
    (GK_RIGHT,       GK_RED,  GOALIE_GLOBAL, "team B keeper"),
    (GOALIE_GLOBAL,  GK_BLUE, None,          "fallback keeper"),
]


def recolour(path: Path, screen_rgb, source: Path = None):
    """Repaint every non-background texel in `screen_rgb`, keeping relative luminance.

    The stock texture is kept once as *_gen3_backup.bmp and is always the source, so this
    is idempotent — re-running never compounds the recolour. When `source` is given the
    file is being CREATED from another texture (the keeper kits), so the backup dance
    applies to the source instead.
    """
    if source is None:
        backup = path.with_name(path.stem + "_gen3_backup.bmp")
        if not backup.exists():
            shutil.copyfile(path, backup)
        src = backup
    else:
        src = source

    im = np.array(Image.open(src).convert("RGB"))

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
    path.parent.mkdir(parents=True, exist_ok=True)
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
        for rel, colour, source, what in KITS:
            n = recolour(dst / rel, colour, dst / source if source else None)
            print(f"    {what:16s} -> RGB{colour}  ({n} texels)")
        made.append(dst_name)
    print("\nbundles:", ", ".join(made))
    print("NOTE: the per-team keeper kits only take effect with GFOOTBALL_TEAM_GK_KITS=1 "
          "and the patched libgame.dylib.")


if __name__ == "__main__":
    build()
