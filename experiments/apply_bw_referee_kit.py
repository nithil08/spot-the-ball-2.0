"""apply_bw_referee_kit.py — force the referee/officials into a BLACK-AND-WHITE
STRIPED jersey in every asset bundle.

The gfootball engine renders 1 referee + 2 linesmen wearing
media/objects/players/textures/referee_kit.bmp. Stock kit is solid green, which
looks like a mystery green team on the pitch. Per the user's standing instruction,
the referee must ALWAYS wear a black-and-white striped shirt.

This rewrites referee_kit.bmp in each bundle:
  * bright-green shirt panels -> vertical black/white stripes
  * darker-green shorts panels -> black
  * everything else (texture background) untouched
The original is backed up once as referee_kit_green_backup.bmp.

Re-run this after building any NEW asset bundle so the striped ref carries over.
"""
from pathlib import Path
import numpy as np
from PIL import Image

BUNDLES = Path(__file__).resolve().parent / "asset_bundles"
REL = "data/media/objects/players/textures/referee_kit.bmp"

STRIPE = 36                       # texture px per stripe (vertical stripes on body)
WHITE = (238, 238, 238)
BLACK = (20, 20, 20)


def restripe(path: Path):
    backup = path.with_name("referee_kit_green_backup.bmp")
    src = backup if backup.exists() else path      # always restripe from the green original
    im = np.array(Image.open(src).convert("RGB"))
    if not backup.exists():
        Image.fromarray(im).save(backup)           # preserve the stock green kit once

    R, G, B = im[:, :, 0].astype(int), im[:, :, 1].astype(int), im[:, :, 2].astype(int)
    green = (G - R > 20) & (G - B > 25) & (G > 55)  # any kit-green pixel
    shirt = green & (G >= 115)                       # bright green = shirt
    shorts = green & (G < 115)                       # darker green = shorts

    out = im.copy()
    xs = np.arange(im.shape[1])
    white_col = ((xs // STRIPE) % 2 == 0)            # per-column stripe choice
    stripe_rgb = np.where(white_col[None, :, None],
                          np.array(WHITE), np.array(BLACK)).astype(np.uint8)
    stripe_rgb = np.broadcast_to(stripe_rgb, im.shape)
    out[shirt] = stripe_rgb[shirt]
    out[shorts] = np.array(BLACK, np.uint8)
    Image.fromarray(out).save(path)
    return int(shirt.sum()), int(shorts.sum())


def main():
    paths = sorted(BUNDLES.glob(f"*/{REL}"))
    if not paths:
        print("no referee_kit.bmp found under", BUNDLES)
        return
    for p in paths:
        s, sh = restripe(p)
        print(f"  {p.parent.parent.parent.parent.parent.name:22s} striped shirt={s} shorts->black={sh}")
    print(f"\ndone — {len(paths)} bundles now have a black/white striped referee.")


if __name__ == "__main__":
    main()
