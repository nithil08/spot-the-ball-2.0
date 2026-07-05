"""Build two asset bundles that suppress player name labels.

The C++ engine renders player names using the font at:
  GFOOTBALL_DATA_DIR/media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf
Replacing that file with a blank font (all glyph outlines emptied) makes
all in-game text render as invisible — no recompile needed.

Bundles created:
  noname              — default visuals, blank font  (use for visible-ball renders)
  noname_ball_invisible — default visuals, ball at 0.5% scale, blank font
                         (use for hidden-ball renders; ball is sub-pixel)

Usage:
  python build_noname_bundles.py
"""

import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
from lib import gfootball_engine_data_dir  # noqa: E402

BUNDLES = REPO / "experiments/asset_bundles"
FONT_REL = Path("media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf")


def create_blank_font(source: Path, dest: Path):
    """Copy source TTF with ALL cmap entries remapped to the space glyph.

    Every character the engine tries to render maps to U+0020 (space), which
    has advance width but zero ink — so all text becomes invisible without
    breaking the font's validity (SDL_ttf can still load and measure it).
    """
    from fontTools import ttLib
    font = ttLib.TTFont(str(source))

    # Find the space glyph name from cmap
    best_cmap = font.getBestCmap()
    space_name = best_cmap.get(0x0020)
    if space_name is None:
        # Fallback: find any glyph named 'space'
        glyph_order = font.getGlyphOrder()
        space_name = next((g for g in glyph_order if g.lower() == "space"), glyph_order[0])

    # Remap every entry in every cmap subtable to space
    for table in font["cmap"].tables:
        for cp in list(table.cmap.keys()):
            table.cmap[cp] = space_name

    dest.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(dest))
    print(f"  blank font written -> {dest.relative_to(REPO)} (all chars -> space glyph '{space_name}')")


def make_noname(src_bundle: str, dest_bundle: str, ball_scale: float = None):
    """Copy src_bundle -> dest_bundle and replace its font with a blank one.

    Optionally also rescale the ball vertices (for the invisible-ball variant).
    """
    src_data = BUNDLES / src_bundle / "data"
    dst_data = BUNDLES / dest_bundle / "data"
    if not src_data.exists():
        sys.exit(f"source bundle not found: {src_data}")

    if dst_data.parent.exists():
        shutil.rmtree(dst_data.parent)
    print(f"  copying {src_bundle} -> {dest_bundle} ...")
    shutil.copytree(src_data, dst_data)

    # Locate the font from the gfootball source tree (installed at a known path).
    engine_font = Path("/Users/nithilbalamurugan/gfootball_src/third_party/gfootball_engine/fonts/AlegreyaSansSC-ExtraBold.ttf")
    if not engine_font.exists():
        try:
            engine_font = gfootball_engine_data_dir() / FONT_REL
        except RuntimeError:
            pass
    if not engine_font.exists():
        sys.exit(f"Cannot find source font: {engine_font}")

    create_blank_font(engine_font, dst_data / FONT_REL)

    if ball_scale is not None:
        _rescale_ball(dst_data, ball_scale)

    print(f"  done -> {dst_data.relative_to(REPO)}")


def _rescale_ball(data_dir: Path, scale: float):
    ase = data_dir / "media/objects/balls/generic.ase"
    text = ase.read_text()
    pat = re.compile(
        r"(\*MESH_VERTEX\s+\d+\s+)"
        r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    )

    def repl(m):
        head = m.group(1)
        x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
        return f"{head}{x*scale:.8f}\t{y*scale:.8f}\t{z*scale:.8f}"

    new_text, n = pat.subn(repl, text)
    if n == 0:
        raise RuntimeError("No MESH_VERTEX matched in generic.ase")
    ase.write_text(new_text)
    print(f"  rescaled {n} ball vertices by {scale} (effectively sub-pixel)")


def main():
    BUNDLES.mkdir(parents=True, exist_ok=True)

    print("\n[noname] building from default bundle (blank font, normal ball)...")
    make_noname("default", "noname")

    print("\n[noname_ball_invisible] building from default bundle (blank font, 5% ball)...")
    make_noname("default", "noname_ball_invisible", ball_scale=0.05)

    print("\nAll done. New bundles:")
    print(f"  {BUNDLES / 'noname'}")
    print(f"  {BUNDLES / 'noname_ball_invisible'}")


if __name__ == "__main__":
    main()
