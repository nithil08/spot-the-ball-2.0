"""Create asset-swap bundles for visual counterfactuals.

Each bundle is a self-contained copy of the engine's data/ directory, with one
targeted edit. Point GFOOTBALL_DATA_DIR at a bundle to use it at runtime.
"""

import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Resolve the engine's stock data/ dir lazily (works for vendored or pip-installed).
sys.path.insert(0, str(REPO / "experiments"))
from lib import gfootball_engine_data_dir  # noqa: E402

OUT = REPO / "experiments/asset_bundles"


def fresh_copy(name):
    src = gfootball_engine_data_dir()
    dst = OUT / name / "data"
    if dst.exists():
        shutil.rmtree(dst.parent)
    shutil.copytree(src, dst, symlinks=False)
    return dst


def patch_ball_transparent(data_dir):
    ase = data_dir / "media/objects/balls/generic.ase"
    text = ase.read_text()
    text = text.replace("*MATERIAL_TRANSPARENCY 0.0000",
                        "*MATERIAL_TRANSPARENCY 1.0000")
    ase.write_text(text)


def patch_ball_tiny(data_dir, scale=0.05):
    ase = data_dir / "media/objects/balls/generic.ase"
    text = ase.read_text()
    # MESH_VERTEX lines look like:
    #   *MESH_VERTEX    0     -0.0150  -0.0210  -0.1080
    pat = re.compile(
        r"(\*MESH_VERTEX\s+\d+\s+)"
        r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    )

    def repl(m):
        head, x, y, z = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        return f"{head}{x*scale:.6f}\t{y*scale:.6f}\t{z*scale:.6f}"

    new_text, n = pat.subn(repl, text)
    if n == 0:
        raise RuntimeError("No MESH_VERTEX matched; check ase format")
    ase.write_text(new_text)
    print(f"  scaled {n} vertices by {scale}")


def patch_uniform_jerseys(data_dir):
    """Make both teams' kit textures identical (use left-team kit for both)."""
    kits = data_dir / "databases/default/images_teams/primeradivision"
    left = kits / "fcbarcelona_kit_02.bmp"
    right = kits / "realmadrid_kit_02.bmp"
    shutil.copy(left, right)


def main():
    try:
        src = gfootball_engine_data_dir()
    except RuntimeError as e:
        sys.exit(str(e))
    if not src.exists():
        sys.exit(f"engine data dir not found: {src}")
    OUT.mkdir(parents=True, exist_ok=True)

    print("[default] copying stock data")
    fresh_copy("default")

    print("[ball_transparent] copying + patching")
    d = fresh_copy("ball_transparent")
    patch_ball_transparent(d)

    print("[ball_tiny] copying + patching")
    d = fresh_copy("ball_tiny")
    patch_ball_tiny(d)

    print("[uniform_jerseys] copying + patching")
    d = fresh_copy("uniform_jerseys")
    patch_uniform_jerseys(d)

    print("done. bundles in", OUT)


if __name__ == "__main__":
    main()
