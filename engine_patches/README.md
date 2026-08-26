# Engine patches

Everything that makes this benchmark's renders different from stock gfootball lives in
`gfootball_engine.patch`. Until 2026-08-23 it existed **only** as uncommitted working-tree
edits in `~/gfootball_src/` on the SFO-bound MacBook — a clone whose `origin` is
`google-research/football`, so a `git checkout .` would have destroyed it. It is now here.

The `football/` directory at the repo root is **pristine upstream** and does not contain
these changes. Do not confuse the two.

## What is in it

| File | What it does |
|---|---|
| `src/onthepitch/match.cpp` | `GFOOTBALL_CAM_OFFSET_X` / `_Y` — shifts the camera. Note this breaks the grid centre bias; centre bias is camera geometry, not scenario design. |
| `src/onthepitch/player/humanoid/humanoidbase.{cpp,hpp}` | Adds a `renderScale` on the humanoid base — the mechanism the other two shrink patches drive. |
| `src/onthepitch/player/playerofficial.cpp` | Officials rendered at `0.02f`, i.e. invisible referees, automatic for every clip. |
| `src/onthepitch/team.cpp` | `GFOOTBALL_HIDE_SLOTS` — a comma list of slot tokens whose players render at ~0 scale while staying in play. Also `GFOOTBALL_TEAM_GK_KITS` — derives each keeper's kit from his own team's `kit_url` (`<kit_url>_gk_kit.png`) instead of the one global `goalie_kit.png`, so the two keepers can wear their own team's colours. Opt-in: `IMG_LoadBmp` dereferences the decoded surface with no null check, so requesting a texture a bundle does not carry is a segfault. |
| `CMakeLists.txt` | `boost::system` is header-only in Boost 1.69+; find it separately. Portability fix, not behavioural. |
| `setup.py` | `makedirs(dest_dir)` before copying the prebuilt lib; `gym>=0.21.0`. Portability fixes. |

The four functional patches are plain `getenv`-driven C++ with no platform-specific code,
so they build on Linux as well as macOS.

## Rebuilding from scratch

```bash
git clone https://github.com/google-research/football.git gfootball_src
cd gfootball_src
git checkout $(cat /path/to/engine_patches/BASE_COMMIT)
git apply /path/to/engine_patches/gfootball_engine.patch
python3 -m pip install .
```

`BASE_COMMIT` pins the upstream commit the patch was cut against (`3d9e754`). Applying it
to a different upstream revision may not merge cleanly.

## Verifying the rebuild

The patches are invisible by construction, so a render that "looks fine" proves nothing —
check the mechanisms directly:

- **Referees** — should be absent from every frame with no env var set.
- **`GFOOTBALL_HIDE_SLOTS`** — set it and confirm the named players vanish while the match
  still plays out identically (ball trajectory unchanged).
- **Camera offset** — set `GFOOTBALL_CAM_OFFSET_X` and confirm the frame shifts.
- **Per-team keeper kits** — with `GFOOTBALL_TEAM_GK_KITS=1` and a `gen3` bundle, the two
  keepers must render in different colours. Measure rather than eyeball: hide all 22
  players for a plate, then show exactly one keeper and diff — the changed pixels are
  that keeper, and their mean colour is the kit. Expect roughly RGB (134,188,241) for the
  blue team's keeper and (230,137,134) for the red team's.

## Regenerating the patch after further engine edits

```bash
cd ~/gfootball_src && git diff > /path/to/engine_patches/gfootball_engine.patch
```

Untracked files in that tree (~323 of them) are generated scenarios written by
`experiments/scenario_factory.py:write_scenario`, plus `libgame.dylib.backup_pre_*`
binaries and fonts. None of it needs preserving — only the diff does.
