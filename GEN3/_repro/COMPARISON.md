# GEN3 rebuild vs the shipped batch

Rebuilt from `clips/ground_truth.csv` + `clips/reproducibility.csv` alone: scenario re-emitted from `SHAPES`, match replayed from frame 0, both render passes taken fresh, grid and split burned in again. Nothing read from `_cache/renders`.

## clip_01 — flankR seed 322, frames 1473-1598, offset (16.0, 10.0)

- **scenario sha** `4b5578f36d1cee66` vs shipped `4b5578f36d1cee66` — identical
- **play sha** `3fadbc1165f94751` vs shipped `3fadbc1165f94751` — identical
- **vis render pass** (raw frames, pre-encode) — identical (76,800,000 pixels, max abs diff 0)
- **inv render pass** (raw frames, pre-encode) — identical (76,800,000 pixels, max abs diff 0)
- **full_visibility.mov** (decoded RGB) — identical (76,800,000 pixels, max abs diff 0); file bytes identical
- **split_1s_4s.mov** (decoded RGB) — identical (76,800,000 pixels, max abs diff 0); file bytes identical
- **ground truth** — ball D8 (624.8, 301.4) -> E5 (328.0, 334.7), identical to shipped

## clip_02 — flankL seed 314, frames 410-535, offset (24.0, 8.0)

- **scenario sha** `64f85f9fc154f0af` vs shipped `64f85f9fc154f0af` — identical
- **play sha** `e740047c47614c29` vs shipped `e740047c47614c29` — identical
- **vis render pass** (raw frames, pre-encode) — identical (76,800,000 pixels, max abs diff 0)
- **inv render pass** (raw frames, pre-encode) — identical (76,800,000 pixels, max abs diff 0)
- **full_visibility.mov** (decoded RGB) — identical (76,800,000 pixels, max abs diff 0); file bytes identical
- **split_1s_4s.mov** (decoded RGB) — identical (76,800,000 pixels, max abs diff 0); file bytes identical
- **ground truth** — ball D2 (88.5, 307.9) -> D9 (664.4, 241.4), identical to shipped

## clip_03 — mid_push seed 316, frames 622-747, offset (-20.0, 10.0)

- **scenario sha** `2c1d98ec513d7df7` vs shipped `2c1d98ec513d7df7` — identical
- **play sha** `5fee18525f87799b` vs shipped `5fee18525f87799b` — identical
- **vis render pass** (raw frames, pre-encode) — identical (76,800,000 pixels, max abs diff 0)
- **inv render pass** (raw frames, pre-encode) — identical (76,800,000 pixels, max abs diff 0)
- **full_visibility.mov** (decoded RGB) — identical (76,800,000 pixels, max abs diff 0); file bytes identical
- **split_1s_4s.mov** (decoded RGB) — identical (76,800,000 pixels, max abs diff 0); file bytes identical
- **ground truth** — ball D6 (439.2, 313.5) -> D15 (1152.3, 297.7), identical to shipped

## Verdict

Every level identical: the rebuild is the shipped clip.

