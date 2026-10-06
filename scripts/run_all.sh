#!/usr/bin/env bash
# Every experiment on the 3 reported split seeds. M3 needs the M1 teacher of the same seed.
set -euo pipefail
SEEDS="${SEEDS:-0 1 2}"
for seed in $SEEDS; do
  for config in b0_encdec_unpool b1_unet_r50 ub_unet_r50 m1_tags_unet_r50 \
                m3_self_train_unet_r50 m4_unimatch_unet_r50; do
    uv run python -m wsss.train --config "configs/${config}.yaml" --seed "$seed"
  done
done
uv run python scripts/aggregate_results.py runs --gt full
uv run python scripts/aggregate_results.py runs --gt eroded
