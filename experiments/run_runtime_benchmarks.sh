#!/usr/bin/env bash
# H36M_RUNTIME_VIDEOS_DIR holds {short,medium,long}_video/<camera>.mp4
source "$(dirname "$0")/launch_common.sh"
require H36M_RUNTIME_VIDEOS_DIR

CONFIGS=configs/experiments/runtime
for length in short medium long; do
  for detector in nlf rtmpose rtmpose_b32_half; do
    config="h36m_runtime_${detector%_b32_half}_${length}_video"
    [ "$detector" = rtmpose_b32_half ] && config+=_b32_half
    run_arm experiments/h36m_runtime_eval.py \
      "$H36M_RUNTIME_VIDEOS_DIR/${length}_video" "$CONFIGS/$config.yaml"
  done
done

"$PYTHON" experiments/figures/generate_timings_figure.py "$OUTPUT_ROOT_DIR" \
  --output-path "$FIGURES_DIR/kineo_timings.pdf"
