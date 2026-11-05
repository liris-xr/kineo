#!/usr/bin/env bash
source "$(dirname "$0")/launch_common.sh"
require AISTPP_DIR DEMAND_DIR
export DEMAND_DIR

CONFIGS=configs/experiments/sync
# In table row order
ARMS=(
  aistpp_sync_mfcc
  aistpp_sync_mfcc_noPreroll
  aistpp_sync_mfcc_noPreroll_park_20dB
  aistpp_sync_mfcc_noPreroll_park_10dB
  aistpp_sync_mfcc_noPreroll_park_0dB
  aistpp_sync_mfcc_noPreroll_cafeteria_20dB
  aistpp_sync_mfcc_noPreroll_cafeteria_10dB
  aistpp_sync_mfcc_noPreroll_cafeteria_0dB
  aistpp_sync_mfcc_noPreroll_synthBeep
)
runs=()
for arm in "${ARMS[@]}"; do
  run_arm experiments/aistpp_eval.py "$AISTPP_DIR" "$CONFIGS/$arm.yaml" \
    "${SEQUENCES_FILTER[@]}"
  "$PYTHON" experiments/score_sync_run.py "$OUTPUT_ROOT_DIR/$arm" \
    --dataset-file "$AISTPP_DIR/aistpp_pose_test_raw_sequences.json"
  runs+=("$OUTPUT_ROOT_DIR/$arm")
done

"$PYTHON" experiments/figures/generate_sync_ablation_rows.py "${runs[@]}" \
  > "$TABLES_DIR/audio_sync_robustness.tex"
