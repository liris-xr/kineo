#!/usr/bin/env bash
source "$(dirname "$0")/launch_common.sh"
require H36M_DIR EGOHUMANS_DIR

CONFIGS=configs/experiments/ablation_study/global_scaling
for arm in moge smpl smpl_oracle_shape; do
  run_arm experiments/h36m_eval.py "$H36M_DIR" \
    "$CONFIGS/h36m_global_scaling_gtKps_$arm.yaml" "${SEQUENCES_FILTER[@]}"
  run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
    "$CONFIGS/egohumans_global_scaling_gtKps_$arm.yaml" "${SEQUENCES_FILTER[@]}"
done

"$PYTHON" experiments/figures/generate_ablation_study_global_scaling_rows.py \
  "$H36M_DIR" "$EGOHUMANS_DIR" "$OUTPUT_ROOT_DIR" > "$TABLES_DIR/global_scaling.tex"
