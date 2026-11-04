#!/usr/bin/env bash
source "$(dirname "$0")/launch_common.sh"
require H36M_DIR EGOHUMANS_DIR

CONFIGS=configs/experiments/ablation_study/camera_parameters
for arm in estRt_estK_estD estRt_estK_omitD estRt_gtK_gtD gtRt_gtK_gtD; do
  run_arm experiments/h36m_eval.py "$H36M_DIR" \
    "$CONFIGS/h36m_cam_params_$arm.yaml" "${SEQUENCES_FILTER[@]}"
  run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
    "$CONFIGS/egohumans_cam_params_$arm.yaml" "${SEQUENCES_FILTER[@]}"
done

"$PYTHON" experiments/figures/generate_ablation_study_camera_params_rows.py \
  "$H36M_DIR" "$EGOHUMANS_DIR" "$OUTPUT_ROOT_DIR" > "$TABLES_DIR/camera_params.tex"
