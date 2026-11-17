#!/usr/bin/env bash
source "$(dirname "$0")/launch_common.sh"
require H36M_DIR EGOHUMANS_DIR

BENCHMARKS=configs/experiments/benchmarks
run_arm experiments/h36m_eval.py "$H36M_DIR" \
  "$BENCHMARKS/h36m_benchmark_nlf_estRt_estK_estD.yaml"
run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
  "$BENCHMARKS/egohumans_benchmark_nlf_estRt_estK_estD.yaml"
for reconstruction in moge vggt; do
  run_arm experiments/h36m_eval.py "$H36M_DIR" \
    "configs/experiments/scene_reconstruction/h36m_scene_reconstruction_nlf_$reconstruction.yaml" \
    --sequences-filter "S9_Directions 1"
done

h36m="$OUTPUT_ROOT_DIR/h36m_benchmark_nlf_estRt_estK_estD/annotations"
egohumans="$OUTPUT_ROOT_DIR/egohumans_benchmark_nlf_estRt_estK_estD/annotations"
"$PYTHON" experiments/figures/generate_method_overview_images.py "$EGOHUMANS_DIR" \
  "$egohumans" "$FIGURES_DIR/method_overview"
"$PYTHON" experiments/figures/generate_correspondences_sampling_figure.py \
  "$H36M_DIR" "$h36m" "$FIGURES_DIR/correspondences"
"$PYTHON" experiments/figures/generate_h36m_scene_reconstruction_images.py \
  "$H36M_DIR" "$OUTPUT_ROOT_DIR" "$FIGURES_DIR/scene_reconstruction"
"$PYTHON" experiments/figures/generate_confidence_scores_figure.py \
  "$FIGURES_DIR/confidence/confidence_score_egohumans.png"
"$PYTHON" experiments/figures/generate_smpl_scaling_figure.py "$H36M_DIR" \
  body_models/smpl "$FIGURES_DIR/smpl_scaling"
"$PYTHON" experiments/figures/generate_moge_scaling_figure.py "$H36M_DIR" \
  "$FIGURES_DIR/moge_scaling"
"$PYTHON" experiments/h36m_moge_depth_diagnostic.py "$H36M_DIR" \
  --output-dir "$OUTPUT_ROOT_DIR/h36m_moge_depth_diagnostic"
