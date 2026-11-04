#!/usr/bin/env bash
# Main benchmarks, detectors, AIST++ synchronization and distortion error
# HSFM_EVAL_DIR (HSfM predictions) adds the tables compared with HSfM
source "$(dirname "$0")/launch_common.sh"
require H36M_DIR EGOHUMANS_DIR AISTPP_DIR

CONFIGS=configs/experiments/benchmarks
for arm in nlf_estRt_estK_estD nlf_estRt_estK_omitD nlf_estRt_gtK_gtD \
    nlf_gtRt_gtK_gtD gtKps_estRt_estK_estD rtmpose_estRt_estK_estD; do
  run_arm experiments/h36m_eval.py "$H36M_DIR" \
    "$CONFIGS/h36m_benchmark_$arm.yaml" "${SEQUENCES_FILTER[@]}"
done
for arm in nlf_estRt_estK_estD nlf_estRt_estK_omitD nlf_estRt_gtK_gtD \
    nlf_gtRt_gtK_gtD gtKps_estRt_estK_estD dwpose_estRt_estK_estD; do
  run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
    "$CONFIGS/egohumans_benchmark_$arm.yaml" "${SEQUENCES_FILTER[@]}"
done
for arm in estRt_estK_estD gtSync_estRt_estK_estD; do
  run_arm experiments/aistpp_eval.py "$AISTPP_DIR" \
    "$CONFIGS/aistpp_benchmark_nlf_$arm.yaml" "${SEQUENCES_FILTER[@]}"
done

est="$OUTPUT_ROOT_DIR/aistpp_benchmark_nlf_estRt_estK_estD"
gt="$OUTPUT_ROOT_DIR/aistpp_benchmark_nlf_gtSync_estRt_estK_estD"
summarize=(--dataset-file "$AISTPP_DIR/aistpp_pose_test_raw_sequences.json"
           --exclude-list kineo/datasets/aistpp/aistpp_ignore_list.txt)
"$PYTHON" experiments/summarize_benchmark.py "$est" "${summarize[@]}" --common-with "$gt"
"$PYTHON" experiments/summarize_benchmark.py "$gt" "${summarize[@]}" --common-with "$est"
"$PYTHON" experiments/figures/generate_aistpp_sync_rows.py "$est" "$gt" \
  > "$TABLES_DIR/aistpp_sync.tex"

"$PYTHON" experiments/figures/generate_ablation_study_detectors_rows.py \
  "$H36M_DIR" "$EGOHUMANS_DIR" "$OUTPUT_ROOT_DIR" > "$TABLES_DIR/detectors.tex"

egohumans="$OUTPUT_ROOT_DIR/egohumans_benchmark_nlf_estRt_estK_estD/annotations"
mkdir -p "$FIGURES_DIR/distortion_error/panels"
"$PYTHON" experiments/figures/generate_distortion_error_stats.py \
  "$EGOHUMANS_DIR" "$egohumans" "$FIGURES_DIR/distortion_error"
"$PYTHON" experiments/figures/generate_distortion_figure.py \
  "$EGOHUMANS_DIR" "$egohumans" "$FIGURES_DIR/distortion_error/distortion_main" \
  --views tennis_008/cam17,fencing_006/cam14,basketball_012/cam08
# Widest camera per scene, unless of the minority focal length
"$PYTHON" experiments/figures/generate_distortion_figure.py \
  "$EGOHUMANS_DIR" "$egohumans" "$FIGURES_DIR/distortion_error/panels" \
  --panels --width 5.91 --views tagging_005/cam04,legoassemble_005/cam02,fencing_006/cam14,basketball_012/cam08,volleyball_005/cam03,badminton_056/cam06,tennis_008/cam17

if [ -n "${HSFM_EVAL_DIR:-}" ]; then
  "$PYTHON" experiments/figures/generate_benchmark_rows.py "$H36M_DIR" \
    "$EGOHUMANS_DIR" "$HSFM_EVAL_DIR" "$OUTPUT_ROOT_DIR" > "$TABLES_DIR/benchmark.tex"
  "$PYTHON" experiments/figures/generate_h36m_comparison_table.py "$H36M_DIR" \
    "$HSFM_EVAL_DIR" "$OUTPUT_ROOT_DIR/h36m_benchmark_nlf_estRt_estK_estD/annotations" \
    "$FIGURES_DIR/min_max_median"
else
  echo "HSFM_EVAL_DIR unset: skipping the tables compared with HSfM"
fi
