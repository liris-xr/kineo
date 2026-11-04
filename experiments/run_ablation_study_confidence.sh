#!/usr/bin/env bash
source "$(dirname "$0")/launch_common.sh"
require H36M_DIR EGOHUMANS_DIR PASCAL_VOC_DIR
export PASCAL_VOC_DIR

CONFIGS=configs/experiments/ablation_study/confidence
BENCHMARKS=configs/experiments/benchmarks
run_arm experiments/h36m_eval.py "$H36M_DIR" \
  "$BENCHMARKS/h36m_benchmark_nlf_estRt_estK_estD.yaml" "${SEQUENCES_FILTER[@]}"
run_arm experiments/h36m_eval.py "$H36M_DIR" \
  "$CONFIGS/h36m_enabled_confidence_nlf_estRt_estK_estD_occluded.yaml" "${SEQUENCES_FILTER[@]}"
for input in "" _occluded; do
  for arm in unit_weights disabled_confidence; do
    run_arm experiments/h36m_eval.py "$H36M_DIR" \
      "$CONFIGS/h36m_${arm}_nlf_estRt_estK_estD$input.yaml" "${SEQUENCES_FILTER[@]}"
  done
done
for detector in nlf dwpose; do
  run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
    "$BENCHMARKS/egohumans_benchmark_${detector}_estRt_estK_estD.yaml" "${SEQUENCES_FILTER[@]}"
  for arm in unit_weights disabled_confidence; do
    run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
      "$CONFIGS/egohumans_${arm}_${detector}_estRt_estK_estD.yaml" "${SEQUENCES_FILTER[@]}"
  done
done

o="$OUTPUT_ROOT_DIR"
for arm in unit_weights disabled_confidence; do
  "$PYTHON" -m experiments.figures.generate_confidence_ablation_rows \
    --pair Human3.6M NLF Original "$o/h36m_${arm}_nlf_estRt_estK_estD" \
      "$o/h36m_benchmark_nlf_estRt_estK_estD" \
    --pair Human3.6M NLF Occluded "$o/h36m_${arm}_nlf_estRt_estK_estD_occluded" \
      "$o/h36m_enabled_confidence_nlf_estRt_estK_estD_occluded" \
    --pair EgoHumans NLF Original "$o/egohumans_${arm}_nlf_estRt_estK_estD" \
      "$o/egohumans_benchmark_nlf_estRt_estK_estD" \
    --pair EgoHumans DWPose Original "$o/egohumans_${arm}_dwpose_estRt_estK_estD" \
      "$o/egohumans_benchmark_dwpose_estRt_estK_estD" \
    --caption-ablation "$([ "$arm" = unit_weights ] && echo weights || echo all)" \
    > "$TABLES_DIR/confidence_$arm.tex"
done
