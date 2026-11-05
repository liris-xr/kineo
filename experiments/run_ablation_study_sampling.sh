#!/usr/bin/env bash
# Sampling strategies, and the confidence filter and weights with RTMPose
# The Human3.6M tables share runs, so they use the same sequences
source "$(dirname "$0")/launch_common.sh"
require H36M_DIR EGOHUMANS_DIR PASCAL_VOC_DIR
export PASCAL_VOC_DIR

CONFIGS=configs/experiments/ablation_study/sampling
# Random sampling is the benchmark config
run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
  configs/experiments/benchmarks/egohumans_benchmark_nlf_estRt_estK_estD.yaml \
  "${SEQUENCES_FILTER[@]}"
for arm in fps topconf; do
  run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
    "$CONFIGS/egohumans_sampling_nlf_$arm.yaml" "${SEQUENCES_FILTER[@]}"
done
# h36m_config ARM INPUT: on the original images, random sampling is the benchmark
h36m_config() {
  if [ "$1$2" = random ]; then
    echo configs/experiments/benchmarks/h36m_benchmark_rtmpose_estRt_estK_estD.yaml
  else
    echo "$CONFIGS/h36m_sampling_rtmpose_$1$2.yaml"
  fi
}
h36m_run() { echo "$OUTPUT_ROOT_DIR/$(basename "$(h36m_config "$@")" .yaml)"; }
h36m_runs=()
for input in "" _occluded; do
  for arm in random fps topconf random_unfiltered; do
    run_arm experiments/h36m_eval.py "$H36M_DIR" "$(h36m_config "$arm" "$input")" \
      "${SEQUENCES_FILTER[@]}"
    h36m_runs+=("$(h36m_run "$arm" "$input")")
  done
  run_arm experiments/h36m_eval.py "$H36M_DIR" \
    "configs/experiments/ablation_study/confidence/h36m_confidence_rtmpose_unit_weights$input.yaml" \
    "${SEQUENCES_FILTER[@]}"
  h36m_runs+=("$OUTPUT_ROOT_DIR/h36m_confidence_rtmpose_unit_weights$input")
done

random="$OUTPUT_ROOT_DIR/egohumans_benchmark_nlf_estRt_estK_estD"
fps="$OUTPUT_ROOT_DIR/egohumans_sampling_nlf_fps"
topp="$OUTPUT_ROOT_DIR/egohumans_sampling_nlf_topconf"
"$PYTHON" -m experiments.benchmark_sampling_runtime "$random" \
  "$OUTPUT_ROOT_DIR/sampling_runtime_egohumans.json"
"$PYTHON" -m experiments.figures.compute_pair_sampling_coverage "$random" "$fps" \
  "$topp" "$OUTPUT_ROOT_DIR/sampling_coverage_egohumans.json"
"$PYTHON" -m experiments.figures.generate_sampling_rows "$random" "$fps" \
  --topp-run-dir "$topp" --runtime-json "$OUTPUT_ROOT_DIR/sampling_runtime_egohumans.json" \
  > "$TABLES_DIR/sampling_egohumans.tex"
"$PYTHON" experiments/figures/generate_pair_sampling_figure.py "$random" "$fps" \
  "$EGOHUMANS_DIR" "$FIGURES_DIR/sampling" --topp-run-dir "$topp"

inputs=()
for input in "" _occluded; do
  runtime="$OUTPUT_ROOT_DIR/sampling_runtime_h36m$input.json"
  "$PYTHON" -m experiments.benchmark_sampling_runtime \
    "$(h36m_run random "$input")" "$runtime" \
    --pairs 4000 --keypoints 400 --threshold 0.5
  label=Original
  [ -n "$input" ] && label=Occluded
  inputs+=(--input "$label" "$(h36m_run random "$input")"
           "$(h36m_run fps "$input")" "$(h36m_run topconf "$input")" "$runtime")
done
"$PYTHON" -m experiments.figures.generate_h36m_sampling_rows "${inputs[@]}" \
  --restrict-to "${h36m_runs[@]}" > "$TABLES_DIR/sampling_h36m.tex"
"$PYTHON" -m experiments.figures.generate_confidence_filter_rows "$OUTPUT_ROOT_DIR" \
  "$H36M_DIR" --workers "${WORKERS:-1}" --restrict-to "${h36m_runs[@]}" \
  > "$TABLES_DIR/confidence_filter_h36m.tex"
pairs=()
for input in "" _occluded; do
  label=Original
  [ -n "$input" ] && label=Occluded
  pairs+=(--pair Human3.6M RTMPose "$label"
          "$OUTPUT_ROOT_DIR/h36m_confidence_rtmpose_unit_weights$input"
          "$(h36m_run random "$input")")
done
"$PYTHON" -m experiments.figures.generate_confidence_ablation_rows "${pairs[@]}" \
  --restrict-to "${h36m_runs[@]}" > "$TABLES_DIR/confidence_weights_h36m.tex"

echo "The coverage column of the EgoHumans tables is in $OUTPUT_ROOT_DIR/sampling_coverage_egohumans.json"
