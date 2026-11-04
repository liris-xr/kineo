#!/usr/bin/env bash
source "$(dirname "$0")/launch_common.sh"
require EGOHUMANS_DIR

CONFIGS=configs/experiments/ablation_study/n_views
est=() gt=()
for n_views in 2 3 4 5 6 7 8; do
  for intrinsics in estK_estD gtK_gtD; do
    run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
      "$CONFIGS/egohumans_benchmark_nlf_estRt_${intrinsics}_${n_views}views.yaml" \
      "${SEQUENCES_FILTER[@]}"
  done
  est+=("$OUTPUT_ROOT_DIR/egohumans_benchmark_nlf_estRt_estK_estD_${n_views}views")
  gt+=("$OUTPUT_ROOT_DIR/egohumans_benchmark_nlf_estRt_gtK_gtD_${n_views}views")
done
for intrinsics in estK_estD gtK_gtD; do
  run_arm experiments/egohumans_eval.py "$EGOHUMANS_DIR" \
    "configs/experiments/benchmarks/egohumans_benchmark_nlf_estRt_$intrinsics.yaml" \
    "${SEQUENCES_FILTER[@]}"
done
est+=("$OUTPUT_ROOT_DIR/egohumans_benchmark_nlf_estRt_estK_estD")
gt+=("$OUTPUT_ROOT_DIR/egohumans_benchmark_nlf_estRt_gtK_gtD")

"$PYTHON" -m experiments.figures.generate_n_views_rows --est-runs "${est[@]}" \
  --gt-runs "${gt[@]}" > "$TABLES_DIR/n_views.tex"
"$PYTHON" experiments/figures/generate_n_views_figure.py "$FIGURES_DIR/n_views" \
  --est-runs "${est[@]}" --gt-runs "${gt[@]}"
