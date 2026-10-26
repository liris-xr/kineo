#!/usr/bin/env bash
# Runs the number-of-views ablation on EgoHumans (8 cameras down to 2) or
# Human3.6M (4 down to 2).
#
# Each arm is a configs/experiments/ablation_study/n_views/ config, run with
# experiments/<dataset>_eval.py and summarized with
# experiments/summarize_benchmark.py. The camera sets come from
# kineo/datasets/<dataset>/camera_subsets.yaml, where every smaller set is
# nested in the largest one: that one runs first and fills the per-view
# detection cache, and the others then detect nothing. Arms run one after
# another in their own process, never at once. An arm whose log already ends
# its run is not rerun, so adding view counts to an existing run only runs the
# new ones.
#
# Usage:
#   DATASET=egohumans|h36m DATASET_DIR=<preprocessed dataset> \
#     [OUTPUT_ROOT_DIR=outputs] [CACHE_ROOT_DIR=$OUTPUT_ROOT_DIR/cache] \
#     [PYTHON=python] [SEQUENCES="<name>,..."] [N_VIEWS="<largest first>"] \
#     [PIPELINE=nlf_estRt_estK_estD|nlf_estRt_gtK_gtD] \
#     bash experiments/run_ablation_study_n_views.sh
# or `pixi run egohumans-n-views-ablation <egohumans_dir>` /
# `pixi run h36m-n-views-ablation <h36m_dir>`.
# CACHE_ROOT_DIR defaults to a cache of the run's own, so its detections are
# not mixed with those of earlier runs. SEQUENCES restricts every arm to a
# subset, e.g. for a smoke run. N_VIEWS picks the arms, largest first so that
# it fills the cache the others read.
set -euo pipefail

: "${DATASET:?set DATASET to egohumans or h36m}"
: "${DATASET_DIR:?set DATASET_DIR to the preprocessed dataset directory}"
case "$DATASET" in
  egohumans) DEFAULT_N_VIEWS="8 7 6 5 4 3 2" ;;
  h36m) DEFAULT_N_VIEWS="4 3 2" ;;
  *) echo "unknown DATASET: $DATASET" >&2; exit 1 ;;
esac
# Relative on purpose: Git Bash turns an absolute $(pwd) into a /d/... path
# that Python cannot open once it sits after an "=" in an argument.
export OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-outputs}"
export CACHE_ROOT_DIR="${CACHE_ROOT_DIR:-$OUTPUT_ROOT_DIR/cache}"
PYTHON="${PYTHON:-python}"
# Which benchmark the arms derive from: estimated intrinsics, or the ground
# truth ones loaded as is (nlf_estRt_gtK_gtD).
PIPELINE="${PIPELINE:-nlf_estRt_estK_estD}"
CONFIGS=configs/experiments/ablation_study/n_views
FILTER_ARGS=()
if [ -n "${SEQUENCES:-}" ]; then
  # Comma-separated: Human3.6M sequence names contain spaces.
  IFS=, read -r -a SUBSET <<< "$SEQUENCES"
  FILTER_ARGS=(--sequences-filter "${SUBSET[@]}")
fi

mkdir -p "$OUTPUT_ROOT_DIR"

read -r -a ARMS <<< "${N_VIEWS:-$DEFAULT_N_VIEWS}"
for n_views in "${ARMS[@]}"; do
  arm=${DATASET}_benchmark_${PIPELINE}_${n_views}views
  log="$OUTPUT_ROOT_DIR/$arm.log"
  # The eval scripts print their failed sequences as their last step.
  if [ -f "$log" ] && grep -q "^Failed sequences:" "$log"; then
    echo "== $arm complete, not rerun"
  else
    echo "== running $arm"
    "$PYTHON" -u "experiments/${DATASET}_eval.py" "$DATASET_DIR" \
      --config-file "$CONFIGS/$arm.yaml" "${FILTER_ARGS[@]}" > "$log" 2>&1
  fi
  grep "^Failed sequences:" "$log"
  "$PYTHON" experiments/summarize_benchmark.py "$OUTPUT_ROOT_DIR/$arm"
done
