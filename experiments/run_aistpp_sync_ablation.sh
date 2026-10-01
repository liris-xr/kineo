#!/usr/bin/env bash
# Runs, scores and tabulates the audio sync ablation on AIST++.
#
# Every arm is a sync-only config under configs/experiments/sync_only/, run
# with experiments/aistpp_eval.py, scored with experiments/score_sync_run.py,
# then experiments/figures/generate_sync_ablation_rows.py writes the LaTeX
# rows and a markdown table to $OUTPUT_ROOT_DIR/sync_ablation_rows.txt.
# An arm whose annotations/ already holds every sequence is not rerun.
#
# Usage:
#   AISTPP_DIR=<preprocessed AIST++> DEMAND_DIR=<DEMAND, see demand-download> \
#     [OUTPUT_ROOT_DIR=outputs] [PYTHON=python] [SEQUENCES_FILE=<names>] \
#     bash experiments/run_aistpp_sync_ablation.sh
# or `pixi run aistpp-sync-ablation <aistpp_dir> <demand_dir>`.
# SEQUENCES_FILE, one sequence name per line (experiments/
# list_aistpp_sync_subset.py), restricts every arm to a subset for sweeps;
# the paper rows are meant to run on the full split.
set -euo pipefail

: "${AISTPP_DIR:?set AISTPP_DIR to the preprocessed AIST++ directory}"
: "${DEMAND_DIR:?set DEMAND_DIR to the DEMAND directory}"
export DEMAND_DIR
# Relative on purpose: Git Bash turns an absolute $(pwd) into a /d/... path
# that Python cannot open once it sits after an "=" in an argument.
export OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-outputs}"
PYTHON="${PYTHON:-python}"
CONFIGS=configs/experiments/sync_only
DATASET_FILE="$AISTPP_DIR/aistpp_pose_test_raw_sequences.json"
N_SEQUENCES=$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$DATASET_FILE")
FILTER_ARGS=()
if [ -n "${SEQUENCES_FILE:-}" ]; then
  mapfile -t SUBSET < <(grep -v '^\s*$' "$SEQUENCES_FILE")
  FILTER_ARGS=(--sequences-filter "${SUBSET[@]}")
  N_SEQUENCES=${#SUBSET[@]}
fi

# The rows of the paper table, in order, then the controls and sweeps.
PAPER_ARMS=(
  aistpp_sync_mfcc
  aistpp_sync_mfcc_noPreroll
  aistpp_sync_mfcc_noPreroll_park_-20dB
  aistpp_sync_mfcc_noPreroll_park_-10dB
  aistpp_sync_mfcc_noPreroll_park_0dB
  aistpp_sync_mfcc_noPreroll_cafeteria_-20dB
  aistpp_sync_mfcc_noPreroll_cafeteria_-10dB
  aistpp_sync_mfcc_noPreroll_cafeteria_0dB
  aistpp_sync_mfcc_noPreroll_synthBeep
)
EXTRA_ARMS=(
  "CountInOnly=aistpp_sync_mfcc_countInOnly"
  "NoBeep jitter 0.1s=aistpp_sync_mfcc_noPreroll_j01"
  "White noise 0dB=aistpp_sync_mfcc_noPreroll_noise"
  "White noise +30dB=aistpp_sync_mfcc_noPreroll_noise30dB"
  "White noise +20dB=aistpp_sync_mfcc_noPreroll_noise20dB"
  "White noise +10dB=aistpp_sync_mfcc_noPreroll_noise10dB"
  "White noise -10dB=aistpp_sync_mfcc_noPreroll_noise-10dB"
)

run_arm() {
  local arm=$1
  local run_dir="$OUTPUT_ROOT_DIR/$arm"
  local done=0
  [ -d "$run_dir/annotations" ] && done=$(ls "$run_dir/annotations" | wc -l)
  if [ "$done" -lt "$N_SEQUENCES" ]; then
    echo "== running $arm ($done/$N_SEQUENCES present)"
    "$PYTHON" -u experiments/aistpp_eval.py "$AISTPP_DIR" \
      --config-file "$CONFIGS/$arm.yaml" "${FILTER_ARGS[@]}" \
      > "$OUTPUT_ROOT_DIR/$arm.log" 2>&1
  else
    echo "== $arm complete, not rerun"
  fi
  "$PYTHON" experiments/score_sync_run.py "$run_dir" --dataset-file "$DATASET_FILE" \
    | grep -E "scored|within 1|signed"
}

for arm in "${PAPER_ARMS[@]}"; do run_arm "$arm"; done
EXTRA_ARGS=()
for entry in "${EXTRA_ARMS[@]}"; do
  arm="${entry#*=}"
  run_arm "$arm"
  EXTRA_ARGS+=(--extra "${entry%%=*}=$OUTPUT_ROOT_DIR/$arm")
done

PAPER_DIRS=()
for arm in "${PAPER_ARMS[@]}"; do PAPER_DIRS+=("$OUTPUT_ROOT_DIR/$arm"); done
"$PYTHON" experiments/figures/generate_sync_ablation_rows.py "${PAPER_DIRS[@]}" \
  "${EXTRA_ARGS[@]}" | tee "$OUTPUT_ROOT_DIR/sync_ablation_rows.txt"
