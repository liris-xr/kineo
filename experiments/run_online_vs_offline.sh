#!/usr/bin/env bash
source "$(dirname "$0")/launch_common.sh"
require H36M_DIR

root="$OUTPUT_ROOT_DIR"
# Each mode needs its own output root
for mode in online offline; do
  (
    export OUTPUT_ROOT_DIR="$root/$mode"
    mkdir -p "$OUTPUT_ROOT_DIR"
    run_arm experiments/h36m_online_eval.py "$H36M_DIR" \
      "configs/experiments/online_vs_offline/h36m_$mode.yaml"
  )
  "$PYTHON" -m experiments.report_online_benchmark "$root/$mode/h36m_online"
done

"$PYTHON" -m experiments.figures.generate_online_rows "$root/offline/h36m_online" \
  "$root/online/h36m_online" > "$TABLES_DIR/online_vs_offline.tex"
