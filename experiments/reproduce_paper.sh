#!/usr/bin/env bash
# Finished runs are skipped
set -euo pipefail
cd "$(dirname "$0")"

bash run_benchmarks.sh
bash run_ablation_study_camera_params.sh
bash run_ablation_study_global_scaling.sh
bash run_ablation_study_sampling.sh
bash run_ablation_study_confidence.sh
bash run_ablation_study_n_views.sh
bash run_aistpp_sync_ablation.sh
bash run_online_vs_offline.sh
bash run_runtime_benchmarks.sh
bash run_method_figures.sh
