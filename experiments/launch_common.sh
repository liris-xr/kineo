set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Datasets default to the layout the previews use
if [ -n "${KINEO_DATASETS_DIR:-}" ]; then
  H36M_DIR="${H36M_DIR:-$KINEO_DATASETS_DIR/H3.6M}"
  EGOHUMANS_DIR="${EGOHUMANS_DIR:-$KINEO_DATASETS_DIR/EgoHumans}"
  AISTPP_DIR="${AISTPP_DIR:-$KINEO_DATASETS_DIR/AIST++}"
fi

# Keep relative, Git Bash mangles absolute paths
export OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-outputs}"
export CACHE_ROOT_DIR="${CACHE_ROOT_DIR:-$OUTPUT_ROOT_DIR/cache}"
PYTHON="${PYTHON:-python}"
TABLES_DIR="${TABLES_DIR:-$OUTPUT_ROOT_DIR/tables}"
FIGURES_DIR="${FIGURES_DIR:-$OUTPUT_ROOT_DIR/figures}"
mkdir -p "$OUTPUT_ROOT_DIR" "$TABLES_DIR" "$FIGURES_DIR"

# Comma-separated sequence names, for a smoke test
SEQUENCES_FILTER=()
if [ -n "${SEQUENCES:-}" ]; then
  IFS=, read -r -a names <<< "$SEQUENCES"
  SEQUENCES_FILTER=(--sequences-filter "${names[@]}")
fi

# require VAR...: each must name an existing directory
require() {
  local var
  for var in "$@"; do
    if [ -z "${!var:-}" ] || [ ! -d "${!var}" ]; then
      echo "set $var to an existing directory" >&2
      exit 1
    fi
  done
}

# run_arm SCRIPT DATASET_DIR CONFIG [ARGS...]: skipped once a full run finished
run_arm() {
  local script=$1 dataset_dir=$2 config=$3
  shift 3
  local arm
  arm=$(basename "$config" .yaml)
  local log="$OUTPUT_ROOT_DIR/$arm.log" done="$OUTPUT_ROOT_DIR/$arm.done"
  # Eval scripts print the failed sequences last
  if [ -z "${SEQUENCES:-}" ] && { [ -f "$done" ] || grep -qs "^Failed sequences:" "$log"; }; then
    echo "== $arm done, not rerun"
    return
  fi
  echo "== running $arm"
  "$PYTHON" -u "$script" "$dataset_dir" --config-file "$config" "$@" > "$log" 2>&1
  if [ -z "${SEQUENCES:-}" ]; then
    touch "$done"
  fi
}
