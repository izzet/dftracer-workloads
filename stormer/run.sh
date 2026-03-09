#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/logs"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output}"
RUNS_DIR="${LOG_DIR}/runs"

LAUNCHER="${LAUNCHER:-none}"
MPIEXEC="${MPIEXEC:-mpiexec}"
NODES="${NODES:-1}"
TASKS="${TASKS:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
DATA_FOLDER="${DATA_FOLDER:-}"

NUM_WORKERS="${NUM_WORKERS:-10}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
EPOCHS="${EPOCHS:-2}"
MAX_TRAINING_STEP="${MAX_TRAINING_STEP:--1}"
PRECISION="${PRECISION:-}"

DFTRACER_ENABLE="${DFTRACER_ENABLE:-0}"
DFTRACER_INIT="${DFTRACER_INIT:-PRELOAD}"
DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA:-1}"
TRAIN_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./run.sh [options] [-- extra train.py args]

Options:
  --launcher MODE            `none` (default), `flux`, or `mpiexec`
  --nodes N                  Number of nodes for flux runs (default: 1)
  --tasks N                  Total tasks/ranks (default: nodes * gpus-per-node)
  --gpus-per-node N          GPUs per node visible to the job (default: 1)
  --gpus-per-task N          GPUs per task for flux runs (default: 1)
  --data-folder PATH         HDF5 dataset root with normalization files
  --output-root PATH         Root directory for workload outputs
  --epochs N                 Training epochs (default: 2)
  --max-training-step N      Max steps per epoch (-1 means unlimited)
  --num-workers N            DataLoader workers (default: 10)
  --prefetch-factor N        DataLoader prefetch factor (default: 2)
  --pin-memory 0|1           Enable DataLoader pin_memory (default: 0)
  --persistent-workers 0|1   Enable DataLoader persistent_workers (default: 0)
  --precision VALUE          Training precision passed to train.py
  --dftracer-enable 0|1      Enable DFTracer instrumentation env (default: 0)
  -h, --help                 Show this help message

Examples:
  DATA_FOLDER=/path/to/hdf5 ./run.sh
  DATA_FOLDER=/path/to/hdf5 DFTRACER_ENABLE=1 ./run.sh --epochs 1
  DATA_FOLDER=/path/to/hdf5 ./run.sh --launcher none -- --precision 32
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --gpus-per-task) GPUS_PER_TASK="$2"; shift 2 ;;
    --data-folder) DATA_FOLDER="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --max-training-step) MAX_TRAINING_STEP="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --pin-memory) PIN_MEMORY="$2"; shift 2 ;;
    --persistent-workers) PERSISTENT_WORKERS="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --dftracer-enable) DFTRACER_ENABLE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      TRAIN_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

source "${ROOT_DIR}/setup_env.sh"
mkdir -p "${RUNS_DIR}" "${OUTPUT_ROOT}"

if [[ -z "${DATA_FOLDER}" ]]; then
  echo "DATA_FOLDER is required. Pass --data-folder or export DATA_FOLDER." >&2
  exit 1
fi

if [[ -z "${TASKS}" ]]; then
  TASKS=$((NODES * GPUS_PER_NODE / GPUS_PER_TASK))
fi

RUN_LOG_DIR="${RUNS_DIR}/${RUN_ID}"
RUN_OUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_FILE="${RUN_LOG_DIR}/output.log"
ENV_FILE="${RUN_LOG_DIR}/env.txt"
mkdir -p "${RUN_LOG_DIR}" "${RUN_OUT_DIR}"

if [[ "${DFTRACER_ENABLE}" == "1" ]]; then
  export DFTRACER_ENABLE
  export DFTRACER_INIT
  export DFTRACER_INC_METADATA
  export DFTRACER_LOG_FILE="${DFTRACER_LOG_FILE:-${RUN_LOG_DIR}/trace}"
  export DFTRACER_DATA_DIR="${DFTRACER_DATA_DIR:-${DATA_FOLDER}:${RUN_OUT_DIR}}"
  if [[ -n "${DFTRACER_PRELOAD_LIB:-}" ]]; then
    export LD_PRELOAD="${DFTRACER_PRELOAD_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
fi

export OMP_PLACES="${OMP_PLACES:-threads}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"

TRAIN_CMD=(
  python -u "${ROOT_DIR}/train.py"
  --data-folder "${DATA_FOLDER}"
  --output-folder "${RUN_OUT_DIR}"
  --log-folder "${RUN_LOG_DIR}"
  --num-workers "${NUM_WORKERS}"
  --batch-size 1
  --prefetch-factor "${PREFETCH_FACTOR}"
  --data-freq 6
  --intervals 6 12 24
  --disable-collation
  --lr 5e-4
  --beta-1 0.9
  --beta-2 0.95
  --weight-decay 1e-5
  --warmup-epochs 10
  --warmup-start-lr 1e-8
  --eta-min 1e-8
  --in-img-size 128 256
  --patch-size 16
  --depth 24
  --num-heads 16
  --mlp-ratio 4.0
  --epochs "${EPOCHS}"
  --max-training-step "${MAX_TRAINING_STEP}"
  --weighted-loss
  --enable-progress-bar
  --ngpus-per-node "${GPUS_PER_NODE}"
  --sync-batchnorm
)

if [[ "${PIN_MEMORY}" == "1" ]]; then
  TRAIN_CMD+=(--pin-memory)
fi

if [[ "${PERSISTENT_WORKERS}" == "1" ]]; then
  TRAIN_CMD+=(--persistent-workers)
fi

if [[ -n "${PRECISION}" ]]; then
  TRAIN_CMD+=(--precision "${PRECISION}")
fi

if [[ ${#TRAIN_EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${TRAIN_EXTRA_ARGS[@]}")
fi

{
  echo "RUN_ID=${RUN_ID}"
  echo "LAUNCHER=${LAUNCHER}"
  echo "MPIEXEC=${MPIEXEC}"
  echo "NODES=${NODES}"
  echo "TASKS=${TASKS}"
  echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "GPUS_PER_TASK=${GPUS_PER_TASK}"
  echo "DATA_FOLDER=${DATA_FOLDER}"
  echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
  echo "RUN_OUT_DIR=${RUN_OUT_DIR}"
  echo "DFTRACER_ENABLE=${DFTRACER_ENABLE}"
  echo "COMMAND=${TRAIN_CMD[*]}"
  echo ""
  env | sort
} > "${ENV_FILE}"

case "${LAUNCHER}" in
  flux)
    if ! command -v flux >/dev/null 2>&1; then
      echo "flux launcher requested but 'flux' is not available." >&2
      exit 1
    fi
    flux run -N "${NODES}" -n "${TASKS}" -g "${GPUS_PER_TASK}" --exclusive \
      "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    ;;
  mpiexec)
    "${MPIEXEC}" -n "${TASKS}" "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    ;;
  none)
    "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    ;;
  *)
    echo "Unsupported launcher: ${LAUNCHER}" >&2
    exit 1
    ;;
esac

echo "Stormer run complete."
echo "  logs:   ${RUN_LOG_DIR}"
echo "  output: ${RUN_OUT_DIR}"
