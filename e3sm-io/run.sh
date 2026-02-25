#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs"
OUT_DIR="${ROOT_DIR}/output"
RUNS_DIR="${LOG_DIR}/runs"

MPIEXEC="${MPIEXEC:-mpiexec}"
NP="${NP:-16}"
MODE_FLAG="-n"
RECORDS="${E3SM_RECORDS:-2}"
KEEP_OUTPUT=1
DATASET="${ROOT_DIR}/e3sm-io/datasets/f_case_866x72_16p.nc"
BIN="${ROOT_DIR}/install/bin/e3sm_io"

DFTRACER_ENABLE="${DFTRACER_ENABLE:-0}"
DFTRACER_INIT="${DFTRACER_INIT:-PRELOAD}"
DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA:-1}"
DFTRACER_DATA_DIR=""
DFTRACER_LOG_PREFIX=""
# Resolve preload lib dynamically (handles Python version changes)
_PRELOAD_DEFAULT=""
if [[ -d "${ROOT_DIR}/.venv/lib" ]]; then
  _PRELOAD_DEFAULT="$(find "${ROOT_DIR}/.venv/lib" -maxdepth 3 \
      -name "libdftracer_preload.so" 2>/dev/null | head -1)"
fi
if [[ -z "${_PRELOAD_DEFAULT}" ]]; then
  for _py in python3.12 python3.11 python3.10 python3.9; do
    _p="${ROOT_DIR}/.venv/lib/${_py}/site-packages/dftracer/lib/libdftracer_preload.so"
    [[ -f "${_p}" ]] && _PRELOAD_DEFAULT="${_p}" && break
  done
fi
DFTRACER_PRELOAD_LIB="${DFTRACER_PRELOAD_LIB:-${_PRELOAD_DEFAULT}}"
DFANALYZER_ENABLED="${DFANALYZER_ENABLED:-1}"
DFANALYZER_CHECKPOINT_ENABLED="${DFANALYZER_CHECKPOINT_ENABLED:-1}"
DFANALYZER_PRESET="${DFANALYZER_PRESET:-posix}"
DFANALYZER_CHECKPOINT_DIR=""

usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

Options:
  --np N                     MPI ranks (default: 16)
  --mpiexec CMD              MPI launcher (default: mpiexec)
  --dataset PATH             Input decomposition netCDF file
  --bin PATH                 e3sm_io executable path
  --records N                Number of records for -r
  --mode-flag FLAG           E3SM mode flag: -n (varn) or -d (vard), default -n
  --keep-output 0|1          Pass -k when set to 1 (default: 1)

  --dftracer-enable 0|1      Enable DFTracer via env + LD_PRELOAD (default: 0)
  --dftracer-init MODE       DFTRACER_INIT value (default: PRELOAD)
  --dftracer-inc-metadata 0|1
                             DFTRACER_INC_METADATA value (default: 1)
  --dftracer-data-dir PATHS  DFTRACER_DATA_DIR value (colon-separated)
  --dftracer-log-prefix PATH DFTRACER_LOG_FILE prefix
  --dftracer-preload-lib PATH
                             Path to libdftracer_preload.so
  --dfanalyzer-enabled 0|1   Run dfanalyzer after DFTracer run (default: 1)
  --dfanalyzer-checkpoint-enabled 0|1
                             Enable dfanalyzer checkpoints (default: 1)
  --dfanalyzer-preset NAME   dfanalyzer preset (default: posix)
  --dfanalyzer-checkpoint-dir PATH
                             dfanalyzer checkpoint directory

  -h, --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --np) NP="$2"; shift 2 ;;
    --mpiexec) MPIEXEC="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --records) RECORDS="$2"; shift 2 ;;
    --mode-flag) MODE_FLAG="$2"; shift 2 ;;
    --keep-output) KEEP_OUTPUT="$2"; shift 2 ;;
    --dftracer-enable) DFTRACER_ENABLE="$2"; shift 2 ;;
    --dftracer-init) DFTRACER_INIT="$2"; shift 2 ;;
    --dftracer-inc-metadata) DFTRACER_INC_METADATA="$2"; shift 2 ;;
    --dftracer-data-dir) DFTRACER_DATA_DIR="$2"; shift 2 ;;
    --dftracer-log-prefix) DFTRACER_LOG_PREFIX="$2"; shift 2 ;;
    --dftracer-preload-lib) DFTRACER_PRELOAD_LIB="$2"; shift 2 ;;
    --dfanalyzer-enabled) DFANALYZER_ENABLED="$2"; shift 2 ;;
    --dfanalyzer-checkpoint-enabled) DFANALYZER_CHECKPOINT_ENABLED="$2"; shift 2 ;;
    --dfanalyzer-preset) DFANALYZER_PRESET="$2"; shift 2 ;;
    --dfanalyzer-checkpoint-dir) DFANALYZER_CHECKPOINT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

source "${ROOT_DIR}/setup_env.sh"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${RUNS_DIR}"

if [[ ! -x "${BIN}" ]]; then
  ALT_BIN="${ROOT_DIR}/e3sm-io/e3sm_io"
  if [[ -x "${ALT_BIN}" ]]; then
    BIN="${ALT_BIN}"
  else
    echo "Missing executable: ${BIN}. Run ./build.sh first." >&2
    exit 1
  fi
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "Missing dataset: ${DATASET}" >&2
  exit 1
fi

if [[ "${MODE_FLAG}" != "-n" && "${MODE_FLAG}" != "-d" ]]; then
  echo "Invalid --mode-flag: ${MODE_FLAG}. Use -n or -d." >&2
  exit 1
fi

RUN_KIND="baseline"
if [[ "${DFTRACER_ENABLE}" == "1" ]]; then
  RUN_KIND="dftracer"
fi

RUN_LOG_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_FILE="${RUN_LOG_DIR}/output.log"
CONFIG_FILE="${RUN_LOG_DIR}/config.json"
RUN_OUT_DIR="${OUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_LOG_DIR}" "${RUN_OUT_DIR}"

if [[ -z "${DFTRACER_DATA_DIR}" ]]; then
  DFTRACER_DATA_DIR="${OUT_DIR}:${ROOT_DIR}/e3sm-io/datasets"
fi
if [[ -z "${DFTRACER_LOG_PREFIX}" ]]; then
  DFTRACER_LOG_PREFIX="${RUN_LOG_DIR}/trace"
fi
if [[ -z "${DFANALYZER_CHECKPOINT_DIR}" ]]; then
  DFANALYZER_CHECKPOINT_DIR="${ROOT_DIR}/tmp/dfanalyzer_${RUN_ID}"
fi

CMD=("${MPIEXEC}" -n "${NP}" "${BIN}")
if [[ "${KEEP_OUTPUT}" == "1" ]]; then
  CMD+=(-k)
fi
CMD+=("${MODE_FLAG}" -r "${RECORDS}" -o "${RUN_OUT_DIR}" "${DATASET}")

ENV_FILE="${RUN_LOG_DIR}/env.txt"
{
  echo "RUN_ID=${RUN_ID}"
  echo "RUN_KIND=${RUN_KIND}"
  echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
  echo "RUN_OUT_DIR=${RUN_OUT_DIR}"
  echo "DATASET=${DATASET}"
  echo "BIN=${BIN}"
  echo "COMMAND=${CMD[*]}"
  echo "DFTRACER_ENABLE=${DFTRACER_ENABLE}"
  echo "DFTRACER_INIT=${DFTRACER_INIT}"
  echo "DFTRACER_INC_METADATA=${DFTRACER_INC_METADATA}"
  echo "DFTRACER_DATA_DIR=${DFTRACER_DATA_DIR}"
  echo "DFTRACER_LOG_FILE=${DFTRACER_LOG_PREFIX}"
  echo "DFTRACER_PRELOAD_LIB=${DFTRACER_PRELOAD_LIB}"
  echo "DFANALYZER_ENABLED=${DFANALYZER_ENABLED}"
  echo "DFANALYZER_CHECKPOINT_ENABLED=${DFANALYZER_CHECKPOINT_ENABLED}"
  echo "DFANALYZER_PRESET=${DFANALYZER_PRESET}"
  echo "DFANALYZER_CHECKPOINT_DIR=${DFANALYZER_CHECKPOINT_DIR}"
  echo ""
  env | sort
} > "${ENV_FILE}"

{
  echo "run_id=${RUN_ID}"
  echo "mode=${RUN_KIND}"
  echo "dataset=${DATASET}"
  echo "output_dir=${RUN_OUT_DIR}"
  echo "command=${CMD[*]}"
  if [[ "${DFTRACER_ENABLE}" == "1" ]]; then
    if [[ ! -f "${DFTRACER_PRELOAD_LIB}" ]]; then
      echo "Missing DFTracer preload library: ${DFTRACER_PRELOAD_LIB}" >&2
      exit 1
    fi
    echo "DFTRACER_ENABLE=${DFTRACER_ENABLE}"
    echo "DFTRACER_INIT=${DFTRACER_INIT}"
    echo "DFTRACER_INC_METADATA=${DFTRACER_INC_METADATA}"
    echo "DFTRACER_DATA_DIR=${DFTRACER_DATA_DIR}"
    echo "DFTRACER_LOG_FILE=${DFTRACER_LOG_PREFIX}"
    echo "LD_PRELOAD=${DFTRACER_PRELOAD_LIB}"
    env \
      DFTRACER_ENABLE="${DFTRACER_ENABLE}" \
      DFTRACER_INIT="${DFTRACER_INIT}" \
      DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA}" \
      DFTRACER_DATA_DIR="${DFTRACER_DATA_DIR}" \
      DFTRACER_LOG_FILE="${DFTRACER_LOG_PREFIX}" \
      LD_PRELOAD="${DFTRACER_PRELOAD_LIB}" \
      "${CMD[@]}"
  else
    "${CMD[@]}"
  fi
} | tee "${LOG_FILE}"

DFANALYZER_OUTPUT="${RUN_LOG_DIR}/dfanalyzer_output.txt"
if [[ "${DFTRACER_ENABLE}" == "1" && "${DFANALYZER_ENABLED}" == "1" ]]; then
  if command -v dfanalyzer >/dev/null 2>&1; then
    CHECKPOINT_BOOL="false"
    if [[ "${DFANALYZER_CHECKPOINT_ENABLED}" == "1" ]]; then
      CHECKPOINT_BOOL="true"
      mkdir -p "${DFANALYZER_CHECKPOINT_DIR}"
    fi
    {
      echo "dfanalyzer trace_path=${RUN_LOG_DIR}"
      echo "dfanalyzer preset=${DFANALYZER_PRESET}"
      echo "dfanalyzer checkpoint=${CHECKPOINT_BOOL}"
      echo "dfanalyzer checkpoint_dir=${DFANALYZER_CHECKPOINT_DIR}"
      dfanalyzer \
        analyzer=dftracer \
        analyzer/preset="${DFANALYZER_PRESET}" \
        trace_path="${RUN_LOG_DIR}" \
        analyzer.checkpoint="${CHECKPOINT_BOOL}" \
        analyzer.checkpoint_dir="${DFANALYZER_CHECKPOINT_DIR}"
    } > "${DFANALYZER_OUTPUT}" 2>&1 || true
  else
    echo "dfanalyzer command not found in environment" > "${DFANALYZER_OUTPUT}"
  fi
fi

cat > "${CONFIG_FILE}" <<EOF
{
  "run_id": "${RUN_ID}",
  "workload": "e3sm-io",
  "mode": "${RUN_KIND}",
  "status": "completed",
  "command": "${CMD[*]}",
  "dataset": "${DATASET}",
  "output_dir": "${RUN_OUT_DIR}",
  "trace_prefix": "${DFTRACER_LOG_PREFIX}",
  "dftracer_enabled": ${DFTRACER_ENABLE},
  "dftracer_inc_metadata": ${DFTRACER_INC_METADATA},
  "env_file": "${ENV_FILE}",
  "dfanalyzer_enabled": ${DFANALYZER_ENABLED},
  "dfanalyzer_output": "${DFANALYZER_OUTPUT}",
  "dfanalyzer_checkpoint_enabled": ${DFANALYZER_CHECKPOINT_ENABLED},
  "dfanalyzer_checkpoint_dir": "${DFANALYZER_CHECKPOINT_DIR}",
  "knobs_file": "dftracer/knobs.yaml"
}
EOF

echo "Run recorded. Logs under: ${RUN_LOG_DIR}"
