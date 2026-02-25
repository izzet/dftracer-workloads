#!/usr/bin/env bash
# run.sh — unified LAMMPS runner for baseline and DFTracer-traced runs.
#
# Workflow:
#   1. Load env (setup_env.sh + build_info.env)
#   2. Copy the prepared case directory into a fresh per-run working directory
#   3. Run lmp -in in.lj_melt (with optional DFTracer LD_PRELOAD)
#   4. Move dump/restart files to output/<run_id>/
#   5. Optionally run dfanalyzer on traces
#   6. Write run summary JSON to logs/runs/<run_id>/config.json
#
# Usage:
#   ./run.sh [options]
#
# Examples:
#   ./run.sh                                          # baseline, 4 ranks
#   ./run.sh --np 8                                  # baseline, 8 ranks
#   ./run.sh --dftracer-enable 1                     # traced run
#   DFTRACER_ENABLE=1 ./run.sh                       # same via env var
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs"
OUT_DIR="${ROOT_DIR}/output"
RUNS_DIR="${LOG_DIR}/runs"

# ---------------------------------------------------------------------------
# Defaults (can be overridden by env vars or CLI flags)
# ---------------------------------------------------------------------------
MPIEXEC="${MPIEXEC:-mpiexec}"
NP="${NP:-4}"
CASE_NAME="${CASE_NAME:-lj_melt}"
CASE_DIR="${CASE_DIR:-${ROOT_DIR}/run/${CASE_NAME}}"
INPUT_FILE="${INPUT_FILE:-in.lj_melt}"

DFTRACER_ENABLE="${DFTRACER_ENABLE:-0}"
DFTRACER_INIT="${DFTRACER_INIT:-PRELOAD}"
DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA:-1}"
DFTRACER_LOG_LEVEL="${DFTRACER_LOG_LEVEL:-ERROR}"
DFTRACER_DATA_DIR=""
DFTRACER_LOG_PREFIX=""
# Resolve preload lib dynamically
_PRELOAD_DEFAULT=""
if [[ -d "${ROOT_DIR}/.venv/lib" ]]; then
  _PRELOAD_DEFAULT="$(find "${ROOT_DIR}/.venv/lib" -maxdepth 3 \
      -name "libdftracer_preload.so" 2>/dev/null | head -1)"
fi
# Fallback: try common Python site-packages paths if find found nothing
if [[ -z "${_PRELOAD_DEFAULT}" ]]; then
  for _py in python3.12 python3.11; do
    _p="${ROOT_DIR}/.venv/lib/${_py}/site-packages/dftracer/lib/libdftracer_preload.so"
    [[ -f "${_p}" ]] && _PRELOAD_DEFAULT="${_p}" && break
  done
fi
DFTRACER_PRELOAD_LIB="${DFTRACER_PRELOAD_LIB:-${_PRELOAD_DEFAULT}}"

DFANALYZER_ENABLED="${DFANALYZER_ENABLED:-1}"
DFANALYZER_CHECKPOINT_ENABLED="${DFANALYZER_CHECKPOINT_ENABLED:-1}"
DFANALYZER_PRESET="${DFANALYZER_PRESET:-posix}"
DFANALYZER_CHECKPOINT_DIR=""

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

Options:
  --np N                     MPI ranks (default: 4)
  --mpiexec CMD              MPI launcher (default: mpiexec)
  --case-dir PATH            Prepared case directory (default: run/lj_melt)
  --case-name NAME           Case name (default: lj_melt)
  --input FILE               LAMMPS input script (default: in.lj_melt)

  --dftracer-enable 0|1      Enable DFTracer via LD_PRELOAD (default: 0)
  --dftracer-init MODE       DFTRACER_INIT value (default: PRELOAD)
  --dftracer-inc-metadata 0|1
  --dftracer-log-level LVL   DFTRACER_LOG_LEVEL (default: ERROR)
  --dftracer-data-dir PATHS  Colon-separated paths for DFTRACER_DATA_DIR
  --dftracer-log-prefix PATH DFTRACER_LOG_FILE prefix
  --dftracer-preload-lib PATH Path to libdftracer_preload.so

  --dfanalyzer-enabled 0|1   Run dfanalyzer post-trace (default: 1)
  --dfanalyzer-checkpoint-enabled 0|1
  --dfanalyzer-preset NAME   dfanalyzer preset (default: posix)
  --dfanalyzer-checkpoint-dir PATH

  -h, --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --np)                          NP="$2"; shift 2 ;;
    --mpiexec)                     MPIEXEC="$2"; shift 2 ;;
    --case-dir)                    CASE_DIR="$2"; shift 2 ;;
    --case-name)                   CASE_NAME="$2"; shift 2 ;;
    --input)                      INPUT_FILE="$2"; shift 2 ;;
    --dftracer-enable)             DFTRACER_ENABLE="$2"; shift 2 ;;
    --dftracer-init)               DFTRACER_INIT="$2"; shift 2 ;;
    --dftracer-inc-metadata)       DFTRACER_INC_METADATA="$2"; shift 2 ;;
    --dftracer-log-level)         DFTRACER_LOG_LEVEL="$2"; shift 2 ;;
    --dftracer-data-dir)          DFTRACER_DATA_DIR="$2"; shift 2 ;;
    --dftracer-log-prefix)         DFTRACER_LOG_PREFIX="$2"; shift 2 ;;
    --dftracer-preload-lib)        DFTRACER_PRELOAD_LIB="$2"; shift 2 ;;
    --dfanalyzer-enabled)          DFANALYZER_ENABLED="$2"; shift 2 ;;
    --dfanalyzer-checkpoint-enabled) DFANALYZER_CHECKPOINT_ENABLED="$2"; shift 2 ;;
    --dfanalyzer-preset)           DFANALYZER_PRESET="$2"; shift 2 ;;
    --dfanalyzer-checkpoint-dir)   DFANALYZER_CHECKPOINT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# Use debug preload when LOG_LEVEL is DEBUG
if [[ "${DFTRACER_ENABLE}" == "1" ]] && [[ "${DFTRACER_LOG_LEVEL}" == "DEBUG" ]]; then
  _DBG_PRELOAD="${DFTRACER_PRELOAD_LIB%.so}_dbg.so"
  if [[ -f "${_DBG_PRELOAD}" ]]; then
    DFTRACER_PRELOAD_LIB="${_DBG_PRELOAD}"
  fi
fi

# ---------------------------------------------------------------------------
# Activate environment
# ---------------------------------------------------------------------------
source "${ROOT_DIR}/setup_env.sh"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${RUNS_DIR}"

RUN_KIND="baseline"
if [[ "${DFTRACER_ENABLE}" == "1" ]]; then
  RUN_KIND="dftracer"
fi

RUN_LOG_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_FILE="${RUN_LOG_DIR}/output.log"
CONFIG_FILE="${RUN_LOG_DIR}/config.json"
RUN_OUT_DIR="${OUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_LOG_DIR}" "${RUN_OUT_DIR}"

# ---------------------------------------------------------------------------
# Validate case directory
# ---------------------------------------------------------------------------
if [[ ! -d "${CASE_DIR}" ]]; then
  echo "ERROR: Case directory not found: ${CASE_DIR}" >&2
  echo "Run ./prepare_case.sh first." >&2
  exit 1
fi

LMP_LINK="${CASE_DIR}/lmp"
INPUT_PATH="${CASE_DIR}/${INPUT_FILE}"

if [[ ! -x "${LMP_LINK}" ]]; then
  echo "ERROR: lmp not found or not executable in ${CASE_DIR}" >&2
  echo "Run ./prepare_case.sh to set up the case directory." >&2
  exit 1
fi

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "ERROR: Input script not found: ${INPUT_PATH}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Create per-run working directory
# ---------------------------------------------------------------------------
WORK_DIR="${ROOT_DIR}/run/runs/${RUN_ID}"
log_msg() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log_msg "Copying case template ${CASE_DIR} -> ${WORK_DIR}"
mkdir -p "$(dirname "${WORK_DIR}")"
cp -a "${CASE_DIR}" "${WORK_DIR}"

# ---------------------------------------------------------------------------
# Inject full paths into LAMMPS input (fixes DFTracer path filtering with
# DFTRACER_DATA_DIR — relative paths like "dump.melt.lammpstrj" are filtered out)
# ---------------------------------------------------------------------------
INPUT_PATH="${WORK_DIR}/${INPUT_FILE}"
if [[ -f "${INPUT_PATH}" ]]; then
  log_msg "Injecting full I/O paths into ${INPUT_FILE}"
  sed -i \
    -e "s|dump\.melt\.lammpstrj|${WORK_DIR}/dump.melt.lammpstrj|g" \
    -e "s|restart\.lj_melt\.restart|${WORK_DIR}/restart.lj_melt.restart|g" \
    "${INPUT_PATH}"
fi

# ---------------------------------------------------------------------------
# DFTracer path setup
# ---------------------------------------------------------------------------
if [[ -z "${DFTRACER_DATA_DIR}" ]]; then
  DFTRACER_DATA_DIR="${WORK_DIR}:${RUN_OUT_DIR}"
fi
if [[ -z "${DFTRACER_LOG_PREFIX}" ]]; then
  DFTRACER_LOG_PREFIX="${RUN_LOG_DIR}/trace"
fi
if [[ -z "${DFANALYZER_CHECKPOINT_DIR}" ]]; then
  DFANALYZER_CHECKPOINT_DIR="${ROOT_DIR}/tmp/dfanalyzer_${RUN_ID}"
fi

# ---------------------------------------------------------------------------
# Build LAMMPS command
# ---------------------------------------------------------------------------
# Use full path for input so DFTracer path filter captures the read
LMP_CMD=("${MPIEXEC}" -n "${NP}" "${WORK_DIR}/lmp" -in "${WORK_DIR}/${INPUT_FILE}")

ENV_FILE="${RUN_LOG_DIR}/env.txt"
{
  echo "RUN_ID=${RUN_ID}"
  echo "RUN_KIND=${RUN_KIND}"
  echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
  echo "RUN_OUT_DIR=${RUN_OUT_DIR}"
  echo "WORK_DIR=${WORK_DIR}"
  echo "CASE_DIR=${CASE_DIR}"
  echo "NP=${NP}"
  echo "MPIEXEC=${MPIEXEC}"
  echo "LMP_CMD=${LMP_CMD[*]}"
  echo "DFTRACER_ENABLE=${DFTRACER_ENABLE}"
  echo "DFTRACER_INIT=${DFTRACER_INIT}"
  echo "DFTRACER_INC_METADATA=${DFTRACER_INC_METADATA}"
  echo "DFTRACER_LOG_LEVEL=${DFTRACER_LOG_LEVEL}"
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

# ---------------------------------------------------------------------------
# Run LAMMPS
# ---------------------------------------------------------------------------
{
  log_msg "=== LAMMPS run (${NP} ranks) ==="
  log_msg "Working dir: ${WORK_DIR}"
  log_msg "Input: ${INPUT_FILE}"
  if [[ "${DFTRACER_ENABLE}" == "1" ]]; then
    if [[ ! -f "${DFTRACER_PRELOAD_LIB}" ]]; then
      log_msg "ERROR: DFTracer preload library not found: ${DFTRACER_PRELOAD_LIB}"
      exit 1
    fi
    log_msg "DFTracer enabled: LD_PRELOAD=${DFTRACER_PRELOAD_LIB}"
    (
      cd "${WORK_DIR}"
      env \
        DFTRACER_ENABLE="${DFTRACER_ENABLE}" \
        DFTRACER_INIT="${DFTRACER_INIT}" \
        DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA}" \
        DFTRACER_LOG_LEVEL="${DFTRACER_LOG_LEVEL}" \
        DFTRACER_DATA_DIR="${DFTRACER_DATA_DIR}" \
        DFTRACER_LOG_FILE="${DFTRACER_LOG_PREFIX}" \
        LD_PRELOAD="${DFTRACER_PRELOAD_LIB}" \
        "${LMP_CMD[@]}"
    )
  else
    (cd "${WORK_DIR}" && "${LMP_CMD[@]}")
  fi
  log_msg "=== LAMMPS done ==="
} 2>&1 | tee "${LOG_FILE}"

# ---------------------------------------------------------------------------
# Move LAMMPS output to output/<run_id>/
# ---------------------------------------------------------------------------
log_msg "Moving dump/restart files to ${RUN_OUT_DIR} ..."
find "${WORK_DIR}" -maxdepth 1 \
  \( -name "dump.*" -o -name "restart.*" \) \
  -exec mv {} "${RUN_OUT_DIR}/" \; 2>/dev/null || true
log_msg "Output files: $(ls "${RUN_OUT_DIR}" 2>/dev/null | wc -l) file(s) in ${RUN_OUT_DIR}"

# ---------------------------------------------------------------------------
# Optional: dfanalyzer post-processing
# ---------------------------------------------------------------------------
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
        "analyzer/preset=${DFANALYZER_PRESET}" \
        "trace_path=${RUN_LOG_DIR}" \
        "analyzer.checkpoint=${CHECKPOINT_BOOL}" \
        "analyzer.checkpoint_dir=${DFANALYZER_CHECKPOINT_DIR}"
    } > "${DFANALYZER_OUTPUT}" 2>&1 || true
  else
    echo "dfanalyzer command not found in environment" > "${DFANALYZER_OUTPUT}"
  fi
fi

# ---------------------------------------------------------------------------
# Write run summary JSON
# ---------------------------------------------------------------------------
cat > "${CONFIG_FILE}" << EOF
{
  "run_id": "${RUN_ID}",
  "workload": "lammps",
  "case": "${CASE_NAME}",
  "mode": "${RUN_KIND}",
  "status": "completed",
  "np": ${NP},
  "mpiexec": "${MPIEXEC}",
  "work_dir": "${WORK_DIR}",
  "output_dir": "${RUN_OUT_DIR}",
  "trace_prefix": "${DFTRACER_LOG_PREFIX}",
  "dftracer_enabled": ${DFTRACER_ENABLE},
  "dftracer_data_dir": "${DFTRACER_DATA_DIR}",
  "dftracer_inc_metadata": ${DFTRACER_INC_METADATA},
  "env_file": "${ENV_FILE}",
  "dfanalyzer_enabled": ${DFANALYZER_ENABLED},
  "dfanalyzer_output": "${DFANALYZER_OUTPUT}",
  "dfanalyzer_checkpoint_enabled": ${DFANALYZER_CHECKPOINT_ENABLED},
  "dfanalyzer_checkpoint_dir": "${DFANALYZER_CHECKPOINT_DIR}",
  "knobs_file": "dftracer/knobs.yaml"
}
EOF

log_msg "Run complete."
log_msg "  Log dir:    ${RUN_LOG_DIR}"
log_msg "  Output dir: ${RUN_OUT_DIR}"
log_msg "  Summary:    ${CONFIG_FILE}"
