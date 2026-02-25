#!/usr/bin/env bash
set -euo pipefail

# run.sh — Unified OpenFOAM pitzDaily runner with optional DFTracer tracing.
#
# Key I/O knobs this script exercises:
#   --io-mode uncollated   Creates processorN/ directories (one per rank).
#                          Demonstrates the "metadata storm" I/O pattern.
#   --io-mode collated     Creates a single processorsN/ directory.
#                          Reduces file-count at scale; different I/O profile.
#   --write-interval N     Controls how often field snapshots are written.
#   --np N                 Number of MPI ranks (affects decomposition + file count).
#
# With --dftracer-enable 1 the run is wrapped via LD_PRELOAD so that all
# POSIX I/O calls are captured to a Chrome trace JSON file.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs"
OUT_DIR="${ROOT_DIR}/output"
RUNS_DIR="${LOG_DIR}/runs"

# --------------------------------------------------------------------------
# Defaults (all overridable via CLI or environment)
# --------------------------------------------------------------------------
NP="${NP:-4}"
MPIEXEC="${MPIEXEC:-mpirun}"
CASE_BASE="${ROOT_DIR}/output/cases/pitzDaily"

# I/O knobs
IO_MODE="${IO_MODE:-uncollated}"      # uncollated | collated
WRITE_INTERVAL="${WRITE_INTERVAL:-50}"
END_TIME="${END_TIME:-100}"

# DFTracer knobs
DFTRACER_ENABLE="${DFTRACER_ENABLE:-0}"
DFTRACER_INIT="${DFTRACER_INIT:-PRELOAD}"
DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA:-1}"
DFTRACER_DATA_DIR="${DFTRACER_DATA_DIR:-}"
DFTRACER_LOG_PREFIX="${DFTRACER_LOG_PREFIX:-}"
# Prefer python3.11 site-packages path; user can override via env or --flag.
DFTRACER_PRELOAD_LIB="${DFTRACER_PRELOAD_LIB:-}"

# --------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

I/O options:
  --np N                     MPI ranks (default: 4)
  --mpiexec CMD              MPI launcher (default: mpirun)
  --case PATH                Base case directory (default: output/cases/pitzDaily)
  --io-mode MODE             uncollated or collated (default: uncollated)
  --write-interval N         writeInterval in controlDict (default: 50)
  --end-time N               endTime in controlDict (default: 100)

DFTracer options:
  --dftracer-enable 0|1      Enable DFTracer via LD_PRELOAD (default: 0)
  --dftracer-init MODE       DFTRACER_INIT value (default: PRELOAD)
  --dftracer-inc-metadata 0|1
  --dftracer-data-dir PATHS  DFTRACER_DATA_DIR (colon-separated paths)
  --dftracer-log-prefix PATH DFTRACER_LOG_FILE prefix
  --dftracer-preload-lib PATH  Path to libdftracer_preload.so

  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --np)                    NP="$2";                    shift 2 ;;
    --mpiexec)               MPIEXEC="$2";               shift 2 ;;
    --case)                  CASE_BASE="$2";              shift 2 ;;
    --io-mode)               IO_MODE="$2";               shift 2 ;;
    --write-interval)        WRITE_INTERVAL="$2";        shift 2 ;;
    --end-time)              END_TIME="$2";              shift 2 ;;
    --dftracer-enable)       DFTRACER_ENABLE="$2";       shift 2 ;;
    --dftracer-init)         DFTRACER_INIT="$2";         shift 2 ;;
    --dftracer-inc-metadata) DFTRACER_INC_METADATA="$2"; shift 2 ;;
    --dftracer-data-dir)     DFTRACER_DATA_DIR="$2";     shift 2 ;;
    --dftracer-log-prefix)   DFTRACER_LOG_PREFIX="$2";   shift 2 ;;
    --dftracer-preload-lib)  DFTRACER_PRELOAD_LIB="$2";  shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# --------------------------------------------------------------------------
# Validate
# --------------------------------------------------------------------------
if [[ "${IO_MODE}" != "uncollated" && "${IO_MODE}" != "collated" ]]; then
  echo "ERROR: --io-mode must be 'uncollated' or 'collated'. Got: ${IO_MODE}" >&2
  exit 1
fi

# --------------------------------------------------------------------------
source "${ROOT_DIR}/setup_env.sh"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${RUNS_DIR}"

if [[ ! -d "${CASE_BASE}" ]]; then
  echo "ERROR: Base case not found at ${CASE_BASE}." >&2
  echo "       Run ./build.sh first." >&2
  exit 1
fi

# --------------------------------------------------------------------------
# Resolve DFTracer preload library path
# --------------------------------------------------------------------------
if [[ -z "${DFTRACER_PRELOAD_LIB}" ]]; then
  # Search common paths inside .venv for the preload library.
  for py_ver in python3.12 python3.11 python3.10 python3.9; do
    candidate="${ROOT_DIR}/.venv/lib/${py_ver}/site-packages/dftracer/lib/libdftracer_preload.so"
    if [[ -f "${candidate}" ]]; then
      DFTRACER_PRELOAD_LIB="${candidate}"
      break
    fi
  done
fi

# --------------------------------------------------------------------------
# Set up per-run directories
# --------------------------------------------------------------------------
RUN_KIND="baseline"
[[ "${DFTRACER_ENABLE}" == "1" ]] && RUN_KIND="dftracer"

RUN_LOG_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_FILE="${RUN_LOG_DIR}/output.log"
CONFIG_FILE="${RUN_LOG_DIR}/config.json"
CASE_DIR="${OUT_DIR}/runs/${RUN_ID}/pitzDaily"

mkdir -p "${RUN_LOG_DIR}" "$(dirname "${CASE_DIR}")"

# --------------------------------------------------------------------------
# Copy base case and configure it
# --------------------------------------------------------------------------
cp -r "${CASE_BASE}" "${CASE_DIR}"

# Set DFTracer data dir and log prefix defaults (use case dir for data)
[[ -z "${DFTRACER_DATA_DIR}" ]]  && DFTRACER_DATA_DIR="${CASE_DIR}"
[[ -z "${DFTRACER_LOG_PREFIX}" ]] && DFTRACER_LOG_PREFIX="${RUN_LOG_DIR}/trace"

# Patch system/controlDict:
#   - writeInterval: controls snapshot frequency (key I/O knob)
#   - endTime:       total simulation time
#   - fileHandler:   uncollated (processorN/) vs collated (processorsN/)
patch_dict() {
  local file="$1" key="$2" value="$3"
  if grep -qE "^[[:space:]]*${key}[[:space:]]" "${file}"; then
    sed -i -E "s|^([[:space:]]*${key}[[:space:]]+)[^;]*(;)|\1${value}\2|" "${file}"
  else
    printf '\n%s    %s;\n' "${key}" "${value}" >> "${file}"
  fi
}

CONTROL_DICT="${CASE_DIR}/system/controlDict"
patch_dict "${CONTROL_DICT}" "writeInterval" "${WRITE_INTERVAL}"
patch_dict "${CONTROL_DICT}" "endTime"        "${END_TIME}"
patch_dict "${CONTROL_DICT}" "fileHandler"    "${IO_MODE}"

# Write decomposeParDict using scotch method (geometry-agnostic).
cat > "${CASE_DIR}/system/decomposeParDict" <<DECOMPOSE_DICT
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}

numberOfSubdomains  ${NP};

method          scotch;
DECOMPOSE_DICT

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
{
  echo "=== OpenFOAM run.sh: run_id=${RUN_ID} mode=${RUN_KIND} ==="
  echo "case_dir=${CASE_DIR}"
  echo "np=${NP}  io_mode=${IO_MODE}  write_interval=${WRITE_INTERVAL}  end_time=${END_TIME}"

  cd "${CASE_DIR}"

  # Decompose domain if parallel
  if [[ "${NP}" -gt 1 ]]; then
    echo "--- decomposePar (NP=${NP}) ---"
    decomposePar -force
  fi

  # Determine solver command: foamRun reads 'solver' from system/controlDict,
  # falling back to the 'application' entry for older OF layouts.
  if command -v foamRun &>/dev/null; then
    if [[ "${NP}" -gt 1 ]]; then
      RUN_CMD=("${MPIEXEC}" -n "${NP}" foamRun -parallel)
    else
      RUN_CMD=(foamRun)
    fi
  else
    # Older layout: read application name from controlDict
    APP=$(grep -m1 "^application" "${CASE_DIR}/system/controlDict" \
          | awk '{print $2}' | tr -d ';')
    if [[ "${NP}" -gt 1 ]]; then
      RUN_CMD=("${MPIEXEC}" -n "${NP}" "${APP}" -parallel)
    else
      RUN_CMD=("${APP}")
    fi
  fi

  echo "command=${RUN_CMD[*]}"

  if [[ "${DFTRACER_ENABLE}" == "1" ]]; then
    if [[ -z "${DFTRACER_PRELOAD_LIB}" || ! -f "${DFTRACER_PRELOAD_LIB}" ]]; then
      echo "ERROR: DFTracer preload library not found." >&2
      echo "       Set --dftracer-preload-lib or ensure dftracer is installed in .venv." >&2
      exit 1
    fi
    echo "DFTRACER_ENABLE=1"
    echo "DFTRACER_DATA_DIR=${DFTRACER_DATA_DIR}"
    echo "DFTRACER_LOG_FILE=${DFTRACER_LOG_PREFIX}"
    echo "LD_PRELOAD=${DFTRACER_PRELOAD_LIB}"
    echo "--- solver (DFTracer enabled) ---"
    env \
      DFTRACER_ENABLE="${DFTRACER_ENABLE}" \
      DFTRACER_INIT="${DFTRACER_INIT}" \
      DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA}" \
      DFTRACER_DATA_DIR="${DFTRACER_DATA_DIR}" \
      DFTRACER_LOG_FILE="${DFTRACER_LOG_PREFIX}" \
      LD_PRELOAD="${DFTRACER_PRELOAD_LIB}" \
      "${RUN_CMD[@]}"
  else
    echo "--- solver (baseline) ---"
    "${RUN_CMD[@]}"
  fi

} 2>&1 | tee "${LOG_FILE}"

# --------------------------------------------------------------------------
# Write run summary JSON (per DFTRACER.md schema)
# --------------------------------------------------------------------------
cat > "${CONFIG_FILE}" <<EOF
{
  "run_id": "${RUN_ID}",
  "workload": "openfoam",
  "mode": "${RUN_KIND}",
  "status": "completed",
  "case_dir": "${CASE_DIR}",
  "np": ${NP},
  "io_mode": "${IO_MODE}",
  "write_interval": ${WRITE_INTERVAL},
  "end_time": ${END_TIME},
  "dftracer_enabled": ${DFTRACER_ENABLE},
  "trace_prefix": "${DFTRACER_LOG_PREFIX}",
  "dftracer_data_dir": "${DFTRACER_DATA_DIR}",
  "knobs_file": "dftracer/knobs.yaml"
}
EOF

echo ""
echo "Run complete. Logs under: ${RUN_LOG_DIR}"
echo "Case output under: ${CASE_DIR}"
