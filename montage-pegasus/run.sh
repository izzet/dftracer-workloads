#!/usr/bin/env bash
set -euo pipefail
#
# Run the Montage Pegasus workflow (montage-workflow.py -> pegasus-run).
# Follows DFTracer docs/pegasus_montage.rst. Requires Condor running.
#
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="${ROOT_DIR}/montage-workflow-v3"
LOG_DIR="${ROOT_DIR}/logs"
OUT_DIR="${ROOT_DIR}/output"
RUNS_DIR="${LOG_DIR}/runs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${RUNS_DIR}/${RUN_ID}"
TRACE_DIR="${LOG_DIR}/traces/${RUN_ID}"

# Workflow params (2 deg mosaic, DSS bands - from montage-workflow-v3 example)
CENTER="${MONTAGE_CENTER:-56.7 24.0}"
DEGREES="${MONTAGE_DEGREES:-2.0}"
BANDS="${MONTAGE_BANDS:---band dss:DSS2B:blue --band dss:DSS2R:green --band dss:DSS2IR:red}"

# DFTracer (Montage uses LD_LIBRARY_PATH + DFTRACER_INSTALLED; LD_PRELOAD optional)
DFTRACER_ENABLE="${DFTRACER_ENABLE:-1}"
DFTRACER_INIT="${DFTRACER_INIT:-PRELOAD}"
DFTRACER_INC_METADATA="${DFTRACER_INC_METADATA:-1}"
DFTRACER_BIND_SIGNALS="${DFTRACER_BIND_SIGNALS:-0}"
DFTRACER_TRACE_COMPRESSION="${DFTRACER_TRACE_COMPRESSION:-1}"
DFTRACER_DATA_DIR=""
DFTRACER_LOG_PREFIX=""
DFTRACER_PRELOAD_LIB="${DFTRACER_PRELOAD_LIB:-}"

# dfanalyzer
DFANALYZER_ENABLED="${DFANALYZER_ENABLED:-1}"
DFANALYZER_CHECKPOINT_ENABLED="${DFANALYZER_CHECKPOINT_ENABLED:-1}"
DFANALYZER_PRESET="${DFANALYZER_PRESET:-posix}"
DFANALYZER_CHECKPOINT_DIR=""

# Wait for workflow completion before running dfanalyzer
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"

usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

Runs montage-workflow.py to create data, plans, then pegasus-run.

Options:
  --center "RA DEC"   Center coords (default: 56.7 24.0)
  --degrees N        Mosaic size in degrees (default: 2.0)
  --wait             Wait for workflow completion before exit (for dfanalyzer)

  --dftracer-enable 0|1      Enable DFTracer (default: 1)
  --dftracer-init MODE      DFTRACER_INIT value (default: PRELOAD)
  --dftracer-inc-metadata 0|1
                            DFTRACER_INC_METADATA value (default: 1)
  --dftracer-data-dir PATHS DFTRACER_DATA_DIR (colon-separated)
  --dftracer-log-prefix PATH
                            DFTRACER_LOG_FILE prefix
  --dftracer-preload-lib PATH
                            Path to libdftracer_preload.so (optional)

  --dfanalyzer-enabled 0|1  Run dfanalyzer after workflow (default: 1)
  --dfanalyzer-checkpoint-enabled 0|1
                            Enable dfanalyzer checkpoints (default: 1)
  --dfanalyzer-preset NAME   dfanalyzer preset (default: posix)
  --dfanalyzer-checkpoint-dir PATH
                            dfanalyzer checkpoint directory

  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --center) CENTER="$2"; shift 2 ;;
    --degrees) DEGREES="$2"; shift 2 ;;
    --wait) WAIT_FOR_COMPLETION=1; shift ;;
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
    *) echo "Unknown: $1" >&2; usage; exit 1 ;;
  esac
done

source "${ROOT_DIR}/setup_env.sh"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${RUNS_DIR}" "${TRACE_DIR}" "${RUN_LOG_DIR}"

# ---- Condor setup (DFTracer pegasus_montage tutorial) ----
CONDOR_DIR="${ROOT_DIR}/condor"
INSTALL_DIR="${ROOT_DIR}/install"
if [[ -d "${CONDOR_DIR}" && -f "${CONDOR_DIR}/etc/condor_config" ]]; then
  export CONDOR_CONFIG="${CONDOR_DIR}/etc/condor_config"
  export PATH="${CONDOR_DIR}/bin:${CONDOR_DIR}/sbin:${PATH}"
elif [[ -d "${INSTALL_DIR}" && -f "${INSTALL_DIR}/etc/condor_config" ]]; then
  export CONDOR_CONFIG="${INSTALL_DIR}/etc/condor_config"
  export PATH="${INSTALL_DIR}/bin:${INSTALL_DIR}/sbin:${PATH}"
else
  echo "Condor not found. Run ./install.sh first." >&2
  exit 1
fi

# Ensure Condor is running (start if needed)
condor_running() {
  condor_status >/dev/null 2>&1
}
if ! condor_running; then
  echo "Condor not running. Starting condor_master ..."
  if [[ -n "${CONDOR_CONFIG:-}" ]]; then
    condor_master 2>/dev/null || true
    for _ in $(seq 1 15); do
      sleep 2
      condor_running && break
    done
  fi
fi
if ! condor_running; then
  echo "Condor failed to start. Run: . install/condor.sh && condor_master" >&2
  exit 1
fi

# ---- Preflight checks (DFTracer pegasus_montage tutorial) ----
preflight_ok=1
if [[ ! -d "${WORKFLOW_DIR}" ]]; then
  echo "Preflight: montage-workflow-v3 missing. Run ./install.sh first." >&2
  preflight_ok=0
fi
if ! command -v pegasus-version >/dev/null 2>&1; then
  echo "Preflight: Pegasus not in PATH. Run ./install.sh first." >&2
  preflight_ok=0
fi
if ! command -v mProject >/dev/null 2>&1; then
  echo "Preflight: Montage tools (mProject) not in PATH. Run ./build.sh first." >&2
  preflight_ok=0
fi
if ! command -v mViewer >/dev/null 2>&1; then
  echo "Preflight: mViewer not in PATH. Run ./build.sh (Viewer may need manual build)." >&2
  preflight_ok=0
fi
if ! condor_running; then
  echo "Preflight: Condor not running. Start with: . install/condor.sh && condor_master" >&2
  preflight_ok=0
fi
# Network check for IPAC archive (metadata + FITS downloads)
# mArchiveList uses montage-web.ipac.caltech.edu:80 (HTTP), not HTTPS
if ! curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "http://montage-web.ipac.caltech.edu" 2>/dev/null | grep -qE "200|301|302"; then
  echo "Preflight: Cannot reach montage-web.ipac.caltech.edu. Check network." >&2
  preflight_ok=0
fi
[[ "${preflight_ok}" -eq 0 ]] && exit 1

# Pegasus credentials for HTTP (Montage archive at montage-web.ipac.caltech.edu)
# Pegasus requires credentials.conf for HTTP stage-in jobs; public archive needs empty section
PEGASUS_CRED_DIR="${ROOT_DIR}/.pegasus"
mkdir -p "${PEGASUS_CRED_DIR}"
PEGASUS_CRED_FILE="${PEGASUS_CRED_DIR}/credentials.conf"
if [[ ! -f "${PEGASUS_CRED_FILE}" ]] || ! grep -q 'montage-web.ipac.caltech.edu' "${PEGASUS_CRED_FILE}" 2>/dev/null; then
  cat > "${PEGASUS_CRED_FILE}" << 'CREDEOF'
# Pegasus credentials for Montage workflow (public HTTP)
[http://montage-web.ipac.caltech.edu]
# Public archive - no auth required
CREDEOF
  chmod 600 "${PEGASUS_CRED_FILE}"
fi
export PEGASUS_CREDENTIALS="${PEGASUS_CRED_FILE}"

# Pegasus Condor/SLURM interface (run once if not configured)
if ! pegasus-configure-glite 2>/dev/null; then
  echo "Note: pegasus-configure-glite had issues (may already be configured)" >&2
fi

# DFTracer defaults (Montage workflow: traces go to TRACE_DIR)
if [[ -z "${DFTRACER_LOG_PREFIX}" ]]; then
  DFTRACER_LOG_PREFIX="${TRACE_DIR}/trace"
fi
if [[ -z "${DFANALYZER_CHECKPOINT_DIR}" ]]; then
  DFANALYZER_CHECKPOINT_DIR="${ROOT_DIR}/tmp/dfanalyzer_${RUN_ID}"
fi

RUN_KIND="baseline"
if [[ "${DFTRACER_ENABLE}" == "1" ]]; then
  RUN_KIND="dftracer"
fi

LOG_FILE="${RUN_LOG_DIR}/output.log"
CONFIG_FILE="${RUN_LOG_DIR}/config.json"
ENV_FILE="${RUN_LOG_DIR}/env.txt"

# Set DFTracer env for workflow jobs (Pegasus will pass these to Montage tasks)
export DFTRACER_ENABLE
export DFTRACER_INIT
export DFTRACER_INC_METADATA
export DFTRACER_BIND_SIGNALS
export DFTRACER_TRACE_COMPRESSION
export DFTRACER_LOG_FILE="${DFTRACER_LOG_PREFIX}"
export LD_LIBRARY_PATH="${DFTRACER_INSTALLED}/lib64:${DFTRACER_INSTALLED}/lib:${LD_LIBRARY_PATH:-}"

if [[ -n "${DFTRACER_DATA_DIR}" ]]; then
  export DFTRACER_DATA_DIR
fi
if [[ "${DFTRACER_ENABLE}" == "1" ]]; then
  export DFTRACER_LOG_FILE
  export DFTRACER_TRACE_COMPRESSION
fi

# Write env snapshot for debugging
{
  echo "RUN_ID=${RUN_ID}"
  echo "RUN_KIND=${RUN_KIND}"
  echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
  echo "TRACE_DIR=${TRACE_DIR}"
  echo "DFTRACER_ENABLE=${DFTRACER_ENABLE}"
  echo "DFTRACER_INIT=${DFTRACER_INIT}"
  echo "DFTRACER_INC_METADATA=${DFTRACER_INC_METADATA}"
  echo "DFTRACER_DATA_DIR=${DFTRACER_DATA_DIR}"
  echo "DFTRACER_LOG_FILE=${DFTRACER_LOG_PREFIX}"
  echo "DFANALYZER_ENABLED=${DFANALYZER_ENABLED}"
  echo "DFANALYZER_CHECKPOINT_ENABLED=${DFANALYZER_CHECKPOINT_ENABLED}"
  echo "DFANALYZER_PRESET=${DFANALYZER_PRESET}"
  echo "DFANALYZER_CHECKPOINT_DIR=${DFANALYZER_CHECKPOINT_DIR}"
  echo ""
  env | sort
} > "${ENV_FILE}"

cd "${WORKFLOW_DIR}"
log_msg() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

# montage-workflow.py fails if data/ exists
rm -rf data

log_msg "Creating workflow data (center=${CENTER} degrees=${DEGREES}) ..."
./montage-workflow.py --center "${CENTER}" --degrees "${DEGREES}" ${BANDS} 2>&1 | tee -a "${LOG_FILE}"

log_msg "Planning workflow (pegasus-plan) ..."
pegasus-plan --dir work --dax data/montage-workflow.yml \
  --output-site local --cluster horizontal 2>&1 | tee -a "${LOG_FILE}"

# pegasus-plan prints "pegasus-run  /path/to/work/XXX" - extract submit dir (path after pegasus-run)
SUBMIT_DIR=$(grep -E 'pegasus-run[[:space:]]+/' "${LOG_FILE}" 2>/dev/null | tail -1 | sed 's/.*pegasus-run[[:space:]]*\([^[:space:]]*\).*/\1/')
if [[ -n "${SUBMIT_DIR}" && -d "${SUBMIT_DIR}" ]]; then
  log_msg "Running: pegasus-run ${SUBMIT_DIR}"
  pegasus-run "${SUBMIT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
else
  echo "Submit dir not found in pegasus-plan output. Check ${LOG_FILE} for errors." >&2
  exit 1
fi

if [[ "${WAIT_FOR_COMPLETION}" == "1" && -n "${SUBMIT_DIR:-}" && -d "${SUBMIT_DIR}" ]]; then
  log_msg "Waiting for workflow completion (polling pegasus-status) ..."
  while true; do
    STATUS_OUT=$(pegasus-status -l "${SUBMIT_DIR}" 2>&1) || true
    echo "${STATUS_OUT}" | tee -a "${LOG_FILE}"
    if echo "${STATUS_OUT}" | grep -qE 'Summary:.*\(Success:'; then
      log_msg "Workflow completed successfully."
      break
    fi
    if echo "${STATUS_OUT}" | grep -qE 'Summary:.*\(Failure:'; then
      log_msg "Workflow failed."
      break
    fi
    sleep 15
  done
fi

# dfanalyzer (run only after --wait, when traces are complete)
DFANALYZER_OUTPUT="${RUN_LOG_DIR}/dfanalyzer_output.txt"
if [[ "${DFTRACER_ENABLE}" == "1" && "${DFANALYZER_ENABLED}" == "1" && "${WAIT_FOR_COMPLETION}" == "1" ]]; then
  if command -v dfanalyzer >/dev/null 2>&1; then
    CHECKPOINT_BOOL="false"
    if [[ "${DFANALYZER_CHECKPOINT_ENABLED}" == "1" ]]; then
      CHECKPOINT_BOOL="true"
      mkdir -p "${DFANALYZER_CHECKPOINT_DIR}"
    fi
    {
      echo "dfanalyzer trace_path=${TRACE_DIR}"
      echo "dfanalyzer preset=${DFANALYZER_PRESET}"
      echo "dfanalyzer checkpoint=${CHECKPOINT_BOOL}"
      echo "dfanalyzer checkpoint_dir=${DFANALYZER_CHECKPOINT_DIR}"
      dfanalyzer \
        analyzer=dftracer \
        analyzer/preset="${DFANALYZER_PRESET}" \
        trace_path="${TRACE_DIR}" \
        analyzer.checkpoint="${CHECKPOINT_BOOL}" \
        analyzer.checkpoint_dir="${DFANALYZER_CHECKPOINT_DIR}"
    } > "${DFANALYZER_OUTPUT}" 2>&1 || true
  else
    echo "dfanalyzer command not found in environment" > "${DFANALYZER_OUTPUT}"
  fi
elif [[ "${DFTRACER_ENABLE}" == "1" && "${DFANALYZER_ENABLED}" == "1" && "${WAIT_FOR_COMPLETION}" != "1" ]]; then
  echo "Skipped (use --wait to run dfanalyzer after workflow completes)" > "${DFANALYZER_OUTPUT}"
fi

cat > "${CONFIG_FILE}" <<EOF
{
  "run_id": "${RUN_ID}",
  "workload": "montage-pegasus",
  "mode": "${RUN_KIND}",
  "status": "completed",
  "center": "${CENTER}",
  "degrees": "${DEGREES}",
  "trace_dir": "${TRACE_DIR}",
  "trace_prefix": "${DFTRACER_LOG_PREFIX}",
  "dftracer_enabled": ${DFTRACER_ENABLE},
  "dftracer_inc_metadata": ${DFTRACER_INC_METADATA},
  "env_file": "${ENV_FILE}",
  "dfanalyzer_enabled": ${DFANALYZER_ENABLED},
  "dfanalyzer_output": "${DFANALYZER_OUTPUT}",
  "dfanalyzer_checkpoint_enabled": ${DFANALYZER_CHECKPOINT_ENABLED},
  "dfanalyzer_checkpoint_dir": "${DFANALYZER_CHECKPOINT_DIR}"
}
EOF

log_msg "Workflow submitted. Traces (if DFTracer enabled) under: ${TRACE_DIR}"
echo "Run recorded. Logs under: ${RUN_LOG_DIR}"
echo "Trace dir: ${TRACE_DIR}"
