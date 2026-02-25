#!/usr/bin/env bash
set -uo pipefail
#
# Clean Montage Pegasus: stop workflows, stop Condor, remove working dirs/data.
# Idempotent - safe to run multiple times.
#
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="${ROOT_DIR}/montage-workflow-v3"
CONDOR_DIR="${ROOT_DIR}/condor"
INSTALL_DIR="${ROOT_DIR}/install"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Condor/Pegasus in PATH
if [[ -d "${CONDOR_DIR}" && -f "${CONDOR_DIR}/etc/condor_config" ]]; then
  export CONDOR_CONFIG="${CONDOR_DIR}/etc/condor_config"
  export PATH="${CONDOR_DIR}/bin:${CONDOR_DIR}/sbin:${PATH}"
elif [[ -d "${INSTALL_DIR}" && -f "${INSTALL_DIR}/etc/condor_config" ]]; then
  export CONDOR_CONFIG="${INSTALL_DIR}/etc/condor_config"
  export PATH="${INSTALL_DIR}/bin:${INSTALL_DIR}/sbin:${PATH}"
fi

log "Cleaning Montage Pegasus..."

# 1. Remove Pegasus workflows
if [[ -d "${WORKFLOW_DIR}/work" ]]; then
  while IFS= read -r -d '' d; do
    pegasus-remove "$d" 2>/dev/null && log "Removed workflow $d" || true
  done < <(find "${WORKFLOW_DIR}/work" -maxdepth 5 -type d -name "run*" -print0 2>/dev/null)
fi

# 2. Remove all Condor jobs
condor_rm -all 2>/dev/null && log "Removed Condor jobs" || true

# 3. Stop Condor daemon
if command -v condor_off &>/dev/null; then
  condor_off -master -fast 2>/dev/null && log "Stopped Condor" || \
  condor_off -master 2>/dev/null && log "Stopped Condor" || true
fi

# 4. Kill any lingering Condor processes
pkill -f "condor_master|condor_schedd|condor_startd|condor_dagman" 2>/dev/null && log "Killed Condor processes" || true
sleep 1

# 5. Clean Condor spool and execute dirs
for base in "${CONDOR_DIR}" "${INSTALL_DIR}"; do
  if [[ -d "${base}/local/spool" ]]; then
    rm -rf "${base}"/local/spool/* 2>/dev/null && log "Cleaned ${base}/local/spool" || true
  fi
  if [[ -d "${base}/local/execute" ]]; then
    rm -rf "${base}"/local/execute/dir_* 2>/dev/null && log "Cleaned ${base}/local/execute" || true
  fi
done

# 6. Remove workflow working directories and data
rm -rf "${WORKFLOW_DIR}/work" 2>/dev/null && log "Removed ${WORKFLOW_DIR}/work" || true
rm -rf "${WORKFLOW_DIR}/data" 2>/dev/null && log "Removed ${WORKFLOW_DIR}/data" || true

# 7. Remove generated sites.yml if present
rm -f "${WORKFLOW_DIR}/sites.yml" 2>/dev/null && log "Removed sites.yml" || true

log "Clean complete."
