#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

ensure_module_command() {
  if type module >/dev/null 2>&1; then
    return 0
  fi

  for init_script in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f "${init_script}" ]]; then
      # shellcheck disable=SC1090
      source "${init_script}"
      break
    fi
  done
}

if [[ -n "${STORMER_MODULES:-}" ]]; then
  ensure_module_command
  if type module >/dev/null 2>&1; then
    for mod in ${STORMER_MODULES}; do
      module load "${mod}"
    done
  else
    echo "Requested STORMER_MODULES but no module command is available." >&2
    return 1 2>/dev/null || exit 1
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run ./install.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export WORKLOAD_ROOT="${ROOT_DIR}"
export WORKLOAD_LOG_DIR="${ROOT_DIR}/logs"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export NETCDF4_PYTHON_DISABLE_HDF5_FILE_LOCKING="${NETCDF4_PYTHON_DISABLE_HDF5_FILE_LOCKING:-1}"

mkdir -p "${WORKLOAD_LOG_DIR}"
