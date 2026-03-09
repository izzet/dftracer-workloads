#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/logs"
PY_LOG="${LOG_DIR}/install_python.log"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WITH_DATA_TOOLS="${INSTALL_DATA_TOOLS:-0}"
PIP_INSTALL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --with-data-tools)
      WITH_DATA_TOOLS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_DIR}"

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
    exit 1
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
if [[ -n "${STORMER_PIP_INDEX_URL:-}" ]]; then
  PIP_INSTALL_ARGS+=(--index-url "${STORMER_PIP_INDEX_URL}")
fi
if [[ -n "${STORMER_PIP_EXTRA_INDEX_URL:-}" ]]; then
  PIP_INSTALL_ARGS+=(--extra-index-url "${STORMER_PIP_EXTRA_INDEX_URL}")
fi
python -m pip install --upgrade pip >> "${PY_LOG}" 2>&1
python -m pip install "${PIP_INSTALL_ARGS[@]}" -r "${ROOT_DIR}/requirements.txt" >> "${PY_LOG}" 2>&1

if [[ "${WITH_DATA_TOOLS}" == "1" ]]; then
  python -m pip install "${PIP_INSTALL_ARGS[@]}" -r "${ROOT_DIR}/requirements-data.txt" >> "${PY_LOG}" 2>&1
fi

echo "Install complete. Log: ${PY_LOG}"
