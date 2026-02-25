#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${ROOT_DIR}" rev-parse --show-toplevel)"
WORKLOAD_NAME="$(basename "${ROOT_DIR}")"
SPACK_YAML="${ROOT_DIR}/spack.yaml"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/logs"
SRC_DIR="${ROOT_DIR}/e3sm-io"
SUBMODULE_URL="https://github.com/Parallel-NetCDF/E3SM-IO.git"
SUBMODULE_TAG="v.1.2.0"
SUBMODULE_REL_PATH="${WORKLOAD_NAME}/e3sm-io"
SPACK_LOG="${LOG_DIR}/install_spack.log"
GIT_LOG="${LOG_DIR}/install_git.log"
PY_LOG="${LOG_DIR}/install_python.log"

mkdir -p "${LOG_DIR}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

log_msg "Logs:"
log_msg "  git:   ${GIT_LOG}"
log_msg "  spack: ${SPACK_LOG}"
log_msg "  python:${PY_LOG}"

if [[ ! -f "${SPACK_YAML}" ]]; then
  echo "spack.yaml not found at ${SPACK_YAML}" >&2
  exit 1
fi

source ~/spack/share/spack/setup-env.sh

# Ensure workload source submodule exists and is pinned to a stable tag.
if [[ ! -e "${SRC_DIR}/.git" ]]; then
  log_msg "git submodule add ${SUBMODULE_REL_PATH} ..."
  git -C "${REPO_ROOT}" submodule add "${SUBMODULE_URL}" "${SUBMODULE_REL_PATH}" >> "${GIT_LOG}" 2>&1
  log_msg "git submodule add ${SUBMODULE_REL_PATH} done"
fi
log_msg "git submodule sync ${SUBMODULE_REL_PATH} ..."
git -C "${REPO_ROOT}" submodule sync -- "${SUBMODULE_REL_PATH}" >> "${GIT_LOG}" 2>&1
log_msg "git submodule sync ${SUBMODULE_REL_PATH} done"
log_msg "git submodule update ${SUBMODULE_REL_PATH} ..."
git -C "${REPO_ROOT}" submodule update --init --recursive "${SUBMODULE_REL_PATH}" >> "${GIT_LOG}" 2>&1
log_msg "git submodule update ${SUBMODULE_REL_PATH} done"
log_msg "git fetch tags in ${SUBMODULE_REL_PATH} ..."
git -C "${SRC_DIR}" fetch --tags --force >> "${GIT_LOG}" 2>&1
log_msg "git fetch tags in ${SUBMODULE_REL_PATH} done"
log_msg "git checkout ${SUBMODULE_TAG} ..."
git -C "${SRC_DIR}" checkout "${SUBMODULE_TAG}" >> "${GIT_LOG}" 2>&1
log_msg "git checkout ${SUBMODULE_TAG} done"

log_msg "spack concretize ..."
spack -e "${ROOT_DIR}" concretize -f >> "${SPACK_LOG}" 2>&1
log_msg "spack concretize done"
log_msg "spack install ..."
spack -e "${ROOT_DIR}" install >> "${SPACK_LOG}" 2>&1
log_msg "spack install done"

SPACK_PYTHON_BIN="$(spack -e "${ROOT_DIR}" location -i python)/bin/python3"
log_msg "Using Spack Python: ${SPACK_PYTHON_BIN}"

if [[ -d "${VENV_DIR}" ]]; then
  VENV_PYTHON_BIN="${VENV_DIR}/bin/python"
  if [[ ! -x "${VENV_PYTHON_BIN}" ]]; then
    log_msg "Recreating venv (missing python executable)"
    rm -rf "${VENV_DIR}"
  else
    VENV_PY_VER="$("${VENV_PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    SPACK_PY_VER="$("${SPACK_PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${VENV_PY_VER}" != "${SPACK_PY_VER}" ]]; then
      log_msg "Recreating venv (version mismatch: venv=${VENV_PY_VER}, spack=${SPACK_PY_VER})"
      rm -rf "${VENV_DIR}"
    fi
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  log_msg "Creating virtual environment at ${VENV_DIR}"
  "${SPACK_PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log_msg "pip upgrade ..."
python -m pip install --upgrade pip >> "${PY_LOG}" 2>&1
log_msg "pip upgrade done"
log_msg "pip install requirements.txt ..."
python -m pip install -r "${ROOT_DIR}/requirements.txt" >> "${PY_LOG}" 2>&1
log_msg "pip install requirements.txt done"

echo "Install complete. Logs:"
echo "  ${GIT_LOG}"
echo "  ${SPACK_LOG}"
echo "  ${PY_LOG}"
echo "Source pinned at tag ${SUBMODULE_TAG} in ${SUBMODULE_REL_PATH}"
