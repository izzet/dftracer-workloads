#!/usr/bin/env bash
# install.sh — one-time environment setup for the WRF workload.
# Idempotent: safe to re-run.  Performs:
#   1. git submodule add/sync/update for WRF source (reference copy, v4.6.1)
#   2. spack -e . concretize + install (this also COMPILES WRF — takes 15-30 min)
#   3. Python .venv creation using Spack-provided Python + dftracer requirements
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${ROOT_DIR}" rev-parse --show-toplevel)"
WORKLOAD_NAME="$(basename "${ROOT_DIR}")"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/logs"
SRC_DIR="${ROOT_DIR}/wrf"
SUBMODULE_URL="https://github.com/wrf-model/WRF.git"
SUBMODULE_TAG="v4.6.1"
SUBMODULE_REL_PATH="${WORKLOAD_NAME}/wrf"
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
log_msg "  git:    ${GIT_LOG}"
log_msg "  spack:  ${SPACK_LOG}"
log_msg "  python: ${PY_LOG}"

if [[ ! -f "${ROOT_DIR}/spack.yaml" ]]; then
  echo "spack.yaml not found at ${ROOT_DIR}/spack.yaml" >&2
  exit 1
fi

source ~/spack/share/spack/setup-env.sh

# ---------------------------------------------------------------------------
# 1. Git submodule (source reference + namelists/datasets)
# Reset to clean state so patches can be applied idempotently (build.sh also applies).
# ---------------------------------------------------------------------------
if [[ ! -e "${SRC_DIR}/.git" ]]; then
  log_msg "git submodule add ${SUBMODULE_REL_PATH} ..."
  git -C "${REPO_ROOT}" submodule add "${SUBMODULE_URL}" "${SUBMODULE_REL_PATH}" >> "${GIT_LOG}" 2>&1
  log_msg "git submodule add done"
fi
log_msg "Resetting WRF submodule to clean state (patches reapplied below) ..."
git -C "${SRC_DIR}" reset --hard HEAD 2>/dev/null || true
git -C "${SRC_DIR}" clean -fd 2>/dev/null || true
log_msg "git submodule sync ${SUBMODULE_REL_PATH} ..."
git -C "${REPO_ROOT}" submodule sync -- "${SUBMODULE_REL_PATH}" >> "${GIT_LOG}" 2>&1
log_msg "git submodule sync done"
log_msg "git submodule update --init --recursive --force ${SUBMODULE_REL_PATH} ..."
git -C "${REPO_ROOT}" submodule update --init --recursive --force "${SUBMODULE_REL_PATH}" >> "${GIT_LOG}" 2>&1
log_msg "git submodule update done"
log_msg "git fetch tags in ${SUBMODULE_REL_PATH} ..."
git -C "${SRC_DIR}" fetch --tags --force >> "${GIT_LOG}" 2>&1
log_msg "git fetch done"
log_msg "git checkout ${SUBMODULE_TAG} ..."
git -C "${SRC_DIR}" checkout -f "${SUBMODULE_TAG}" >> "${GIT_LOG}" 2>&1
log_msg "git checkout done"

# ---------------------------------------------------------------------------
# Apply patches from patches/ (enables ideal.exe to use namelist paths for full paths)
# ---------------------------------------------------------------------------
PATCHES_DIR="${ROOT_DIR}/patches"
if [[ -d "${PATCHES_DIR}" ]]; then
  for p in $(find "${PATCHES_DIR}" -maxdepth 1 -name '*.patch' -type f | sort); do
    log_msg "Applying patch: $(basename "${p}")"
    (cd "${SRC_DIR}" && patch -p1 -N --forward < "${p}") >> "${GIT_LOG}" 2>&1 || true
  done
fi

# ---------------------------------------------------------------------------
# 2. Spack environment: concretize + install WRF + all deps
#    spack.yaml has a `develop:` block pointing at ./wrf (the submodule).
#    Spack will compile WRF FROM the local source, not a downloaded tarball.
#    This allows source edits + incremental rebuilds via ./build.sh.
#    NOTE: first run compiles WRF from scratch (~15-30 min).
# ---------------------------------------------------------------------------
log_msg "spack concretize ..."
spack -e "${ROOT_DIR}" concretize -f >> "${SPACK_LOG}" 2>&1
log_msg "spack concretize done"
log_msg "spack install — builds WRF from ./wrf source (first run: ~15-30 min) ..."
spack -e "${ROOT_DIR}" install >> "${SPACK_LOG}" 2>&1
log_msg "spack install done"
log_msg "Note: for incremental rebuilds after source edits, use ./build.sh"

# ---------------------------------------------------------------------------
# 3. Python venv using Spack-installed Python
# ---------------------------------------------------------------------------
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

log_msg "Install complete."
log_msg "  git log:    ${GIT_LOG}"
log_msg "  spack log:  ${SPACK_LOG}"
log_msg "  python log: ${PY_LOG}"
log_msg "Source pinned at tag ${SUBMODULE_TAG} in ${SUBMODULE_REL_PATH}"
log_msg "Next: ./build.sh  →  ./prepare_case.sh  →  ./run.sh"
