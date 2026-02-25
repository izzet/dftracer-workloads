#!/usr/bin/env bash
set -euo pipefail

# install.sh
# Sets up the full OpenFOAM + DFTracer workload:
#   1. Adds/syncs the OpenFOAM-12 git submodule at openfoam/openfoam/.
#   2. Concretizes and installs the Spack environment.
#      Because spack.yaml has a develop: section pointing at the submodule,
#      Spack builds OpenFOAM FROM SOURCE (the submodule) rather than from a
#      tarball.  This gives full source edit + patch freedom.
#   3. Creates the Python virtual environment with dftracer and pyyaml.
#
# Incremental rebuild after source changes:
#   spack -e <workload-root> install --only package openfoam-org
# or directly from the source dir:
#   source setup_env.sh && cd openfoam && ./Allwmake -j$(nproc)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${ROOT_DIR}" rev-parse --show-toplevel)"
WORKLOAD_NAME="$(basename "${ROOT_DIR}")"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/logs"

SRC_DIR="${ROOT_DIR}/openfoam"
SUBMODULE_URL="https://github.com/OpenFOAM/OpenFOAM-12.git"
SUBMODULE_TAG="version-12"
SUBMODULE_REL_PATH="${WORKLOAD_NAME}/openfoam"  # relative from repo root

SPACK_LOG="${LOG_DIR}/install_spack.log"
GIT_LOG="${LOG_DIR}/install_git.log"
PY_LOG="${LOG_DIR}/install_python.log"

mkdir -p "${LOG_DIR}"

echo "==> Activating Spack..."
source ~/spack/share/spack/setup-env.sh

# ------------------------------------------------------------------
# 1. Set up git submodule (OpenFOAM-12 source)
# ------------------------------------------------------------------
echo "==> Setting up OpenFOAM-12 git submodule at ${SUBMODULE_REL_PATH}..."
{
  if [[ ! -e "${SRC_DIR}/.git" ]]; then
    git -C "${REPO_ROOT}" submodule add \
        "${SUBMODULE_URL}" "${SUBMODULE_REL_PATH}"
  fi
  git -C "${REPO_ROOT}" submodule sync -- "${SUBMODULE_REL_PATH}"
  git -C "${REPO_ROOT}" submodule update --init --recursive "${SUBMODULE_REL_PATH}"
  git -C "${SRC_DIR}" fetch --tags --force
  git -C "${SRC_DIR}" checkout "${SUBMODULE_TAG}"
  echo "Submodule pinned to tag: ${SUBMODULE_TAG}"
  echo "Commit: $(git -C "${SRC_DIR}" rev-parse HEAD)"
} 2>&1 | tee "${GIT_LOG}"

# ------------------------------------------------------------------
# 2. Concretize + install Spack environment
#    spack.yaml has develop: openfoam-org -> path: openfoam (the submodule)
#    so Spack will build OpenFOAM from our checked-out source tree.
# ------------------------------------------------------------------
echo "==> Concretizing Spack environment..."
echo "    (openfoam-org@12 will be built from the git submodule via spack develop)"
spack -e "${ROOT_DIR}" concretize -f 2>&1 | tee "${SPACK_LOG}"

echo "==> Installing Spack environment (builds OpenFOAM from source — takes ~1h first run)..."
spack -e "${ROOT_DIR}" install 2>&1 | tee -a "${SPACK_LOG}"

# ------------------------------------------------------------------
# 3. Python virtual environment
# ------------------------------------------------------------------
echo "==> Locating Spack-installed Python..."
SPACK_PYTHON_BIN="$(spack -e "${ROOT_DIR}" location -i python)/bin/python3"

if [[ -d "${VENV_DIR}" ]]; then
  VENV_PY="${VENV_DIR}/bin/python"
  if [[ ! -x "${VENV_PY}" ]]; then
    echo "==> Recreating venv (missing python executable)"
    rm -rf "${VENV_DIR}"
  else
    VENV_PY_VER="$("${VENV_PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")"
    SPACK_PY_VER="$("${SPACK_PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${VENV_PY_VER}" != "${SPACK_PY_VER}" ]]; then
      echo "==> Recreating venv (version mismatch: venv=${VENV_PY_VER}, spack=${SPACK_PY_VER})"
      rm -rf "${VENV_DIR}"
    fi
  fi
fi

echo "==> Creating Python virtual environment at ${VENV_DIR}..."
if [[ ! -d "${VENV_DIR}" ]]; then
  "${SPACK_PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "==> Installing Python requirements (dftracer, pyyaml)..."
python -m pip install --upgrade pip 2>&1 | tee "${PY_LOG}"
python -m pip install -r "${ROOT_DIR}/requirements.txt" 2>&1 | tee -a "${PY_LOG}"

echo ""
echo "Install complete."
echo "  Git log   : ${GIT_LOG}"
echo "  Spack log : ${SPACK_LOG}"
echo "  Python log: ${PY_LOG}"
echo "  Source    : ${SRC_DIR} (tag: ${SUBMODULE_TAG})"
echo ""
echo "Source is freely editable at: ${SRC_DIR}"
echo "Rebuild after changes: spack -e ${ROOT_DIR} install --only package openfoam-org"
echo ""
echo "Next step: run ./build.sh to prepare the pitzDaily benchmark case."
