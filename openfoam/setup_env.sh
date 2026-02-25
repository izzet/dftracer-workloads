#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

source ~/spack/share/spack/setup-env.sh

if [[ ! -f "${ROOT_DIR}/spack.yaml" ]]; then
  echo "Missing ${ROOT_DIR}/spack.yaml." >&2
  return 1 2>/dev/null || exit 1
fi
spack env activate "${ROOT_DIR}"

# Source OpenFOAM's own bashrc so that FOAM_TUTORIALS, WM_PROJECT_DIR,
# and all OpenFOAM tools (blockMesh, foamRun, foamDictionary, etc.) are
# fully configured.  We locate the installation via the active Spack env.
if FOAM_INSTALL="$(spack -e "${ROOT_DIR}" location -i openfoam-org 2>/dev/null)"; then
  FOAM_ETC="${FOAM_INSTALL}/etc"
  if [[ -f "${FOAM_ETC}/bashrc" ]]; then
    # Suppress strict-mode errors that OpenFOAM's own scripts may trigger.
    set +euo pipefail
    # shellcheck disable=SC1091
    source "${FOAM_ETC}/bashrc"
    set -euo pipefail
  fi
  export FOAM_INSTALL FOAM_TUTORIALS="${FOAM_INSTALL}/tutorials"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run ./install.sh first." >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export WORKLOAD_ROOT="${ROOT_DIR}"
export WORKLOAD_LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${WORKLOAD_LOG_DIR}"
