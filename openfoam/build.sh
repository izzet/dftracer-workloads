#!/usr/bin/env bash
set -euo pipefail

# build.sh — Two-stage build for the OpenFOAM workload.
#
# Stage A: Compile OpenFOAM from the spack develop source tree (openfoam/).
#   Spack's install phase sets up all configuration files (etc/bashrc,
#   wmake settings, etc.) but leaves the actual C++ compilation to us.
#   We invoke the spack-Allwmake wrapper that Spack created in the source
#   tree; it sources the local etc/bashrc and then calls ./Allwmake.
#   ~30-60 minutes on first run; subsequent runs are incremental.
#
# Stage B: Prepare the pitzDaily benchmark case.
#   Copies the tutorial, runs blockMesh to generate the mesh.
#   The result is the base case template used by run.sh.
#
# Incremental re-compile after source edits (skip Stage B):
#   ./build.sh --of-only
#
# Re-run Stage B without recompiling (skip Stage A):
#   ./build.sh --case-only

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
OUT_DIR="${ROOT_DIR}/output"
CASE_DIR="${OUT_DIR}/cases/pitzDaily"
BUILD_LOG="${LOG_DIR}/build.log"

# openfoam/ is a symlink -> OpenFOAM-12/ (Spack renames the develop dir)
SRC_DIR="${ROOT_DIR}/openfoam"

STAGE_OF=1
STAGE_CASE=1

for arg in "$@"; do
  case "${arg}" in
    --of-only)   STAGE_CASE=0 ;;
    --case-only) STAGE_OF=0   ;;
  esac
done

mkdir -p "${LOG_DIR}" "${OUT_DIR}/cases"

source "${ROOT_DIR}/setup_env.sh"

{
  echo "=== OpenFOAM build.sh: $(date) ==="
  echo "SRC_DIR=${SRC_DIR}"

  # ----------------------------------------------------------------
  # Stage A: Compile OpenFOAM via spack-Allwmake
  # ----------------------------------------------------------------
  if [[ "${STAGE_OF}" == "1" ]]; then
    echo ""
    echo "=== Stage A: Compiling OpenFOAM from source ==="

    if [[ ! -f "${SRC_DIR}/spack-Allwmake" ]]; then
      echo "ERROR: ${SRC_DIR}/spack-Allwmake not found." >&2
      echo "       Run ./install.sh first." >&2
      exit 1
    fi

    # Check if already compiled (any blockMesh binary in platform dirs)
    if ls "${SRC_DIR}"/platforms/*/bin/blockMesh 2>/dev/null | grep -q blockMesh; then
      echo "OpenFOAM binaries already present — running incremental build."
    else
      echo "No compiled binaries found — running full Allwmake (30-60 min)..."
    fi

    NPROC=$(nproc 2>/dev/null || echo 4)
    echo "Using ${NPROC} parallel jobs."

    (
      cd "${SRC_DIR}"
      bash spack-Allwmake -j"${NPROC}"
    )

    # Verify key binaries were produced
    if ! ls "${SRC_DIR}"/platforms/*/bin/blockMesh 2>/dev/null | grep -q blockMesh; then
      echo "ERROR: blockMesh binary not found after Allwmake." >&2
      echo "       Check the build output above for errors." >&2
      exit 1
    fi

    PLATFORM_BIN="$(ls "${SRC_DIR}"/platforms/*/bin/blockMesh 2>/dev/null | head -1 | xargs dirname)"
    echo "OpenFOAM compiled successfully."
    echo "Platform bin: ${PLATFORM_BIN}"
    echo "Binaries: $(ls "${PLATFORM_BIN}" | wc -l) executables"
  fi

  # ----------------------------------------------------------------
  # Stage B: Prepare pitzDaily benchmark case
  # ----------------------------------------------------------------
  if [[ "${STAGE_CASE}" == "1" ]]; then
    echo ""
    echo "=== Stage B: Preparing pitzDaily benchmark case ==="

    if [[ -z "${FOAM_TUTORIALS:-}" ]]; then
      echo "ERROR: FOAM_TUTORIALS not set after sourcing OpenFOAM environment." >&2
      exit 1
    fi
    echo "FOAM_TUTORIALS=${FOAM_TUTORIALS}"

    # Search for pitzDaily tutorial (OF 12 layout first, older fallback)
    TUTORIAL_CANDIDATES=(
      "${FOAM_TUTORIALS}/incompressibleFluid/pitzDaily"
      "${FOAM_TUTORIALS}/incompressible/simpleFoam/pitzDaily"
      "${FOAM_TUTORIALS}/incompressible/pitzDaily"
    )
    TUTORIAL_SRC=""
    for candidate in "${TUTORIAL_CANDIDATES[@]}"; do
      if [[ -d "${candidate}" ]]; then
        TUTORIAL_SRC="${candidate}"
        break
      fi
    done
    if [[ -z "${TUTORIAL_SRC}" ]]; then
      echo "ERROR: Could not find pitzDaily tutorial under ${FOAM_TUTORIALS}." >&2
      for c in "${TUTORIAL_CANDIDATES[@]}"; do echo "  tried: ${c}" >&2; done
      exit 1
    fi
    echo "Tutorial source: ${TUTORIAL_SRC}"

    if [[ -d "${CASE_DIR}" ]]; then
      echo "Removing existing case at ${CASE_DIR} for a clean setup..."
      rm -rf "${CASE_DIR}"
    fi
    cp -r "${TUTORIAL_SRC}" "${CASE_DIR}"
    echo "Case copied to: ${CASE_DIR}"

    echo "Running blockMesh..."
    (cd "${CASE_DIR}" && blockMesh)
    echo "blockMesh complete."

    echo ""
    echo "Base case ready at: ${CASE_DIR}"
    echo "Next step: ./run.sh"
  fi

} 2>&1 | tee "${BUILD_LOG}"

echo "Build log: ${BUILD_LOG}"
