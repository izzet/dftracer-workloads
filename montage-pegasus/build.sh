#!/usr/bin/env bash
set -euo pipefail
#
# Build Montage and pegasus-mpi-cluster.
# DFTracer patch can be applied later via dftracer/apply_montage_patches.py when needed.
#
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${ROOT_DIR}/logs/build.log"
MONTAGE_SRC="${ROOT_DIR}/montage"
MONTAGE_INSTALL="${ROOT_DIR}/install"
PEGASUS_SRC="${ROOT_DIR}/pegasus"

mkdir -p "${ROOT_DIR}/logs"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log_msg() { echo "[$(timestamp)] $*" | tee -a "${LOG_FILE}"; }

log_msg "Build log: ${LOG_FILE}"
log_msg "Sourcing setup_env.sh ..."
source "${ROOT_DIR}/setup_env.sh"
log_msg "Environment ready"

# ---- Montage ----
if [[ ! -d "${MONTAGE_SRC}" ]]; then
  echo "montage source missing. Run ./install.sh first." >&2
  exit 1
fi

log_msg "Building Montage ..."
# Apply compatibility patches from patches/ (sorted for deterministic order)
PATCHES_DIR="${ROOT_DIR}/patches"
if [[ -d "${PATCHES_DIR}" ]]; then
  for p in $(find "${PATCHES_DIR}" -maxdepth 1 -name '*.patch' -type f | sort); do
    log_msg "Applying patch: $(basename "${p}")"
    (cd "${MONTAGE_SRC}" && patch -p1 -N --forward < "${p}") >> "${LOG_FILE}" 2>&1 || true
  done
fi
(cd "${MONTAGE_SRC}" && make clean 2>/dev/null || true)
(cd "${MONTAGE_SRC}" && make -j"$(nproc)") >> "${LOG_FILE}" 2>&1

# Ensure mViewer is built (required by montage-workflow for PNG output)
# MontageLib/Viewer may be skipped on errors; build explicitly if missing
if [[ ! -x "${MONTAGE_SRC}/bin/mViewer" ]]; then
  log_msg "Building mViewer (MontageLib/Viewer) ..."
  (cd "${MONTAGE_SRC}/MontageLib/Viewer" && ./Configure.sh && make && make install) >> "${LOG_FILE}" 2>&1 || true
fi
if [[ ! -x "${MONTAGE_SRC}/bin/mViewer" ]]; then
  log_msg "Building mViewer (util/Viewer) ..."
  (cd "${MONTAGE_SRC}/util/Viewer" && ./Configure.sh && make && make install) >> "${LOG_FILE}" 2>&1 || true
fi

# Copy Montage bin to install
mkdir -p "${MONTAGE_INSTALL}/bin"
if [[ -d "${MONTAGE_SRC}/bin" ]]; then
  cp -f "${MONTAGE_SRC}"/bin/* "${MONTAGE_INSTALL}/bin/" 2>/dev/null || true
fi
log_msg "Montage built. Binaries in ${MONTAGE_INSTALL}/bin"

# ---- pegasus-mpi-cluster ----
if [[ -d "${PEGASUS_SRC}" ]]; then
  log_msg "Building pegasus-mpi-cluster ..."
  ANT_BIN="$(spack -e "${ROOT_DIR}" location -i ant 2>/dev/null)/bin/ant" || ANT_BIN="ant"
  if command -v "${ANT_BIN}" >/dev/null 2>&1; then
    (cd "${PEGASUS_SRC}" && "${ANT_BIN}" compile-pegasus-mpi-cluster) >> "${LOG_FILE}" 2>&1
    MPI_CLUSTER=""
    for p in "${PEGASUS_SRC}"/packages/pegasus-mpi-cluster/pegasus-mpi-cluster \
             "${PEGASUS_SRC}"/pegasus-mpi-cluster; do
      [[ -x "${p}" ]] && MPI_CLUSTER="${p}" && break
    done
    if [[ -n "${MPI_CLUSTER}" && -x "${MPI_CLUSTER}" ]]; then
      cp -f "${MPI_CLUSTER}" "${MONTAGE_INSTALL}/bin/"
      log_msg "pegasus-mpi-cluster installed to ${MONTAGE_INSTALL}/bin"
    else
      log_msg "WARN: pegasus-mpi-cluster binary not found after ant build"
    fi
  else
    log_msg "WARN: ant not found. Skipping pegasus-mpi-cluster. Install via Spack."
  fi
else
  log_msg "Pegasus source not found. Run ./install.sh for submodule."
fi

log_msg "Build complete. Log: ${LOG_FILE}"
