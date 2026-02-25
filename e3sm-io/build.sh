#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${ROOT_DIR}/logs/build.log"
SRC_DIR="${ROOT_DIR}/e3sm-io"
BUILD_DIR="${ROOT_DIR}/build"
INSTALL_DIR="${ROOT_DIR}/install"

mkdir -p "${ROOT_DIR}/logs" "${BUILD_DIR}" "${INSTALL_DIR}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

log_msg "Build log: ${LOG_FILE}"
log_msg "Activating environment via setup_env.sh ..."
source "${ROOT_DIR}/setup_env.sh"
log_msg "Environment activation done"

if [[ ! -e "${SRC_DIR}/.git" ]]; then
  log_msg "Missing source directory: ${SRC_DIR}"
  log_msg "Add E3SM-IO submodule under e3sm-io/e3sm-io first."
  exit 1
fi

if [[ ! -f "${SRC_DIR}/configure.ac" && ! -f "${SRC_DIR}/Makefile" ]]; then
  log_msg "Expected E3SM-IO source in ${SRC_DIR}, but neither configure.ac nor Makefile was found."
  exit 1
fi

# Resolve dependency prefixes from active Spack environment.
log_msg "Resolving dependency prefixes from Spack ..."
PNETCDF_PREFIX="$(spack location -i parallel-netcdf)"
HDF5_PREFIX="$(spack location -i hdf5 2>/dev/null || true)"
NETCDF4_PREFIX="$(spack location -i netcdf-c 2>/dev/null || true)"
ADIOS2_PREFIX="$(spack location -i adios2 2>/dev/null || true)"
log_msg "Resolved dependency prefixes"

CONFIG_ARGS=(
  "--prefix=${INSTALL_DIR}"
  "--with-pnetcdf=${PNETCDF_PREFIX}"
  "CC=mpicc"
  "CXX=mpicxx"
)

if [[ -n "${HDF5_PREFIX}" ]]; then
  CONFIG_ARGS+=("--with-hdf5=${HDF5_PREFIX}")
fi
if [[ -n "${NETCDF4_PREFIX}" ]]; then
  CONFIG_ARGS+=("--with-netcdf4=${NETCDF4_PREFIX}")
fi
if [[ -n "${ADIOS2_PREFIX}" ]]; then
  CONFIG_ARGS+=("--with-adios2=${ADIOS2_PREFIX}")
fi

if [[ -f "${SRC_DIR}/configure.ac" ]]; then
  log_msg "Autotools source detected"
  log_msg "autoreconf ..."
  (
    cd "${SRC_DIR}"
    autoreconf -i
  ) >> "${LOG_FILE}" 2>&1
  log_msg "autoreconf done"

  log_msg "configure ..."
  (
    cd "${SRC_DIR}"
    ./configure "${CONFIG_ARGS[@]}"
  ) >> "${LOG_FILE}" 2>&1
  log_msg "configure done"

  log_msg "make ..."
  (
    cd "${SRC_DIR}"
    make -j"$(nproc)"
  ) >> "${LOG_FILE}" 2>&1
  log_msg "make done"

  log_msg "make install ..."
  (
    cd "${SRC_DIR}"
    make install
  ) >> "${LOG_FILE}" 2>&1
  log_msg "make install done"
else
  # The v.1.2.0 tree uses a hand-written Makefile with no install target.
  # Work around upstream token usage in this tag where _FillValue is not a macro.
  E3SM_CFLAGS='-O2 -fcommon -D_FillValue=\"_FillValue\"'
  log_msg "Makefile-only source detected"
  log_msg "make clean ..."
  (
    cd "${SRC_DIR}"
    make clean
  ) >> "${LOG_FILE}" 2>&1 || true
  log_msg "make clean done"

  log_msg "make ..."
  (
    cd "${SRC_DIR}"
    make -j"$(nproc)" PnetCDF_DIR="${PNETCDF_PREFIX}" CFLAGS="${E3SM_CFLAGS}"
  ) >> "${LOG_FILE}" 2>&1
  log_msg "make done"

  log_msg "staging binaries into ${INSTALL_DIR}/bin ..."
  mkdir -p "${INSTALL_DIR}/bin"
  [[ -x "${SRC_DIR}/e3sm_io" ]] && cp -f "${SRC_DIR}/e3sm_io" "${INSTALL_DIR}/bin/"
  [[ -x "${SRC_DIR}/dat2nc" ]] && cp -f "${SRC_DIR}/dat2nc" "${INSTALL_DIR}/bin/"
  [[ -x "${SRC_DIR}/e3sm_io.romio_patch" ]] && cp -f "${SRC_DIR}/e3sm_io.romio_patch" "${INSTALL_DIR}/bin/"
  log_msg "staging binaries done"
fi

log_msg "Build complete. Log: ${LOG_FILE}"
