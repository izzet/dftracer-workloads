#!/usr/bin/env bash
# build.sh — (re)build WRF from the local source submodule via spack develop.
#
# Because spack.yaml has a `develop:` section pointing at ./wrf, running
# `spack -e . install` compiles WRF from the local source tree rather than
# a downloaded tarball.  This means:
#   - First run:       full WRF compilation (~15-30 min)
#   - After edits:     incremental rebuild (only changed .F/.c files recompiled)
#
# Workflow:
#   ./install.sh   — one-time: submodule checkout + spack concretize + initial
#                    build + Python venv
#   ./build.sh     — after source edits: incremental rebuild + update build_info.env
#
# Typical use when adding knob hooks:
#   1. Edit wrf/share/module_io.F (or similar)
#   2. ./build.sh
#   3. ./run.sh  (or ./run.sh --dftracer-enable 1)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${ROOT_DIR}/logs/build.log"
BUILD_INFO="${ROOT_DIR}/logs/build_info.env"

mkdir -p "${ROOT_DIR}/logs"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_msg() {
  local msg="[$(timestamp)] $*"
  echo "${msg}"
  echo "${msg}" >> "${LOG_FILE}"
}

log_msg "Build log: ${LOG_FILE}"
log_msg "Activating environment via setup_env.sh ..."
source "${ROOT_DIR}/setup_env.sh"
log_msg "Environment active"

# ---------------------------------------------------------------------------
# Verify source submodule is present
# ---------------------------------------------------------------------------
SRC_DIR="${ROOT_DIR}/wrf"
if [[ ! -e "${SRC_DIR}/.git" ]]; then
  log_msg "ERROR: WRF source submodule not found at ${SRC_DIR}"
  log_msg "Run ./install.sh first."
  exit 1
fi
log_msg "WRF source: ${SRC_DIR} ($(git -C "${SRC_DIR}" describe --tags 2>/dev/null || echo 'unknown tag'))"

# ---------------------------------------------------------------------------
# Apply patches (idempotent: patch -N skips if already applied)
# ---------------------------------------------------------------------------
PATCHES_DIR="${ROOT_DIR}/patches"
if [[ -d "${PATCHES_DIR}" ]]; then
  for p in $(find "${PATCHES_DIR}" -maxdepth 1 -name '*.patch' -type f | sort); do
    log_msg "Applying patch: $(basename "${p}")"
    (cd "${SRC_DIR}" && patch -p1 -N --forward < "${p}") >> "${LOG_FILE}" 2>&1 || true
  done
fi

# ---------------------------------------------------------------------------
# Incremental rebuild via spack develop
# spack install is smart: it only recompiles if sources changed.
# ---------------------------------------------------------------------------
log_msg "spack install (incremental rebuild from ./wrf source) ..."
spack -e "${ROOT_DIR}" install 2>&1 | tee -a "${LOG_FILE}"
log_msg "spack install done"

# ---------------------------------------------------------------------------
# Locate installed executables and write build_info.env
# ---------------------------------------------------------------------------
log_msg "Locating WRF installation via spack ..."
WRF_PREFIX="$(spack -e "${ROOT_DIR}" location -i wrf)"
log_msg "WRF prefix: ${WRF_PREFIX}"

# Spack WRF may place executables under main/, run/, or bin/.
WRF_EXE=""
IDEAL_EXE=""

for candidate in \
    "${WRF_PREFIX}/main/wrf.exe" \
    "${WRF_PREFIX}/run/wrf.exe" \
    "${WRF_PREFIX}/bin/wrf.exe"; do
  if [[ -x "${candidate}" ]]; then
    WRF_EXE="${candidate}"
    break
  fi
done

for candidate in \
    "${WRF_PREFIX}/main/ideal.exe" \
    "${WRF_PREFIX}/run/ideal.exe" \
    "${WRF_PREFIX}/bin/ideal.exe"; do
  if [[ -x "${candidate}" ]]; then
    IDEAL_EXE="${candidate}"
    break
  fi
done

if [[ -z "${WRF_EXE}" ]]; then
  log_msg "ERROR: wrf.exe not found under ${WRF_PREFIX}"
  log_msg "Searched: main/, run/, bin/"
  exit 1
fi

if [[ -z "${IDEAL_EXE}" ]]; then
  log_msg "WARNING: ideal.exe not found under ${WRF_PREFIX}"
  log_msg "Check compile_type=em_b_wave is set in spack.yaml."
fi

# Locate run/ directory (contains LANDUSE.TBL, namelists, etc.)
WRF_RUN_DIR=""
for candidate in \
    "${WRF_PREFIX}/run" \
    "${WRF_PREFIX}/WRFV3/run"; do
  if [[ -d "${candidate}" && -f "${candidate}/LANDUSE.TBL" ]]; then
    WRF_RUN_DIR="${candidate}"
    break
  fi
done

# Locate test/em_b_wave directory
WRF_BWAVE_DIR=""
for candidate in \
    "${WRF_PREFIX}/test/em_b_wave" \
    "${WRF_PREFIX}/WRFV3/test/em_b_wave"; do
  if [[ -d "${candidate}" ]]; then
    WRF_BWAVE_DIR="${candidate}"
    break
  fi
done

cat > "${BUILD_INFO}" << EOF
WRF_PREFIX=${WRF_PREFIX}
WRF_EXE=${WRF_EXE}
IDEAL_EXE=${IDEAL_EXE:-}
WRF_RUN_DIR=${WRF_RUN_DIR:-}
WRF_BWAVE_DIR=${WRF_BWAVE_DIR:-}
EOF

log_msg "Build info written to ${BUILD_INFO}:"
log_msg "  WRF_PREFIX=${WRF_PREFIX}"
log_msg "  WRF_EXE=${WRF_EXE}"
log_msg "  IDEAL_EXE=${IDEAL_EXE:-<not found>}"
log_msg "  WRF_RUN_DIR=${WRF_RUN_DIR:-<not found>}"
log_msg "  WRF_BWAVE_DIR=${WRF_BWAVE_DIR:-<not found>}"
log_msg "Build complete.  Re-run after source edits for incremental rebuild."
log_msg "Run ./prepare_case.sh to refresh the case directory with the new binaries."
