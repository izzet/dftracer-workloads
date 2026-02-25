#!/usr/bin/env bash
# build.sh — (re)build LAMMPS from the local source submodule via spack develop.
#
# Because spack.yaml has a `develop:` section pointing at ./lammps, running
# `spack -e . install` compiles LAMMPS from the local source tree rather than
# a downloaded tarball.  This means:
#   - First run:       full LAMMPS compilation (~10-20 min)
#   - After edits:     incremental rebuild (only changed files recompiled)
#
# Workflow:
#   ./install.sh   — one-time: submodule checkout + spack concretize + initial
#                    build + Python venv
#   ./build.sh     — after source edits: incremental rebuild + update build_info.env
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${ROOT_DIR}/logs/build.log"
BUILD_INFO="${ROOT_DIR}/logs/build_info.env"
SRC_DIR="${ROOT_DIR}/lammps"

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
if [[ ! -e "${SRC_DIR}/.git" ]]; then
  log_msg "ERROR: LAMMPS source submodule not found at ${SRC_DIR}"
  log_msg "Run ./install.sh first."
  exit 1
fi
log_msg "LAMMPS source: ${SRC_DIR} ($(git -C "${SRC_DIR}" describe --tags 2>/dev/null || echo 'unknown tag'))"

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
# ---------------------------------------------------------------------------
log_msg "spack install (incremental rebuild from ./lammps source) ..."
spack -e "${ROOT_DIR}" install 2>&1 | tee -a "${LOG_FILE}"
log_msg "spack install done"

# ---------------------------------------------------------------------------
# Locate installed executable and write build_info.env
# ---------------------------------------------------------------------------
log_msg "Locating LAMMPS installation via spack ..."
LAMMPS_PREFIX="$(spack -e "${ROOT_DIR}" location -i lammps)"
log_msg "LAMMPS prefix: ${LAMMPS_PREFIX}"

# Spack LAMMPS installs the executable as lmp in bin/
LMP_EXE=""
for candidate in \
    "${LAMMPS_PREFIX}/bin/lmp" \
    "${LAMMPS_PREFIX}/bin/lammps"; do
  if [[ -x "${candidate}" ]]; then
    LMP_EXE="${candidate}"
    break
  fi
done

if [[ -z "${LMP_EXE}" ]]; then
  log_msg "ERROR: lmp executable not found under ${LAMMPS_PREFIX}"
  log_msg "Searched: bin/lmp, bin/lammps"
  exit 1
fi

cat > "${BUILD_INFO}" << EOF
LAMMPS_PREFIX=${LAMMPS_PREFIX}
LMP_EXE=${LMP_EXE}
EOF

log_msg "Build info written to ${BUILD_INFO}:"
log_msg "  LAMMPS_PREFIX=${LAMMPS_PREFIX}"
log_msg "  LMP_EXE=${LMP_EXE}"
log_msg "Build complete.  Re-run after source edits for incremental rebuild."
log_msg "Run ./prepare_case.sh to refresh the case directory with the new binary."
