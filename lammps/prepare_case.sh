#!/usr/bin/env bash
# prepare_case.sh — set up a run directory for the LAMMPS LJ melt test case.
#
# The Lennard-Jones melt is a minimal, self-contained case:
#   - No external data files needed
#   - Creates atoms in a box, runs MD
#   - Exercises dump (trajectory) and restart I/O — good for DFTracer
#   - Runs in seconds (good for iteration)
#
# Usage:
#   ./prepare_case.sh [--case-dir PATH]
#
# Output: a self-contained run directory at run/lj_melt/ (default).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_INFO="${ROOT_DIR}/logs/build_info.env"
CASE_NAME="lj_melt"
CASE_DIR="${ROOT_DIR}/run/${CASE_NAME}"
LOG_FILE="${ROOT_DIR}/logs/prepare_case.log"
NP="${NP:-4}"
MPIEXEC="${MPIEXEC:-mpiexec}"

# Knob defaults (match dftracer/knobs.yaml)
DUMP_INTERVAL="${DUMP_INTERVAL:-100}"
RESTART_INTERVAL="${RESTART_INTERVAL:-500}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-dir) CASE_DIR="$2"; shift 2 ;;
    --np) NP="$2"; shift 2 ;;
    --mpiexec) MPIEXEC="$2"; shift 2 ;;
    --dump-interval) DUMP_INTERVAL="$2"; shift 2 ;;
    --restart-interval) RESTART_INTERVAL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${ROOT_DIR}/logs"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_msg() {
  local msg="[$(timestamp)] $*"
  echo "${msg}"
  echo "${msg}" >> "${LOG_FILE}"
}

log_msg "Activating environment ..."
source "${ROOT_DIR}/setup_env.sh"
log_msg "Environment active"

# ---------------------------------------------------------------------------
# Load build info written by build.sh
# ---------------------------------------------------------------------------
if [[ ! -f "${BUILD_INFO}" ]]; then
  log_msg "ERROR: ${BUILD_INFO} not found. Run ./build.sh first."
  exit 1
fi
# shellcheck disable=SC1090
source "${BUILD_INFO}"

if [[ -z "${LMP_EXE:-}" ]]; then
  log_msg "ERROR: LMP_EXE not set in ${BUILD_INFO}. Run ./build.sh to verify install."
  exit 1
fi

log_msg "LAMMPS prefix:  ${LAMMPS_PREFIX}"
log_msg "lmp:            ${LMP_EXE}"
log_msg "Case dir:       ${CASE_DIR}"
log_msg "dump_interval:  ${DUMP_INTERVAL}"
log_msg "restart_interval: ${RESTART_INTERVAL}"

# ---------------------------------------------------------------------------
# Create the run directory
# ---------------------------------------------------------------------------
log_msg "Creating case directory: ${CASE_DIR}"
mkdir -p "${CASE_DIR}"

# ---------------------------------------------------------------------------
# Link executable into the case directory
# ---------------------------------------------------------------------------
ln -sf "${LMP_EXE}" "${CASE_DIR}/lmp"
log_msg "Linked lmp -> ${LMP_EXE}"

# ---------------------------------------------------------------------------
# Write LAMMPS input script (LJ melt — minimal, self-contained)
# Knob values (dump_interval, restart_interval) are injected.
# ---------------------------------------------------------------------------
INPUT_FILE="${CASE_DIR}/in.lj_melt"
cat > "${INPUT_FILE}" << EOF
# LAMMPS Lennard-Jones melt — DFTracer workload
# Minimal case: no external data, runs in seconds.
# Knobs: dump_interval, restart_interval (see dftracer/knobs.yaml)

units           lj
atom_style      atomic

lattice         fcc 0.8442
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box

mass            1 1.0

velocity        all create 1.44 87287 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

neighbor        0.3 bin
neigh_modify    delay 0 every 20 check no

fix             1 all nve
fix             2 all langevin 1.0 1.0 10.0 904297

# DFTracer knob: dump_interval (default 100)
thermo          ${DUMP_INTERVAL}
thermo_style    custom step temp epair etotal press

# DFTracer knob: dump interval
dump            1 all atom ${DUMP_INTERVAL} dump.melt.lammpstrj

# DFTracer knob: restart interval
restart         ${RESTART_INTERVAL} restart.lj_melt.restart

run             2000
EOF
log_msg "Wrote ${INPUT_FILE}"

# ---------------------------------------------------------------------------
# Write case metadata
# ---------------------------------------------------------------------------
cat > "${CASE_DIR}/case_info.env" << EOF
CASE_NAME=${CASE_NAME}
CASE_DIR=${CASE_DIR}
LMP_EXE=${LMP_EXE}
LAMMPS_PREFIX=${LAMMPS_PREFIX}
NP_DEFAULT=${NP}
MPIEXEC_DEFAULT=${MPIEXEC}
DUMP_INTERVAL=${DUMP_INTERVAL}
RESTART_INTERVAL=${RESTART_INTERVAL}
INPUT_FILE=in.lj_melt
EOF

log_msg "Case directory ready: ${CASE_DIR}"
log_msg "Contents:"
ls "${CASE_DIR}" | while read -r f; do log_msg "  ${f}"; done
log_msg ""
log_msg "Next steps:"
log_msg "  ./run.sh                   # baseline LAMMPS run (no tracing)"
log_msg "  ./run.sh --dftracer-enable 1   # traced run with DFTracer"
