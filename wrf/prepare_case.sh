#!/usr/bin/env bash
# prepare_case.sh — set up a run directory for the WRF em_b_wave idealized test case.
#
# The em_b_wave (baroclinic wave) case is fully idealized:
#   - No real input data or WPS preprocessing needed
#   - ideal.exe generates initial conditions from scratch
#   - Runs in minutes (good for DFTracer iteration)
#   - Still exercises WRF's full I/O stack (history + restart files)
#
# Usage:
#   ./prepare_case.sh [--case-dir PATH] [--np N]
#
# Output: a self-contained run directory at run/em_b_wave/ (default).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_INFO="${ROOT_DIR}/logs/build_info.env"
CASE_NAME="em_b_wave"
CASE_DIR="${ROOT_DIR}/run/${CASE_NAME}"
LOG_FILE="${ROOT_DIR}/logs/prepare_case.log"
NP="${NP:-4}"
MPIEXEC="${MPIEXEC:-mpiexec}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-dir) CASE_DIR="$2"; shift 2 ;;
    --np) NP="$2"; shift 2 ;;
    --mpiexec) MPIEXEC="$2"; shift 2 ;;
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

if [[ -z "${WRF_EXE:-}" ]]; then
  log_msg "ERROR: WRF_EXE not set in ${BUILD_INFO}. Run ./build.sh to verify install."
  exit 1
fi

log_msg "WRF prefix:  ${WRF_PREFIX}"
log_msg "wrf.exe:     ${WRF_EXE}"
log_msg "ideal.exe:   ${IDEAL_EXE:-<not found>}"
log_msg "Case dir:    ${CASE_DIR}"

# ---------------------------------------------------------------------------
# Create the run directory
# ---------------------------------------------------------------------------
log_msg "Creating case directory: ${CASE_DIR}"
mkdir -p "${CASE_DIR}"

# ---------------------------------------------------------------------------
# Populate run directory from WRF installation
# ---------------------------------------------------------------------------

# Copy run-time support files (TBL, dat, txt, etc.) from WRF run/ directory.
if [[ -n "${WRF_RUN_DIR:-}" && -d "${WRF_RUN_DIR}" ]]; then
  log_msg "Copying support files from ${WRF_RUN_DIR} ..."
  # Copy all non-executable support files (skip .exe and large binary outputs)
  find "${WRF_RUN_DIR}" -maxdepth 1 -type f \
    ! -name "*.exe" ! -name "wrfinput*" ! -name "wrfout*" ! -name "wrfrst*" \
    -exec cp -n {} "${CASE_DIR}/" \; 2>/dev/null || true
  log_msg "Support files copied"
else
  log_msg "WARNING: WRF_RUN_DIR not available. Attempting to find support files..."
  # Try to locate them from the submodule source if available
  if [[ -d "${ROOT_DIR}/wrf/run" ]]; then
    find "${ROOT_DIR}/wrf/run" -maxdepth 1 -type f \
      ! -name "*.exe" ! -name "wrfinput*" ! -name "wrfout*" ! -name "wrfrst*" \
      -exec cp -n {} "${CASE_DIR}/" \; 2>/dev/null || true
    log_msg "Support files copied from submodule run/"
  fi
fi

# Link (or copy) executables into the case directory.
for exe_var in WRF_EXE IDEAL_EXE; do
  exe_path="${!exe_var:-}"
  if [[ -n "${exe_path}" && -x "${exe_path}" ]]; then
    exe_name="$(basename "${exe_path}")"
    ln -sf "${exe_path}" "${CASE_DIR}/${exe_name}"
    log_msg "Linked ${exe_name} -> ${exe_path}"
  fi
done

# ---------------------------------------------------------------------------
# Namelist setup
# Preference order:
#   1. ${WRF_BWAVE_DIR}/namelist.input  (from Spack install, case-specific)
#   2. ${ROOT_DIR}/wrf/test/em_b_wave/namelist.input  (from submodule)
#   3. Generate a minimal working namelist (fallback)
# ---------------------------------------------------------------------------
NAMELIST_DST="${CASE_DIR}/namelist.input"
NAMELIST_SRC=""

if [[ -n "${WRF_BWAVE_DIR:-}" && -f "${WRF_BWAVE_DIR}/namelist.input" ]]; then
  NAMELIST_SRC="${WRF_BWAVE_DIR}/namelist.input"
  log_msg "Using namelist from Spack install: ${NAMELIST_SRC}"
elif [[ -f "${ROOT_DIR}/wrf/test/em_b_wave/namelist.input" ]]; then
  NAMELIST_SRC="${ROOT_DIR}/wrf/test/em_b_wave/namelist.input"
  log_msg "Using namelist from submodule: ${NAMELIST_SRC}"
else
  log_msg "Generating minimal em_b_wave namelist.input ..."
fi

# Also copy all other case-specific data files (e.g. input_jet, README.namelist)
# from WRF_BWAVE_DIR — these are needed by ideal.exe at runtime.
if [[ -n "${WRF_BWAVE_DIR:-}" && -d "${WRF_BWAVE_DIR}" ]]; then
  find "${WRF_BWAVE_DIR}" -maxdepth 1 -type f ! -name "*.exe" \
    -exec cp -n {} "${CASE_DIR}/" \;
  log_msg "Copied case-specific files from ${WRF_BWAVE_DIR}"
elif [[ -d "${ROOT_DIR}/wrf/test/em_b_wave" ]]; then
  find "${ROOT_DIR}/wrf/test/em_b_wave" -maxdepth 1 -type f ! -name "*.exe" \
    -exec cp -n {} "${CASE_DIR}/" \;
  log_msg "Copied case-specific files from submodule test/em_b_wave"
fi

if [[ -n "${NAMELIST_SRC}" ]]; then
  # namelist.input may have already been copied above; overwrite to be sure
  cp "${NAMELIST_SRC}" "${NAMELIST_DST}"
  log_msg "Copied namelist.input"
else
  # Fallback: write a minimal WRF 4.6 em_b_wave namelist.
  # The key I/O knobs are annotated; these are the ones in dftracer/knobs.yaml.
  cat > "${NAMELIST_DST}" << 'NAMELIST_EOF'
 &time_control
 run_days                            = 0,
 run_hours                           = 12,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 0001,
 start_month                         = 01,
 start_day                           = 01,
 start_hour                          = 00,
 start_minute                        = 00,
 start_second                        = 00,
 end_year                            = 0001,
 end_month                           = 01,
 end_day                             = 01,
 end_hour                            = 12,
 end_minute                          = 00,
 end_second                          = 00,
 history_interval                    = 60,       ! DFTracer knob: output frequency (minutes)
 frames_per_outfile                  = 1,         ! DFTracer knob: timesteps per wrfout file
 restart                             = .false.,
 restart_interval                    = 1440,      ! DFTracer knob: restart interval (minutes)
 io_form_history                     = 2,         ! DFTracer knob: 2=NetCDF4, 11=pnetcdf
 io_form_restart                     = 2,
 io_form_input                       = 2,
 io_form_boundary                    = 2,
 /

 &domains
 time_step                           = 60,
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 1,
 e_we                                = 100,
 e_sn                                = 100,
 e_vert                              = 41,
 p_top_requested                     = 5000,
 num_metgrid_levels                  = 27,
 num_metgrid_soil_levels             = 4,
 dx                                  = 100000,
 dy                                  = 100000,
 grid_id                             = 1,
 parent_id                           = 0,
 i_parent_start                      = 1,
 j_parent_start                      = 1,
 parent_grid_ratio                   = 1,
 parent_time_step_ratio              = 1,
 feedback                            = 1,
 smooth_option                       = 0,
 /

 &physics
 physics_suite                       = 'CONUS',
 mp_physics                          = -1,
 cu_physics                          = -1,
 ra_lw_physics                       = -1,
 ra_sw_physics                       = -1,
 bl_pbl_physics                      = -1,
 sf_sfclay_physics                   = -1,
 sf_surface_physics                  = -1,
 /

 &dynamics
 hybrid_opt                          = 2,
 w_damping                           = 0,
 diff_opt                            = 1,
 km_opt                              = 4,
 diff_6th_opt                        = 0,
 diff_6th_factor                     = 0.12,
 base_temp                           = 290.,
 damp_opt                            = 3,
 zdamp                               = 5000.,
 dampcoef                            = 0.2,
 khdif                               = 0,
 kvdif                               = 0,
 non_hydrostatic                     = .true.,
 moist_adv_opt                       = 1,
 scalar_adv_opt                      = 1,
 gwd_opt                             = 0,
 /

 &bdy_control
 spec_bdy_width                      = 5,
 spec_zone                           = 1,
 relax_zone                          = 4,
 specified                           = .false.,
 periodic_x                          = .true.,
 symmetric_y                         = .false.,
 open_xs                             = .false.,
 open_xe                             = .false.,
 open_ys                             = .true.,
 open_ye                             = .true.,
 nested                              = .false.,
 /

 &namelist_quilt
 nio_tasks_per_group                 = 0,         ! DFTracer knob: I/O quilting tasks per group (0=off)
 nio_groups                          = 1,         ! DFTracer knob: number of I/O quilting groups
 /
NAMELIST_EOF
  log_msg "Generated minimal namelist.input (fallback)"
fi

# ---------------------------------------------------------------------------
# Write case metadata
# ---------------------------------------------------------------------------
cat > "${CASE_DIR}/case_info.env" << EOF
CASE_NAME=${CASE_NAME}
CASE_DIR=${CASE_DIR}
WRF_EXE=${WRF_EXE}
IDEAL_EXE=${IDEAL_EXE:-}
WRF_PREFIX=${WRF_PREFIX}
NP_DEFAULT=${NP}
MPIEXEC_DEFAULT=${MPIEXEC}
EOF

log_msg "Case directory ready: ${CASE_DIR}"
log_msg "Contents:"
ls "${CASE_DIR}" | while read -r f; do log_msg "  ${f}"; done
log_msg ""
log_msg "Next steps:"
log_msg "  ./run.sh                   # baseline WRF run (no tracing)"
log_msg "  ./run.sh --dftracer-enable 1   # traced WRF run with DFTracer"
