#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/data-preparation"

DATA_ROOT="${DATA_ROOT:-/p/lustre5/izzet/datasets/era5}"
RAW_NC_DIR="${RAW_NC_DIR:-${DATA_ROOT}/nc}"
REGRID_DIR="${REGRID_DIR:-${DATA_ROOT}/regrid}"
HDF5_DIR="${HDF5_DIR:-${DATA_ROOT}/hdf5}"
START_YEAR="${START_YEAR:-1979}"
END_YEAR="${END_YEAR:-2019}"
GRID_DEG="${GRID_DEG:-1.40625}"
DO_DOWNLOAD="${DO_DOWNLOAD:-1}"
DO_REGRID="${DO_REGRID:-1}"
DO_PREPROCESS="${DO_PREPROCESS:-1}"
DO_NORMALIZATION="${DO_NORMALIZATION:-1}"

WB2_FILE="${WB2_FILE:-1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}"
WB2_MISSING_FILE="${WB2_MISSING_FILE:-1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr}"

source "${ROOT_DIR}/setup_env.sh"

run_step() {
  local name="$1"
  shift
  echo "== ${name} =="
  "$@"
}

download_normalization_constants() {
  local base_url="https://raw.githubusercontent.com/tung-nd/stormer/refs/heads/main/normalization_constants"
  local files=(
    normalize_mean.npz
    normalize_std.npz
    normalize_diff_mean_6.npz
    normalize_diff_mean_12.npz
    normalize_diff_mean_24.npz
    normalize_diff_std_6.npz
    normalize_diff_std_12.npz
    normalize_diff_std_24.npz
  )

  mkdir -p "${HDF5_DIR}"
  for f in "${files[@]}"; do
    if [[ ! -f "${HDF5_DIR}/${f}" ]]; then
      curl -L --fail --retry 3 -o "${HDF5_DIR}/${f}" "${base_url}/${f}"
    fi
  done
}

mkdir -p "${RAW_NC_DIR}" "${REGRID_DIR}" "${HDF5_DIR}"

if [[ "${DO_DOWNLOAD}" == "1" ]]; then
  run_step "download weatherbench2 era5" \
    python "${SCRIPT_DIR}/download-wb2.py" \
      --file "${WB2_FILE}" \
      --save-dir "${RAW_NC_DIR}" \
      --start-year "${START_YEAR}" \
      --end-year "${END_YEAR}"

  run_step "download missing lake_depth" \
    python "${SCRIPT_DIR}/download-wb2-missing-var.py" \
      --file "${WB2_MISSING_FILE}" \
      --save-dir "${RAW_NC_DIR}" \
      --start-year "${START_YEAR}" \
      --end-year "${END_YEAR}"
fi

if [[ "${DO_REGRID}" == "1" ]]; then
  run_step "regrid netcdf" \
    python "${SCRIPT_DIR}/regrid-wb2.py" \
      --root-dir "${RAW_NC_DIR}" \
      --save-dir "${REGRID_DIR}" \
      --ddeg-out "${GRID_DEG}" \
      --start-year "${START_YEAR}" \
      --end-year "${END_YEAR}"
fi

if [[ "${DO_PREPROCESS}" == "1" ]]; then
  run_step "convert regridded netcdf to hdf5 train split" \
    python "${SCRIPT_DIR}/preprocess-data.py" \
      --root-dir "${REGRID_DIR}" \
      --save-dir "${HDF5_DIR}" \
      --split train \
      --start-year "${START_YEAR}" \
      --end-year "${END_YEAR}"
fi

if [[ "${DO_NORMALIZATION}" == "1" ]]; then
  run_step "download normalization constants" download_normalization_constants
fi

echo "Dataset prepared under ${DATA_ROOT}"
