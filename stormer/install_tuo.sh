#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STORMER_MODULES="${STORMER_MODULES:-cray-python/3.11.7 gcc/13.3.1 rocm/6.3.1}"
export STORMER_PIP_EXTRA_INDEX_URL="${STORMER_PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/logs"
PY_LOG="${LOG_DIR}/install_python.log"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ensure_module_command() {
  if type module >/dev/null 2>&1; then
    return 0
  fi

  for init_script in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f "${init_script}" ]]; then
      # shellcheck disable=SC1090
      source "${init_script}"
      break
    fi
  done
}

mkdir -p "${LOG_DIR}"

ensure_module_command
for mod in ${STORMER_MODULES}; do
  module load "${mod}"
done

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip >> "${PY_LOG}" 2>&1
python -m pip uninstall -y pydftracer >> "${PY_LOG}" 2>&1 || true
python -m pip install \
  --extra-index-url "${STORMER_PIP_EXTRA_INDEX_URL}" \
  "torch==2.9.1+rocm6.3" \
  "torchvision==0.24.1+rocm6.3" \
  "dftracer==1.0.15" >> "${PY_LOG}" 2>&1
python -m pip install \
  h5py \
  numpy \
  strenum \
  timm \
  tqdm >> "${PY_LOG}" 2>&1

echo "Install complete. Log: ${PY_LOG}"
