#!/usr/bin/env bash
set -euo pipefail
#
# Montage Pegasus install - follows DFTracer docs/pegasus_montage.rst
# Uses Spack for: python, py-astropy, ant, openjdk (externals from packages.yaml when available)
# Submodules: Montage, montage-workflow-v3, pegasus (source for mpi-cluster)
# Condor and Pegasus from tarballs (platform-specific)
#
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${ROOT_DIR}" rev-parse --show-toplevel)"
WORKLOAD_NAME="$(basename "${ROOT_DIR}")"
SPACK_YAML="${ROOT_DIR}/spack.yaml"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/logs"
INSTALL_DIR="${ROOT_DIR}/install"
DOWNLOAD_DIR="${ROOT_DIR}/downloads"
CONDOR_DIR="${ROOT_DIR}/condor"
PEGASUS_BINARY_DIR="${ROOT_DIR}/pegasus-binary"
# Format: name|url|rel_path (use | because URLs contain :)
SUBMODULES=(
  "montage|https://github.com/Caltech-IPAC/Montage.git|${WORKLOAD_NAME}/montage"
  "montage-workflow-v3|https://github.com/pegasus-isi/montage-workflow-v3.git|${WORKLOAD_NAME}/montage-workflow-v3"
  "pegasus|https://github.com/pegasus-isi/pegasus.git|${WORKLOAD_NAME}/pegasus"
)
MONTAGE_TAG="v6.0"
SPACK_LOG="${LOG_DIR}/install_spack.log"
GIT_LOG="${LOG_DIR}/install_git.log"
PY_LOG="${LOG_DIR}/install_python.log"
CONDOR_PEGASUS_LOG="${LOG_DIR}/install_condor_pegasus.log"

mkdir -p "${LOG_DIR}" "${DOWNLOAD_DIR}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log_msg() { echo "[$(timestamp)] $*"; }

log_msg "Logs: git=${GIT_LOG} spack=${SPACK_LOG} python=${PY_LOG}"

if [[ ! -f "${SPACK_YAML}" ]]; then
  echo "spack.yaml not found at ${SPACK_YAML}" >&2
  exit 1
fi

source ~/spack/share/spack/setup-env.sh

# ---- Submodules ----
# Reset submodules to clean state (patches reapplied by build.sh); idempotent
log_msg "Resetting submodules for clean checkout ..."
git -C "${REPO_ROOT}" submodule foreach --recursive "git reset --hard HEAD 2>/dev/null; git clean -fd 2>/dev/null; true" >> "${GIT_LOG}" 2>&1

for entry in "${SUBMODULES[@]}"; do
  IFS='|' read -r name url rel_path <<< "${entry}"
  src_dir="${ROOT_DIR}/${name}"
  if [[ ! -e "${src_dir}/.git" ]]; then
    log_msg "git submodule add ${rel_path} ..."
    git -C "${REPO_ROOT}" submodule add "${url}" "${rel_path}" >> "${GIT_LOG}" 2>&1
    log_msg "git submodule add ${rel_path} done"
  fi
  log_msg "git submodule sync ${rel_path} ..."
  git -C "${REPO_ROOT}" submodule sync -- "${rel_path}" >> "${GIT_LOG}" 2>&1
  log_msg "git submodule update --init --recursive --force ${rel_path} ..."
  git -C "${REPO_ROOT}" submodule update --init --recursive --force "${rel_path}" >> "${GIT_LOG}" 2>&1
  log_msg "git submodule update ${rel_path} done"
done

# Pin montage to stable tag (--force discards local changes; build.sh reapplies patches)
if [[ -d "${ROOT_DIR}/montage" ]]; then
  log_msg "git checkout ${MONTAGE_TAG} in montage ..."
  (cd "${ROOT_DIR}/montage" && git fetch --tags --force && git checkout -f "${MONTAGE_TAG}" 2>/dev/null || true) >> "${GIT_LOG}" 2>&1
  log_msg "montage pinned at ${MONTAGE_TAG}"
fi

# ---- Spack ----
log_msg "spack concretize ..."
spack -e "${ROOT_DIR}" concretize -f >> "${SPACK_LOG}" 2>&1
log_msg "spack install ..."
spack -e "${ROOT_DIR}" install >> "${SPACK_LOG}" 2>&1
log_msg "spack install done"

# ---- Python venv ----
SPACK_PYTHON_BIN="$(spack -e "${ROOT_DIR}" location -i python)/bin/python3"
log_msg "Using Spack Python: ${SPACK_PYTHON_BIN}"

if [[ -d "${VENV_DIR}" ]]; then
  VENV_PY="${VENV_DIR}/bin/python"
  if [[ ! -x "${VENV_PY}" ]]; then
    log_msg "Recreating venv (missing python)"
    rm -rf "${VENV_DIR}"
  else
    VVER="$("${VENV_PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")"
    SVER="$("${SPACK_PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${VVER}" != "${SVER}" ]]; then
      log_msg "Recreating venv (version mismatch venv=${VVER} spack=${SVER})"
      rm -rf "${VENV_DIR}"
    fi
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  log_msg "Creating virtual environment at ${VENV_DIR}"
  "${SPACK_PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log_msg "pip install -r requirements.txt ..."
pip install --upgrade pip >> "${PY_LOG}" 2>&1
pip install -r "${ROOT_DIR}/requirements.txt" >> "${PY_LOG}" 2>&1
log_msg "pip install done"

# ---- Condor + Pegasus (from tarballs) ----
install_condor_pegasus() {
  log_msg "Installing Condor + Pegasus ..."
  detect_platform() {
    if [[ -f /etc/os-release ]]; then
      # shellcheck disable=SC1091
      . /etc/os-release
      case "${ID:-}" in
        ubuntu)
          case "${VERSION_ID:-}" in
            20.04|22.04|24.04) echo "ubuntu${VERSION_ID%.*}"; return ;;
            *) echo "ubuntu22"; return ;;
          esac ;;
        rhel|centos|rocky|almalinux)
          v="${VERSION_ID:-8}"
          echo "rhel${v%.*}"; return ;;
        *) echo "rhel7"; return ;;
      esac
    fi
    echo "rhel7"
  }

  PLATFORM="${CONDOR_PEGASUS_PLATFORM:-$(detect_platform)}"
  log_msg "Platform: ${PLATFORM}"

  CONDOR_BASE="https://research.cs.wisc.edu/htcondor/tarball/24.x/current"
  PEGASUS_BASE="https://download.pegasus.isi.edu/pegasus/5.0.7"
  case "${PLATFORM}" in
    ubuntu20)
      CONDOR_URL="${CONDOR_TARBALL_URL:-https://research.cs.wisc.edu/htcondor/tarball/23.x/current/condor-x86_64_Ubuntu20-stripped.tar.gz}"
      PEGASUS_BIN="${PEGASUS_BINARY_URL:-${PEGASUS_BASE}/pegasus-binary-5.0.7-x86_64_ubuntu_20.tar.gz}"
      PEGASUS_WRK="${PEGASUS_WORKER_URL:-${PEGASUS_BASE}/pegasus-worker-5.0.7-x86_64_ubuntu_20.tar.gz}"
      ;;
    ubuntu22)
      CONDOR_URL="${CONDOR_TARBALL_URL:-${CONDOR_BASE}/condor-x86_64_Ubuntu22-stripped.tar.gz}"
      PEGASUS_BIN="${PEGASUS_BINARY_URL:-${PEGASUS_BASE}/pegasus-binary-5.0.7-x86_64_ubuntu_22.tar.gz}"
      PEGASUS_WRK="${PEGASUS_WORKER_URL:-${PEGASUS_BASE}/pegasus-worker-5.0.7-x86_64_ubuntu_22.tar.gz}"
      ;;
    ubuntu24)
      CONDOR_URL="${CONDOR_TARBALL_URL:-${CONDOR_BASE}/condor-x86_64_Ubuntu24-stripped.tar.gz}"
      # Pegasus has no ubuntu24; use ubuntu_22
      PEGASUS_BIN="${PEGASUS_BINARY_URL:-${PEGASUS_BASE}/pegasus-binary-5.0.7-x86_64_ubuntu_22.tar.gz}"
      PEGASUS_WRK="${PEGASUS_WORKER_URL:-${PEGASUS_BASE}/pegasus-worker-5.0.7-x86_64_ubuntu_22.tar.gz}"
      ;;
    rhel7|rhel8)
      CONDOR_URL="${CONDOR_TARBALL_URL:-https://research.cs.wisc.edu/htcondor/tarball/10.x/current/condor-x86_64_CentOS8-stripped.tar.gz}"
      PEGASUS_BIN="${PEGASUS_BINARY_URL:-https://download.pegasus.isi.edu/pegasus/5.0.7/pegasus-binary-5.0.7-x86_64_rhel_7.tar.gz}"
      PEGASUS_WRK="${PEGASUS_WORKER_URL:-https://download.pegasus.isi.edu/pegasus/5.0.7/pegasus-worker-5.0.7-x86_64_rhel_7.tar.gz}"
      ;;
    *)
      if [[ -z "${CONDOR_TARBALL_URL:-}" || -z "${PEGASUS_BINARY_URL:-}" ]]; then
        log_msg "Unknown platform ${PLATFORM}. Set CONDOR_TARBALL_URL, PEGASUS_BINARY_URL, PEGASUS_WORKER_URL to skip or override."
        return 0
      fi
      CONDOR_URL="${CONDOR_TARBALL_URL}"
      PEGASUS_BIN="${PEGASUS_BINARY_URL}"
      PEGASUS_WRK="${PEGASUS_WORKER_URL}"
      ;;
  esac

  CONDOR_TAR="${DOWNLOAD_DIR}/$(basename "${CONDOR_URL}")"
  if [[ ! -f "${CONDOR_TAR}" ]]; then
    log_msg "Downloading Condor ..."
    curl -fSL -o "${CONDOR_TAR}" "${CONDOR_URL}" 2>/dev/null || wget -q -O "${CONDOR_TAR}" "${CONDOR_URL}" 2>/dev/null || { log_msg "Condor download failed"; return 0; }
  fi

  rm -rf "${CONDOR_DIR}"
  mkdir -p "${CONDOR_DIR}"
  log_msg "Extracting Condor ..."
  (cd /tmp && rm -rf condor-*stripped && tar -x -f "${CONDOR_TAR}" -C /tmp)
  CONDOR_EXTRACTED=$(find /tmp -maxdepth 1 -type d -name 'condor-*stripped' 2>/dev/null | head -1)
  if [[ -n "${CONDOR_EXTRACTED}" && -d "${CONDOR_EXTRACTED}" ]]; then
    cp -r "${CONDOR_EXTRACTED}"/* "${CONDOR_DIR}/"
    rm -rf "${CONDOR_EXTRACTED}"
  else
    tar -x -f "${CONDOR_TAR}" -C "${CONDOR_DIR}" --strip-components=1
  fi
  log_msg "Configuring Condor ..."
  (cd "${CONDOR_DIR}" && ./bin/make-personal-from-tarball) >> "${CONDOR_PEGASUS_LOG}" 2>&1

  for url in "${PEGASUS_BIN}" "${PEGASUS_WRK}"; do
    f="${DOWNLOAD_DIR}/$(basename "${url}")"
    [[ ! -f "${f}" ]] && { curl -fSL -o "${f}" "${url}" 2>/dev/null || wget -q -O "${f}" "${url}" 2>/dev/null || true; }
  done
  rm -rf "${PEGASUS_BINARY_DIR}"
  mkdir -p "${PEGASUS_BINARY_DIR}"
  for t in "${DOWNLOAD_DIR}"/pegasus-*.tar.gz; do
    [[ -f "${t}" ]] && tar -xzf "${t}" -C "${PEGASUS_BINARY_DIR}" --strip-components=1 2>/dev/null || tar -xzf "${t}" -C "${PEGASUS_BINARY_DIR}" 2>/dev/null || true
  done

  log_msg "Creating consolidated install ..."
  rm -rf "${INSTALL_DIR}"
  mkdir -p "${INSTALL_DIR}"
  cp -a "${CONDOR_DIR}"/. "${INSTALL_DIR}/"
  for sub in bin sbin lib lib64 share etc; do
    if [[ -d "${PEGASUS_BINARY_DIR}/${sub}" ]]; then
      mkdir -p "${INSTALL_DIR}/${sub}"
      cp -r "${PEGASUS_BINARY_DIR}/${sub}"/* "${INSTALL_DIR}/${sub}/" 2>/dev/null || true
    fi
  done
  log_msg "Condor + Pegasus installed to ${INSTALL_DIR}"
}

install_condor_pegasus 2>&1 | tee -a "${CONDOR_PEGASUS_LOG}" || log_msg "Condor/Pegasus install had issues (check ${CONDOR_PEGASUS_LOG})"

# Install six into Pegasus install (pegasus-transfer in Condor jobs needs it)
PEGASUS_PY_LIB=$(find "${INSTALL_DIR}"/lib* "${INSTALL_DIR}"/usr/lib* -maxdepth 3 -type d -path "*/python*/dist-packages" 2>/dev/null | head -1)
if [[ -n "${PEGASUS_PY_LIB}" ]]; then
  log_msg "Installing six into Pegasus install (${PEGASUS_PY_LIB}) ..."
  pip install --target "${PEGASUS_PY_LIB}" six >> "${PY_LOG}" 2>&1
  log_msg "six installed for pegasus-transfer"
fi

log_msg "Install complete."
echo ""
echo "Next: ./build.sh to compile Montage and pegasus-mpi-cluster"
echo "Source pinned: montage ${MONTAGE_TAG}, montage-workflow-v3, pegasus (main)"
echo "Logs: ${GIT_LOG} ${SPACK_LOG} ${PY_LOG} ${CONDOR_PEGASUS_LOG}"
