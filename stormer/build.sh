#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${ROOT_DIR}/logs/build.log"

mkdir -p "${ROOT_DIR}/logs"
source "${ROOT_DIR}/setup_env.sh"

mapfile -t PY_FILES < <(find "${ROOT_DIR}" \
  -path "${ROOT_DIR}/.venv" -prune -o \
  -path "${ROOT_DIR}/logs" -prune -o \
  -path "${ROOT_DIR}/output" -prune -o \
  -name '*.py' -print | sort)

python -m py_compile "${PY_FILES[@]}" > "${LOG_FILE}" 2>&1
python - <<'PY' >> "${LOG_FILE}" 2>&1
import train
print("stormer import OK")
PY

echo "Build checks complete. Log: ${LOG_FILE}"
