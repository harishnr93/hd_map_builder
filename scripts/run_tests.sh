#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/pytest_${timestamp}.log"

echo "Running pytest, logs -> ${log_file}"
(
  cd "${ROOT_DIR}"
  pytest
) 2>&1 | tee "${log_file}"

echo "Test log saved to ${log_file}"
