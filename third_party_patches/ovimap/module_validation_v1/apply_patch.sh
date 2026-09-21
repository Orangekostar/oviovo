#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT="f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_PATH="${SCRIPT_DIR}/ovimap_module_validation_v1.patch"
MODE="apply"

if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
  shift
fi

UPSTREAM_ROOT="${1:-$PWD}"
ACTUAL_COMMIT="$(git -C "${UPSTREAM_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "error: expected OVI-MAP ${EXPECTED_COMMIT}, found ${ACTUAL_COMMIT}" >&2
  exit 2
fi

TARGETS=(
  "mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/include/consistent_mapping/global_segment_map_py.h"
  "mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/global_segment_map_py.cpp"
  "scripts/panoptic_mapping_.py"
  "scripts/view_selection.py"
)

if ! git -C "${UPSTREAM_ROOT}" diff --quiet -- "${TARGETS[@]}"; then
  echo "error: patch target files have tracked modifications" >&2
  exit 3
fi
if [[ -n "$(git -C "${UPSTREAM_ROOT}" ls-files --others --exclude-standard -- "${TARGETS[@]}")" ]]; then
  echo "error: patch target files include untracked paths" >&2
  exit 3
fi

git -C "${UPSTREAM_ROOT}" apply --check "${PATCH_PATH}"
if [[ "${MODE}" == "check" ]]; then
  echo "OVI-MAP module-validation patch check passed at ${ACTUAL_COMMIT}"
  exit 0
fi

git -C "${UPSTREAM_ROOT}" apply "${PATCH_PATH}"
echo "Applied OVI-MAP module-validation patch at ${ACTUAL_COMMIT}"
