#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date +%Y-%m-%d)"
output_dir="${1:-${repo_root}/dist}"
archive="${output_dir}/z1-smolvla-core-${stamp}.tar.zst"

mkdir -p "${output_dir}"

tar \
  --exclude='./.git' \
  --exclude='./.gitattributes' \
  --exclude='./.gitignore' \
  --exclude='./dist' \
  --exclude='./ros2_ws/build' \
  --exclude='./ros2_ws/install' \
  --exclude='./ros2_ws/log' \
  --exclude='./data/raw' \
  --exclude='./data/quarantine' \
  --exclude='./data/reviewed' \
  --exclude='./data/lerobot' \
  --exclude='./data/runtime' \
  --exclude='./docs/*.html' \
  --exclude='./docs/WORK_LOG_*' \
  --exclude='./docs/GRASP_LIFT_ISSUES_*' \
  --exclude='./docs/REPRODUCTION_GUIDE_*' \
  --exclude='./isaac_assets/z1/generated' \
  --exclude='./isaac_assets/z1/*backup*' \
  --exclude='./isaac_assets/z1/*.bak' \
  --exclude='./isaac_assets/z1/*.before_*' \
  --exclude='./isaac_assets/z1/scanned_objects/candidates' \
  --exclude='./isaac_assets/z1/scanned_objects/retired' \
  --exclude='*/__pycache__' \
  --exclude='*/.pytest_cache' \
  --exclude='*/.~*' \
  --exclude='*.pyc' \
  -I 'zstd -T0 -10' \
  --transform='s,^\./,z1-smolvla/,' \
  -cf "${archive}" \
  -C "${repo_root}" .

printf 'Created: %s\n' "${archive}"
printf 'Extract with: tar -I zstd -xf %s\n' "$(basename "${archive}")"
