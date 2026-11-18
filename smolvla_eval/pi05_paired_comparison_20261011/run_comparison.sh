#!/usr/bin/env bash
set -euo pipefail
cd /home/lp531/smolvla_eval/pi05_paired_comparison_20261011
exec /usr/bin/python3 -u run_comparison.py >> comparison.log 2>&1
