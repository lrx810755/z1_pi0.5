#!/bin/bash
set -eo pipefail
replay_output="/home/lp531/smolvla_eval/replay_debug_20261003/verified_replay_$(date +%Y%m%d_%H%M%S)"
exec /bin/bash /home/lp531/smolvla_eval/replay_debug_20261003/run_debug.sh --suite settled_replay --output "$replay_output"
