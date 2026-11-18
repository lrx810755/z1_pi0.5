"""Run the existing evaluator offline, with the explicit paired seed list."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
from z1_isaac_bridge.smolvla_sim_rollout import main
main()
