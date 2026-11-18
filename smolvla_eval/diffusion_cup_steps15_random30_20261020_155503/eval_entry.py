"""Exclude all source and prior evaluation seeds, keeping the same coverage gate."""
import json,os
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
from z1_isaac_bridge import smolvla_sim_rollout as module
load=module.load_training_coverage
excluded=set(json.loads(Path(__file__).with_name('excluded_seeds.json').read_text()))
def coverage(root):
 cells,positions,seeds=load(root)
 return cells,positions,seeds|excluded
module.load_training_coverage=coverage
module.main()
