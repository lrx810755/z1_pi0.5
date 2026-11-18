"""Scoped PI0.5 cup evaluator; shared ROS packages stay unchanged."""
import importlib.util
import json
import os
from pathlib import Path
import sys

R = Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
name = 'z1_isaac_bridge.pi05_cup_scoped_rollout'
spec = importlib.util.spec_from_file_location(name, R/'runtime_rollout.py')
rollout = importlib.util.module_from_spec(spec)
sys.modules[name] = rollout
spec.loader.exec_module(rollout)
original = rollout.load_training_coverage


def coverage(root):
    cells, positions, seeds = original(root)
    path = R/'excluded_seeds.json'
    if not path.exists():
        path = R/'excluded_seed_baseline.json'
    return cells, positions, seeds | set(json.loads(path.read_text()))


rollout.load_training_coverage = coverage

if __name__ == '__main__':
    rollout.main()
