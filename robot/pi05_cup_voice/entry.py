"""Isolated voice backend using the verified 10Hz cup evaluation runtime."""
import importlib.util
from pathlib import Path
import sys
R=Path(__file__).resolve().parent

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

runtime=load('z1_isaac_bridge.pi05_cup_voice_runtime',R/'runtime_rollout.py')
backend=load('z1_isaac_bridge.pi05_cup_voice_backend',R/'backend.py')
if __name__=='__main__':
    backend.main()
