"""Check pinned collection runtime before starting the authorized30-task job."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
pins=json.loads((ROOT/'runtime_pins.json').read_text())
for name,expected in pins.items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Collection runtime changed: '+name)
os.execv('/bin/bash',['bash','/home/lp531/SH/start_bottle_approach_height30_20261112.sh'])
