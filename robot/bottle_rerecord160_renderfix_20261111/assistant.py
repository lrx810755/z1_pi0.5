#!/usr/bin/python3
"""Original bottle behavior; only the requested candidate ordering is overridden."""
import importlib.util,sys
from pathlib import Path
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('z1_isaac_bridge.original_bottle_snapshot',R/'base_assistant_snapshot.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
base.GRASP_PROFILES['beverage_bottle']['candidate_azimuth_offsets_deg']=(0.,0.,30.,45.,45.,-30.,-45.,-45.)
base.GRASP_PROFILES['beverage_bottle']['fallback_relative_azimuths_deg']=(10.,20.,40.,-10.,-20.,-40.)
base.GRASP_PROFILES['beverage_bottle']['candidate_count']=8+len(base.GRASP_PROFILES['beverage_bottle'].get('extra_candidate_azimuths_deg',()))+6
class OriginalBottleAssistant(base.GraspAssistant):
    def __init__(self):
        super().__init__()
        # Retain the scheduler handshake; original grasp logic does not use it.
        self.declare_parameter('bottle_collection_column',-1)
base.GraspAssistant=OriginalBottleAssistant
if __name__=='__main__':
    if (R/'horizontal50_ready.json').exists():
        from horizontal50 import main
        main()
    else:base.main()
