"""Apple-specific bounded curriculum; no bottle-failure magnitudes reused."""
from core import make_tasks,digest,atomic_json

def build(destination):
    plan=dict(version=1,target_object='apple',tasks=make_tasks(),
      amplitudes=dict(lateral_m=[.005,.010],shallow_m=[.007,.014],deep_m=[.002,.004],world_z_m=[.002,.004],tilt_deg=[3,5]),
      note='Conservative apple curriculum based on low fruit geometry and current contact/edge failures; not measured autonomous apple failure quantiles.',
      recording_goal='correct_open_grasp_lift_hold',minimum_lift_m=.05,lift_target_m=.06,hold_s=2.2,
      spatial_shape=[4,5],quota=60,split_rule='all fragments in cells (1,1),(2,3) reserved for validation; reconcile with base dataset split before training')
    plan['hash']=digest(plan);atomic_json(destination,plan);return plan
