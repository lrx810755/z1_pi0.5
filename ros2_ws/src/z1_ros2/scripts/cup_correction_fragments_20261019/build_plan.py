"""Cup side-pinch perturbations on the twenty validated base cells."""
from core import make_tasks,digest,atomic_json

def build(destination):
    plan=dict(version=1,target_object='cup',tasks=make_tasks(),
      amplitudes=dict(lateral_m=[.003,.006],shallow_m=[.005,.010],deep_m=[.0015,.003],world_z_m=[.001,.002],tilt_deg=[1.5,3]),
      note='Bottle correction workflow with cup-specific shallow side-pinch geometry. Conservative perturbations; not measured autonomous failure quantiles.',
      recording_goal='correct_open_grasp_lift_hold',minimum_lift_m=.05,lift_target_m=.06,hold_s=.6,
      spatial_shape=[5,4],quota=60,split_rule='whole cells (1,1),(2,3) reserved for validation; reconcile with base split before training',
      base_dataset='/home/lp531/smolvla_data/z1_tabletop_raw_cup_standard_three_camera_20261019')
    plan['hash']=digest(plan);atomic_json(destination,plan);return plan
