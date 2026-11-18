from core import make_tasks,digest,atomic_json

def build(destination):
 plan=dict(version=1,target_object='cup',tasks=make_tasks(),quota=48,
  amplitudes={'failure_derived':'bounded translation and tilt in failure_directions.json, scales .55/.8/1','prepared_cup_shift_m':[.008,.012,.015]},
  recording_goal='dynamic_recovery_close_lift_hold',minimum_lift_m=.05,lift_target_m=.06,hold_s=.6,
  freeze_policy='Allowed only before approach. Unfreeze and enable collisions before any arm preparation/approach movement; reject any frozen recorded sample.',
  prepared_shift_note='Relocation in scene setup only, then physics settles. No teleport or freeze in recorded motion. Stale gripper pose deliberately preserves old aim.',
  split_rule='Whole cells (1,1),(2,3) validation, same as base dataset.',
  base_dataset='/home/lp531/smolvla_data/z1_tabletop_raw_cup_standard_three_camera_20261019',
  eval_usage='20261019 evaluation used to design perturbations; use fresh seeds for unbiased next evaluation.')
 plan['hash']=digest(plan);atomic_json(destination,plan);return plan
