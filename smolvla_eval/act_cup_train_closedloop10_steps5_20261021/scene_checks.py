"""Reject a diagnostic reset unless it reproduces the recorded scene."""
import math

def check_scene(expected,status,joints,position_tolerance=.002,angle_tolerance=2.,joint_tolerance=.025):
 if status.get('seed')!=expected['seed'] or status.get('target_grid')!=expected['grid']:
  raise ValueError('Scene seed/grid does not match selected training demonstration')
 position_errors={};angle_errors={}
 for name,wanted in expected['object_positions_m'].items():
  got=status['object_positions_m'][name]
  error=math.sqrt(sum((a-b)**2 for a,b in zip(wanted,got)))
  if error>position_tolerance:raise ValueError(f'{name}: scene position differs by {error:.6f}m')
  position_errors[name]=error
 for name,wanted in expected['object_orientations_xyzw'].items():
  got=status['object_orientations_xyzw'][name]
  dot=abs(sum(a*b for a,b in zip(wanted,got)))/math.sqrt(sum(a*a for a in wanted)*sum(a*a for a in got))
  error=math.degrees(2*math.acos(min(1.,dot)))
  if error>angle_tolerance:raise ValueError(f'{name}: scene orientation differs by {error:.3f}deg')
  angle_errors[name]=error
 if len(joints)!=len(expected['initial_joint_state']) or any(x is None for x in joints):raise ValueError('Missing initial joints')
 joint_error=max(abs(a-b) for a,b in zip(joints,expected['initial_joint_state']))
 if joint_error>joint_tolerance:raise ValueError(f'Initial joints differ by {joint_error:.6f}rad')
 return dict(position_errors_m=position_errors,orientation_errors_deg=angle_errors,max_initial_joint_error_rad=joint_error)
