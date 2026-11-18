"""Time alignment and causal action checks on the recorded frame metadata."""
import numpy as np

IMAGE_STAMPS=['wrist_image_stamp','global_image_stamp','global_right_image_stamp']

def validate_timeline(rows):
    times=np.asarray([r['simulation_time'] for r in rows],float)
    stamps=np.asarray([[r[k] for k in IMAGE_STAMPS] for r in rows],float)
    commands=np.asarray([r['action_stamp'] for r in rows],float)
    brackets=np.asarray([r['joint_state_source_stamps'] for r in rows],float)
    tolerance=np.asarray([r['sync_tolerance_s'] for r in rows],float)
    if stamps.shape!=(len(rows),3) or brackets.shape!=(len(rows),2):raise ValueError('Unaligned sensor timestamp arrays')
    if not all(np.isfinite(a).all() for a in [times,stamps,commands,brackets,tolerance]):raise ValueError('Nonfinite sensor timestamps')
    if np.any(tolerance<=0) or np.any(tolerance>.045+1e-7):raise ValueError('Unapproved synchronization tolerance')
    if np.any(np.diff(stamps,axis=0)<=0):raise ValueError('Repeated or backward source image timestamps')
    spread=np.ptp(np.column_stack([times,stamps]),axis=1)
    if np.any(spread>tolerance+1e-5):raise ValueError('Camera/state timestamp spread exceeds recorded tolerance')
    ages=times-commands
    if np.any(ages< -1e-5) or np.any(ages>.4+1e-5):raise ValueError('Action is noncausal or older than0.4s')
    if np.any(brackets[:,0]>times+1e-5) or np.any(brackets[:,1]<times-1e-5) or np.any(np.diff(brackets,axis=1)>.1+1e-5):raise ValueError('Invalid joint-state interpolation bracket')
    return dict(max_image_state_spread_s=float(spread.max()),max_causal_action_age_s=float(ages.max()),max_state_bracket_s=float(np.diff(brackets,axis=1).max()))
