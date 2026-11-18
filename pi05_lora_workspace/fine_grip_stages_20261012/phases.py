"""Stage labels for sampling only; commands and binary action targets are unchanged."""
import numpy as np
NAMES=['other','preclose','closing','prelift_hold','initial_lift','stable_hold']
WEIGHTS=np.array([1.,1.5,2.,2.,1.5,1.])

def annotate(rows,close,opening=None,fps=10,tolerance=.005,min_hold_frames=3):
    end=len(rows) if opening is None else opening
    if not (0<close<end<=len(rows)):raise ValueError('Invalid close/release boundary')
    lift=next((i for i in range(close,end) if rows[i]['phase']=='PHYSICAL_TEST_LIFT'),None)
    if lift is None or lift<=close:raise ValueError('Missing physical lift onset')
    commands=np.array([r['action'][-1] for r in rows],dtype=float)
    if not np.isfinite(commands).all():raise ValueError('Nonfinite gripper command')
    # A terminal plateau avoids treating a temporary pause before a re-adjustment as hold.
    hold=next((i for i in range(close+1,lift-min_hold_frames+1)
               if all(r['phase']=='GRASPING' for r in rows[i:lift])
               and np.ptp(commands[i-1:lift])<=tolerance),None)
    stable=next((i for i in range(lift+4,end)
                 if all(r.get('physical_grasp_ok') and r.get('grasp_success')
                        and not r.get('object_touching_table') for r in rows[i-4:i+1])),None)
    if stable is None:raise ValueError('No five-frame stable grasp before release/end')
    ids=np.zeros(len(rows),dtype=np.int8)
    ids[max(0,close-fps):close]=1
    ids[close:lift]=2
    if hold is not None:ids[hold:lift]=3
    ids[lift:stable+1]=4
    ids[stable+1:min(end,stable+1+int(1.5*fps))]=5
    return ids,WEIGHTS[ids],dict(preclose_start=max(0,close-fps),close_start=close,
          hold_start=hold,lift_start=lift,stable_confirmed=stable,opening=opening,
          hold_frames=0 if hold is None else lift-hold,
          hold_detection='terminal commanded-gripper plateau; not a contact measurement',
          plateau_tolerance_rad=tolerance,min_hold_frames=min_hold_frames)
