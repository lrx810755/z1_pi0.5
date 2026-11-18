"""Peach physical acceptance and command-aligned labels for partial clips."""
def valid_hold(s):
 p=s.get('object_position_m',[])
 return s.get('target_object')=='peach' and len(p)==3 and p[2]>=.05 and all(s.get(k) is True for k in ['physical_grasp_ok','lifted','bilateral_contact','target_collisions_enabled']) and all(s.get(k) is False for k in ['frozen','object_touching_table','left_finger_touching_table','right_finger_touching_table'])

def partial_labels(rows, close_command_time=None, reopen_command_times=()):
 """Label command-aligned closure after an explicit request, not an opening ramp.

 The no-event form is retained for legacy callers; current recording requires
 the explicit event and uses the causal action stamp to exclude stale commands.
 """
 import math
 actions=[float(r['action'][-1]) for r in rows]
 if not actions or not all(math.isfinite(x) for x in actions):
  raise ValueError('invalid gripper command sequence')
 if close_command_time is not None:
  close_command_time=float(close_command_time)
  if not math.isfinite(close_command_time):raise ValueError('invalid close command time')
  if any(float(t)>=close_command_time for t in reopen_command_times):
   raise ValueError('unexpected reopening: explicit open command after close request')
  stamps=[float(r.get('action_stamp',r.get('simulation_time',float('nan')))) for r in rows]
  if not all(math.isfinite(t) for t in stamps):raise ValueError('missing causal command timestamps')
  close=next((i for i,(a,t) in enumerate(zip(actions,stamps)) if t>=close_command_time and a>-.895),None)
 else:
  close=next((i for i,a in enumerate(actions) if a>-.895),None)
 if close is None or close==0:raise ValueError('missing open-to-close command transition')
 # Slow closing can legitimately traverse [-.895, -.88] for several frames.
 # Only a later return to the open band after definite closure is a reopening.
 closed_band=False
 for a in actions[close:]:
  if closed_band and a<=-.895:raise ValueError('unexpected reopening: command returned to open band')
  if a>=-.88:closed_band=True
 return [int(i>=close) for i in range(len(rows))],close,max(0,len(rows)-16)
