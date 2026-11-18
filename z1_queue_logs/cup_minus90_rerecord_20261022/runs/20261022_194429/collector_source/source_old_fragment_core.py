"""Cup physical acceptance and command-aligned labels for partial clips."""
def valid_hold(s):
 p=s.get('object_position_m',[])
 return s.get('target_object')=='cup' and len(p)==3 and p[2]>=.05 and all(s.get(k) is True for k in ['physical_grasp_ok','lifted','bilateral_contact','target_collisions_enabled']) and all(s.get(k) is False for k in ['frozen','object_touching_table','left_finger_touching_table','right_finger_touching_table'])

def partial_labels(rows):
 close=next((i for i,r in enumerate(rows) if r['action'][-1]>-.895),None)
 if close is None or close==0:raise ValueError('missing open-to-close command transition')
 if any(r['action'][-1]<=-.895 for r in rows[close+1:]):raise ValueError('unexpected reopening')
 return [int(i>=close) for i in range(len(rows))],close,max(0,len(rows)-16)
