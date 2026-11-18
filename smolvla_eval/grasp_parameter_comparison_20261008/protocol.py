import math
from collections import defaultdict

PROFILES={f'mu{s:g}_effort{e}':dict(static=s,dynamic=d,effort=e)
          for s,d in [(1.,.8),(3.,2.5),(6.,5.)] for e in [80,240,800]}
BASELINE='mu6_effort800'

def validate_profile(static,dynamic,effort):
    if not all(math.isfinite(x) for x in [static,dynamic,effort]):
        raise ValueError('nonfinite physics parameter')
    if not (0<=dynamic<=static<=6 and 0<effort<=800):
        raise ValueError('invalid friction or effort; expected 0<=dynamic<=static<=6, 0<effort<=800')

def rank_profiles(rows,expected,efforts):
    grouped=defaultdict(list)
    for row in rows:grouped[row['profile']].append(row)
    ranked=[]
    for name,trials in grouped.items():
        if len(trials)!=expected or not any(x['strict_success'] for x in trials):continue
        ranked.append(dict(profile=name,trials=len(trials),
                           strict_successes=sum(x['strict_success'] for x in trials),
                           grasps=sum(x['grasped'] for x in trials),
                           successes=sum(x['success'] for x in trials),
                           penetration_max_m=max(x['penetration_m'] for x in trials),
                           effort=efforts[name]))
    return sorted(ranked,key=lambda x:(-x['strict_successes'],-x['grasps'],-x['successes'],x['penetration_max_m'],x['effort'],x['profile']))

def strict_placement(result):
    st=result.get('final_status',{})
    xyz=st.get('object_position_m',[])
    return bool(result.get('success') and st.get('physical_grasp_ok') and st.get('release_sequence_complete')
                and len(xyz)==3 and abs(xyz[0]-.32)<=.055-.0242-.003
                and abs(xyz[1]-.14)<=.055-.0242-.003 and 0<=xyz[2]<=.025)
