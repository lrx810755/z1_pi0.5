"""Isolated, paired cup comparison contract."""
def geometry(shape):
 if shape=='cylinder':return dict(shape=shape,radius_m=.0252,height_m=.061)
 if shape=='frustum':return dict(shape=shape,bottom_radius_m=.018,top_radius_m=.0252,height_m=.061,sides=64)
 raise ValueError(shape)
def cases():
 return [dict(id=f'{shape}_h{height}_{position}_{repeat:02}',shape=shape,height_mm=height,position=position,repeat=repeat,target_grid=[column,1],cup_position_m=[round(.36+(column+.5)*(.69-.36)/5,6),round(-.27+1.5*(.16+.27)/5,6),0.],planning_attempts=1) for shape in ['frustum','cylinder'] for height in [34,32] for position,column in [('near',0),('middle',2),('far',4)] for repeat in range(1,11)]
def passed(result):
 last=result.get('final_status',{})
 return bool(result.get('service_success') and result.get('ever_physical_grasp') and last.get('physical_grasp_ok') and result.get('maximum_rise_m',0)>=.03 and not last.get('object_touching_table',True))

def valid_trial(result):
 """Setup failures and unfinished/cross-attempt operations cannot be grasp trials."""
 return result.get('failed_stage')!='reset' and not str(result.get('error','')).startswith('Service timeout:') and not result.get('cleanup_error')

# Same distraction layout in every paired condition; centres do not overlap cup tests.
DISTRACTOR_XY={'apple':(.43,.12),'peach':(.51,.12),'mouse':(.60,.12),'beverage_bottle':(.69,.12)}
