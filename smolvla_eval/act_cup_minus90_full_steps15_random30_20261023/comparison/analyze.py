import json,math,xml.etree.ElementTree as ET,collections
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent;E=R.parent
TRAIN=Path('/home/lp531/pi05_lora_workspace/act_cup_minus90_full_cached_20261023')
manifest=json.loads((TRAIN/'split_manifest.json').read_text());URDF=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf');robot=ET.parse(URDF).getroot();chain=[]
for i in range(1,7):
 j=robot.find(f"joint[@name='joint{i}']");o=j.find('origin');chain.append((np.fromstring(o.get('xyz'),sep=' '),Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix(),np.fromstring(j.find('axis').get('xyz'),sep=' ')))
j=robot.find("joint[@name='gripperStatorLink']/origin");stator_translation=np.fromstring(j.get('xyz'),sep=' ');stator_rotation=Rotation.from_euler('xyz',np.fromstring(j.get('rpy'),sep=' ')).as_matrix()
def fk(q):
 p=np.zeros(3);r=np.eye(3)
 for angle,(offset,rotation,axis) in zip(q,chain):p+=r@offset;r=r@rotation@Rotation.from_rotvec(float(angle)*axis).as_matrix()
 return p+r@stator_translation,r@stator_rotation

def roll(q):
 _,r=fk(q);x=r[:,0];z=np.array([0.,0,1]);z=z-(z@x)*x;z/=np.linalg.norm(z);y=np.cross(z,x)
 return math.degrees(math.atan2(r[:,1]@z,r[:,1]@y))

def inside(p):
 return bool(p is not None and len(p)==3 and .11283045<=p[0]<=.12883045 and abs(p[1])<=.07 and .008<=p[2]<=.0227797)
def point(t):return t.get('contact_status',{}).get('relative_position_gripper_m')
def quant(x):
 a=np.asarray(x);return dict(min=np.min(a,axis=0).tolist(),p10=np.quantile(a,.1,axis=0).tolist(),median=np.median(a,axis=0).tolist(),p90=np.quantile(a,.9,axis=0).tolist(),max=np.max(a,axis=0).tolist()) if len(a) else None

demos=[]
for rec in manifest['source_records']:
 path=Path(rec['source_path']);rows=[json.loads(l) for l in (path/'frames.jsonl').read_text().splitlines()];c=rec['close_frame'];o=rec['open_frame'];q=rows[max(0,c-1)]['observation.state'];p,rot=fk(q)
 # Root position remains approximately fixed before first closure; use the
 # independently archived actual pre-approach position for this comparison.
 start=json.loads((path/'episode_start.json').read_text());scene=start['scene_randomization'];obj=np.asarray(scene['object_positions_m']['cup'])+np.array([0,0,.0305]);local=(rot.T@(obj-p)).tolist()
 a=np.asarray([x['action'][-1] for x in rows]);strong=next(i for i in range(c,len(a)) if a[i]>-.75)
 low=next((i for i in range(c,len(a)) if a[i]>-.8),None)
 demos.append(dict(episode=rec['episode_index'],source=rec['source'],path=str(path),grid=rec['grid'],seed=rec['seed'],position=scene['object_positions_m']['cup'],frames=len(rows),close_frame=c,open_frame=o,roll_preclose=roll(q),preclose_local=local,preclose_state=q,first_strong_close_s=(strong-c)/10,close_action_start=float(a[c]),close_action_at_strong=float(a[strong]),max_jaw_command_step=float(np.max(abs(np.diff(a)))),closing_ramp_actions=a[max(0,c-2):min(len(a),strong+5)].tolist()))
base=[d for d in demos if d['source']=='original'];median=np.median([d['preclose_local'] for d in base],axis=0)
episodes=[]
report=json.loads((E/'final_box_results.json').read_text())
for final in report['episodes']:
 path=Path(final['source_result']);result=json.loads(path.read_text());trace=[json.loads(l) for l in (path.parent/'trace.jsonl').read_text().splitlines()];chunks=[json.loads(l) for l in (path.parent/'prediction_chunks.jsonl').read_text().splitlines()]
 events=[];was=False
 for i,t in enumerate(trace):
  now=t['execution_target'][-1]>-.75
  if now!=was:
   contact=t.get('contact_status',{});loc=point(t)
   events.append(dict(kind='close' if now else 'open',trace_index=i,time=t['simulation_elapsed_s'],local=loc,inside_capture=inside(loc),roll=roll(t['joint_state']),object_position=contact.get('object_position_m'),ee_position=contact.get('gripper_world_position_m'),left=contact.get('left_finger_contact_object'),right=contact.get('right_finger_contact_object'),physical_grasp=contact.get('physical_grasp_ok'),was_grasped=contact.get('was_grasped'),actual_jaw=t['joint_state'][-1],command_jaw=t['command'][-1]))
  was=now
 first=next((x for x in events if x['kind']=='close'),None)
 pre=trace[:first['trace_index']+1] if first else trace
 contacts=[t for t in trace if t['contact_status'].get('left_finger_contact_object')=='cup' or t['contact_status'].get('right_finger_contact_object')=='cup']
 phys=[t for t in trace if t.get('physical_grasp_ok') or t['contact_status'].get('physical_grasp_ok')]
 bilateral=[t for t in trace if t['contact_status'].get('left_finger_contact_object')=='cup' and t['contact_status'].get('right_finger_contact_object')=='cup']
 nearest=min(base,key=lambda d:float(np.linalg.norm(np.array(d['position'][:2])-result['target_position_m'][:2])))
 candidates=[(i,t) for i,t in enumerate(pre) if point(t) is not None]
 best=min(candidates,key=lambda it:np.linalg.norm(np.array(point(it[1]))-median)) if candidates else None
 initial=np.asarray(result['target_position_m']);beforemax=max((np.linalg.norm(np.asarray(t['contact_status']['object_position_m'])[:2]-initial[:2]) for t in pre if t['contact_status'].get('object_position_m')),default=0.)
 speeds=[]
 for a,b in zip(trace,trace[1:]):
  dt=b['simulation_elapsed_s']-a['simulation_elapsed_s']
  if dt>1e-5:speeds.append(abs(b['joint_state'][-1]-a['joint_state'][-1])/dt)
 predmax=max((max(x['gripper_prediction']) for x in chunks),default=None)
 out=dict(episode=final['episode'],seed=final['seed'],success=final['success'],category=final['legacy_failure_category'],path=str(path.parent),target_position=result['target_position_m'],first_close=first,events=events,close_count=sum(x['kind']=='close' for x in events),open_count=sum(x['kind']=='open' for x in events),ever_capture_before_close=any(inside(point(t)) for t in pre),ever_capture=any(inside(point(t)) for t in trace),ever_physical=bool(phys),first_physical_time=phys[0]['simulation_elapsed_s'] if phys else None,bilateral_seen=bool(bilateral),finger_table=any(t['contact_status'].get(k) for t in trace for k in ['left_finger_touching_table','right_finger_touching_table']),max_cup_z=max((t['contact_status'].get('object_position_m') or [0,0,0])[2] for t in trace),cup_shift_before_first_close_m=float(beforemax),best_preclose=dict(trace_index=best[0],time=best[1]['simulation_elapsed_s'],local=point(best[1]),roll=roll(best[1]['joint_state'])) if best else None,max_predicted_close=predmax,predicted_close_only_discarded_last_step=sum(x.get('first_close_step_06')==15 for x in chunks),nearest_demo=nearest,nearest_demo_xy_distance_m=float(np.linalg.norm(np.array(nearest['position'][:2])-result['target_position_m'][:2])),final_status=result['final_status'],max_measured_jaw_speed_rad_s=max(speeds,default=0),video_paths={cam:str(path.parent/'videos'/name) for cam,name in [('global','global_third_person.mp4'),('right','global_right.mp4'),('wrist','wrist.mp4')]})
 episodes.append(out)
summary=dict(total=30,successes=sum(e['success'] for e in episodes),never_closed=[e['episode'] for e in episodes if not e['first_close']],first_close_outside_capture=[e['episode'] for e in episodes if e['first_close'] and not e['first_close']['inside_capture']],first_close_roll_outside_3deg=[e['episode'] for e in episodes if e['first_close'] and abs(e['first_close']['roll']+90)>3],first_close_roll_outside_10deg=[e['episode'] for e in episodes if e['first_close'] and abs(e['first_close']['roll']+90)>10],never_entered_capture=[e['episode'] for e in episodes if not e['ever_capture']],never_entered_capture_before_close=[e['episode'] for e in episodes if not e['ever_capture_before_close']],ever_physical=[e['episode'] for e in episodes if e['ever_physical']],never_bilateral=[e['episode'] for e in episodes if not e['bilateral_seen']],repeated_close=[e['episode'] for e in episodes if e['close_count']>1],preclose_cup_shift_over5mm=[e['episode'] for e in episodes if e['cup_shift_before_first_close_m']>.005],finger_table=[e['episode'] for e in episodes if e['finger_table']],demo_basic_roll=quant([d['roll_preclose'] for d in base]),demo_basic_preclose_local=quant([d['preclose_local'] for d in base]),demo_close_ramp_to_minus075_s=quant([d['first_strong_close_s'] for d in demos]),demo_max_jaw_command_step=quant([d['max_jaw_command_step'] for d in demos]),eval_firstclose_roll=quant([e['first_close']['roll'] for e in episodes if e['first_close']]),demo_and_evaluation_different_gripper_execution=True)
(R/'comparison.json').write_text(json.dumps(dict(summary=summary,demos=demos,episodes=episodes),ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
for e in episodes:
 c=e['first_close'];print(e['episode'],e['category'],'close_n',e['close_count'],'first',None if c is None else [round(c['time'],2),[round(x,3) for x in c['local']],round(c['roll'],1),c['inside_capture']],'phys',e['ever_physical'],'best',e['best_preclose']['local'],'shift',round(e['cup_shift_before_first_close_m']*1000,1))
