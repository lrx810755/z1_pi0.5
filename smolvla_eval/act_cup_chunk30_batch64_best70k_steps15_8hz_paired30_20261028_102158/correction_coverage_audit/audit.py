import pathlib,json,sys,collections,numpy as np
from scipy.spatial.transform import Rotation
A=pathlib.Path(__file__).resolve().parent;E=A.parent
T=pathlib.Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_chunk30_batch64_20261028_003937')
sys.path.insert(0,str(E/'execution_audit'));from check import fk
split=json.loads((T/'split_manifest.json').read_text());records=split['source_records']
with np.load(T/'sampling_arrays.npz') as z:sampling={k:z[k] for k in z.files}
def stat(v):return dict(zip(['min','median','max'],map(float,np.percentile(v,[0,50,100])))) if v else None
def gravity(q):return Rotation.from_quat(q).inv().apply([0,0,1])
pool=[];metadata=[]
for r in records:
 if r['source']!='correction':continue
 p=pathlib.Path(r['source_path']);c=json.loads((p/'correction.json').read_text());frames=[json.loads(l) for l in (p/'frames.jsonl').read_text().splitlines()];physics=[json.loads(l) for l in (p/'physics_status.jsonl').read_text().splitlines()];times=np.array([x['simulation_time'] for x in physics]);begin=r['dataset_from_index'];end=r['dataset_to_index'];eligible=sampling['eligible_train'][begin:end]
 nstage=0;eligible_stage=0
 for f,row in enumerate(frames):
  instage=any(s['start']<=row['simulation_time']<=s['end'] for s in c['recovery_stages']);nstage+=instage;eligible_stage+=instage and bool(eligible[f])
  if not eligible[f] or f>=c['close_command_frame']:continue
  j=int(np.searchsorted(times,row['simulation_time']));j=min([max(0,j-1),min(j,len(times)-1)],key=lambda j:abs(times[j]-row['simulation_time']))
  if abs(times[j]-row['simulation_time'])>.075:continue
  s=physics[j]['status'];pos=s.get('relative_position_gripper_m');q=s.get('gripper_world_orientation_xyzw')
  if pos is None or q is None:continue
  pool.append(dict(episode=r['episode_index'],frame=f,kind=r['kind'],relative_position=pos,gravity=gravity(q).tolist()))
 keys=['task_id','source_type','measured_start_deviation_m','measured_world_z_deviation_m','measured_disturbance_m','placement_vertical_excess_m','placement_xy_error_m','placement_hover_duration_s','pregrasp_mode','prepared_cup_shift']
 metadata.append(dict(episode=r['episode_index'],kind=r['kind'],split=r['split'],frames=r['frames'],training_windows=int(eligible.sum()),training_sampling_mass=float(sampling['weights'][begin:end].sum()),recovery_stage_frames=nstage,eligible_training_stage_frames=int(eligible_stage),close_command_frame=c['close_command_frame'],**{k:c[k] for k in keys if k in c}))
assert pool
pos=np.array([p['relative_position'] for p in pool]);grav=np.array([p['gravity'] for p in pool]);failures=[]
for e in json.loads((E/'结果与失败分类.json').read_text())['episodes']:
 if e['success']:continue
 path=pathlib.Path(e['source_result']).parent;t=[json.loads(l) for l in (path/'trace.jsonl').read_text().splitlines()];q=np.array([x['joint_state'][:6] for x in t]);point,rot=fk(q,(.12083045,-.0028452,.0194797));obj=np.array([x['contact_status']['object_position_m'] for x in t]);oq=np.array([x['contact_status']['object_orientation_xyzw'] for x in t]);center=obj+Rotation.from_quat(oq).apply(np.tile([0,0,.0305],(len(t),1)))
 moved=np.linalg.norm(obj-np.array(e['initial_cup_position_m']),axis=1)>.03;firstmove=np.flatnonzero(moved)[0] if moved.any() else len(t);n=int(np.argmin(np.where(np.arange(len(t))<firstmove,np.linalg.norm(point-center,axis=1),np.inf)))
 s=t[n]['contact_status'];p0=np.array(s['relative_position_gripper_m']);g0=gravity(s['gripper_world_orientation_xyzw']);distance=np.linalg.norm(pos-p0,axis=1);angle=np.degrees(np.arccos(np.clip(grav@g0,-1,1)));ok=angle<=10
 k=int(np.argmin(np.where(ok,distance,np.inf))) if ok.any() else None
 maxdisp=float(np.max(np.linalg.norm(obj[:,:2]-obj[0,:2],axis=1)))
 # First displacement is only motion evidence, not proof it was caused by a finger impact.
 failures.append(dict(episode=e['episode'],category=e['observed_category'],chosen_time_s=t[n]['simulation_elapsed_s'],chosen_relative_position=p0.tolist(),max_cup_xy_displacement_m=maxdisp,first_3cm_displacement_s=float(t[firstmove]['simulation_elapsed_s']) if firstmove<len(t) else None,nearest_train_correction_position_distance_mm=float(distance[k]*1000) if k is not None else None,orientation_gravity_angle_deg=float(angle[k]) if k is not None else None,nearest_example=pool[k] if k is not None else None,coarse_match_20mm_10deg=bool(k is not None and distance[k]<=.02),coarse_match_10mm_10deg=bool(k is not None and distance[k]<=.01)))
summary={}
for kind in ['approach','close_transition','shifted_target','low_clearance','regrasp','transport','placement']:
 rows=[m for m in metadata if m['kind']==kind];summary[kind]=dict(episodes=len(rows),train_episodes=sum(m['split']=='train' for m in rows),training_mass=sum(m['training_sampling_mass'] for m in rows),start_deviation_m=stat([m['measured_start_deviation_m'] for m in rows]),z_deviation_m=stat([m['measured_world_z_deviation_m'] for m in rows]),loaded_disturbance_m=stat([m['measured_disturbance_m'] for m in rows if 'measured_disturbance_m' in m]),recovery_stage_frames=sum(m['recovery_stage_frames'] for m in rows),eligible_training_stage_frames=sum(m['eligible_training_stage_frames'] for m in rows))
report=dict(total_corrections=100,source_types=dict(collections.Counter(m['source_type'] for m in metadata)),summary=summary,train_correction_preclose_geometry_samples=len(pool),failure_representatives=len(failures),coarse_matches20mm=sum(x['coarse_match_20mm_10deg'] for x in failures),coarse_matches10mm=sum(x['coarse_match_10mm_10deg'] for x in failures),coarse_match_definition='one pre3cm-displacement closest-approach state per failed trial vs eligible chunk30 training correction preclose states; relative cup-center position within20mm and gravity-in-tool direction within10deg. Necessary geometric similarity only, not RGB/contact/trajectory coverage.',metadata=metadata,failures=failures)
(A/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['metadata','failures']},ensure_ascii=False,indent=2))
