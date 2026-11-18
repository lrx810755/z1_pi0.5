import collections,hashlib,json,time,xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
OUT=Path(__file__).resolve().parent
ROOT=Path('/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1')
INFO=json.loads((ROOT/'dataset_info.json').read_text())
CAMS=('wrist','global','global_right')
cv2.setNumThreads(1)
robot=ET.parse(OUT/'robot.urdf').getroot();chain=[];limits=[]
for i in range(1,7):
 j=robot.find(f"joint[@name='joint{i}']");o=j.find('origin');t=np.eye(4);t[:3,3]=np.fromstring(o.get('xyz'),sep=' ');t[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix();chain.append((t,np.fromstring(j.find('axis').get('xyz'),sep=' ')));lim=j.find('limit');limits.append((float(lim.get('lower')),float(lim.get('upper'))))
limits.append((-np.pi/2,0));limits=np.array(limits)
def fk(q):
 t=np.eye(4)
 for angle,(origin,axis) in zip(q,chain):
  rot=np.eye(4);rot[:3,:3]=R.from_rotvec(angle*axis).as_matrix();t=t@origin@rot
 return t[:3,3]+t[:3,:3]@np.array([.051,0,0]),t[:3,:3]
def roll(matrix):
 x=matrix[:,0];z=np.array([0.,0.,1.]);z=z-x*np.dot(z,x);z/=np.linalg.norm(z);y=np.cross(z,x)
 return float(np.rad2deg(np.arctan2(np.dot(matrix[:,1],z),np.dot(matrix[:,1],y))))
def stats(v):
 a=np.array(v,dtype=float);a=a[np.isfinite(a)]
 return {} if not len(a) else dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))
def audit_episode(p):
 rows=[json.loads(l) for l in (p/'frames.jsonl').read_text().splitlines() if l.strip()]
 summary=json.loads((p/'episode_summary.json').read_text());start=json.loads((p/'episode_start.json').read_text())
 q=np.array([r['observation.state'] for r in rows]);a=np.array([r['action'] for r in rows]);n=len(rows)
 violations=collections.Counter();examples=[]
 def flag(kind,i):
  violations[kind]+=1
  if len(examples)<15:examples.append(dict(kind=kind,frame=i))
 spreads=[];state_gaps=[];ages=[];latencies=[];prev={};hashes={};repeat=collections.Counter();repeat_moving=collections.Counter();maxrepeat={c:0 for c in CAMS};runrepeat={c:0 for c in CAMS};dark=collections.Counter();means={c:[] for c in CAMS}
 for i,r in enumerate(rows):
  if r['frame_index']!=i:flag('frame_index',i)
  if abs(r['timestamp']-i/10)>1e-6:flag('relative_timestamp',i)
  if r['target_object']!='beverage_bottle':flag('target',i)
  t=r['simulation_time'];ss=[r[c+'_image_stamp'] for c in CAMS];spread=max([t,*ss])-min([t,*ss]);spreads.append(spread)
  if spread>.045+1e-8:flag('camera_sync',i)
  left,right=r['joint_state_source_stamps'];state_gaps.append(right-left)
  if not left-1e-8<=t<=right+1e-8 or right-left>.1+1e-8:flag('state_interpolation',i)
  age=t-r['action_stamp'];ages.append(age);latencies.append(r['capture_time']-t)
  if age< -1e-8 or age>.4+1e-8:flag('causal_action_age',i)
  if abs(age-r['action_age_s'])>1e-7:flag('action_age_metadata',i)
  if i and abs(t-rows[i-1]['simulation_time']-.1)>1e-6:flag('simulation_time_gap',i)
  if r['capture_time']<t:flag('capture_before_sample',i)
  if not np.isfinite(q[i]).all() or not np.isfinite(a[i]).all():flag('nonfinite_label',i)
  if q.shape[1]!=7 or a.shape[1]!=7:flag('label_shape',i)
  if np.any(a[i]<limits[:,0]-1e-6) or np.any(a[i]>limits[:,1]+1e-6):flag('action_joint_limit',i)
  # Measured dynamics may have small boundary overshoot; track >0.01 rad separately.
  if np.any(q[i]<limits[:,0]-.01) or np.any(q[i]>limits[:,1]+.01):flag('state_joint_limit_gt_0.01rad',i)
  current_hash={}
  for c in CAMS:
   if c in prev and r[c+'_image_stamp']<=prev[c]:flag('reused_image_stamp',i)
   prev[c]=r[c+'_image_stamp'];rel=Path(r['observation.images.'+c]);path=p/rel
   if not path.resolve().is_relative_to(p.resolve()):flag('image_path_escape',i);continue
   try:blob=path.read_bytes()
   except OSError:flag('missing_image',i);continue
   img=cv2.imdecode(np.frombuffer(blob,dtype=np.uint8),cv2.IMREAD_COLOR)
   if img is None:flag('undecodable_image',i);continue
   if list(img.shape)!=INFO['image_shapes'][c]:flag('image_shape',i)
   digest=hashlib.sha256(blob).hexdigest();current_hash[c]=digest
   mean=float(img.mean());std=float(img.std());means[c].append(mean)
   if mean<2 and std<2:dark[c]+=1
   if hashes.get(c)==digest:
    repeat[c]+=1;runrepeat[c]+=1
    if i and np.max(np.abs(q[i,:6]-q[i-1,:6]))>.005:repeat_moving[c]+=1
   else:runrepeat[c]=0
   maxrepeat[c]=max(maxrepeat[c],runrepeat[c]);hashes[c]=digest
  if len(set(current_hash.values()))<len(current_hash):flag('identical_camera_payloads',i)
 if n!=summary['frame_count']:flag('summary_frame_count',n)
 for key in ('task_success','physical_grasp_ok','bilateral_contact_ever','object_lifted_clear_of_table_ever','release_sequence_complete','placement_stable'):
  if summary.get(key) is not True:flag('missing_success_evidence_'+key,n)
 if summary.get('dropped_frames',0):flag('recording_dropped_frames',n)
 closing=np.flatnonzero(a[:,-1]>=-.75);c=int(closing[0]) if len(closing) else None
 result=dict(episode=p.name,frames=n,start_unix=start['started_at_unix_s'],grid=summary['scene_randomization']['target_grid'],seed=summary['scene_randomization']['seed'],violations=dict(violations),examples=examples,
  camera_spread_s=stats(spreads),state_bracket_s=stats(state_gaps),action_age_s=stats(ages),capture_delay_s=stats(latencies),action_state_arm_error_rad=stats(np.max(np.abs(a[:,:6]-q[:,:6]),axis=1)),action_state_gripper_error_rad=stats(np.abs(a[:,-1]-q[:,-1])),action_arm_step_rad=stats(np.max(np.abs(np.diff(a[:,:6],axis=0)),axis=1)),dark_frames=dict(dark),consecutive_identical_images=dict(repeat),identical_images_while_arm_moves=dict(repeat_moving),longest_identical_image_run=maxrepeat,mean_brightness={k:stats(v) for k,v in means.items()},phase_counts=dict(collections.Counter(r['phase'] for r in rows)),first_close_frame=c,trajectory_sha256=hashlib.sha256(np.concatenate([q,a],axis=1).tobytes()).hexdigest())
 if c is not None:
  cache={}
  def pose(i):
   if i not in cache:cache[i]=fk(q[i,:6])
   return cache[i]
  pc,rc=pose(c)
  waits={}
  for mm,deg in [(1,.5),(2,1),(5,2)]:
   k=c
   while k>0:
    pk,rk=pose(k-1)
    if np.linalg.norm(pk-pc)>mm/1000 or R.from_matrix(rk@rc.T).magnitude()>np.deg2rad(deg):break
    k-=1
   waits[f'{mm}mm_{deg}deg_s']=(c-k)/10
  result['preclose_stationary_s']=waits;result['close_roll_actual_deg']=roll(rc);result['close_roll_command_deg']=roll(fk(a[c,:6])[1])
  result['close_pitch_actual_deg']=float(np.rad2deg(np.arcsin(rc[2,0])))
  result['close_action_arm_error_rad']=float(np.max(np.abs(a[c,:6]-q[c,:6])))
  result['first_close_timestamp']=rows[c]['simulation_time'];result['close_camera_paths']={cam:str(p/rows[c]['observation.images.'+cam]) for cam in CAMS}
 else:flag('no_close_command',n);result['violations']=dict(violations)
 return result

def main():
 paths=sorted(ROOT.glob('episode_[0-9][0-9][0-9][0-9][0-9][0-9]'));results=[];started=time.time()
 with ThreadPoolExecutor(max_workers=8) as pool:
  futures={pool.submit(audit_episode,p):p for p in paths}
  for f in as_completed(futures):
   try:r=f.result()
   except Exception as e:r=dict(episode=futures[f].name,audit_error=repr(e));print('ERROR',r,flush=True)
   results.append(r)
   (OUT/'progress.json').write_text(json.dumps(dict(done=len(results),total=len(paths),elapsed_s=time.time()-started)))
   if len(results)%10==0:print('AUDITED',len(results),'/',len(paths),flush=True)
 results.sort(key=lambda r:r['episode'])
 (OUT/'episodes.json').write_text(json.dumps(results,indent=2))
 aggregate=collections.Counter()
 for r in results:aggregate.update(r.get('violations',{}))
 seeds=[r['seed'] for r in results if 'seed' in r];trajectories=[r['trajectory_sha256'] for r in results if 'trajectory_sha256' in r]
 summary=dict(episodes=len(results),frames=sum(r.get('frames',0) for r in results),images=sum(r.get('frames',0)*3 for r in results),violations=dict(aggregate),audit_errors=[r for r in results if 'audit_error'in r],duplicate_seeds=len(seeds)-len(set(seeds)),duplicate_trajectories=len(trajectories)-len(set(trajectories)),roll_actual_deg=stats([r['close_roll_actual_deg'] for r in results if 'close_roll_actual_deg'in r]),preclose_stationary_s={key:stats([r['preclose_stationary_s'][key] for r in results if 'preclose_stationary_s'in r]) for key in ['1mm_0.5deg_s','2mm_1deg_s','5mm_2deg_s']},roll_groups={})
 for label,pred in [('negative90',lambda x:abs(x+90)<20),('positive90',lambda x:abs(x-90)<20),('other',lambda x:abs(x+90)>=20 and abs(x-90)>=20)]:
  group=[r for r in results if 'close_roll_actual_deg'in r and pred(r['close_roll_actual_deg'])]
  summary['roll_groups'][label]=dict(count=len(group),episodes=[r['episode'] for r in group],wait_2mm_1deg_s=stats([r['preclose_stationary_s']['2mm_1deg_s'] for r in group]))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
