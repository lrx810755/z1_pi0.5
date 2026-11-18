from pathlib import Path
import json, math, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
WORK=Path('/home/lp531/pi05_lora_workspace')
cfg=json.loads(Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json').read_text())['object_grasp_profiles']['beverage_bottle']
def inside(v,e):
 return e['x_min_m']<=v[0]<=e['x_max_m'] and abs(v[1])<=e['y_abs_max_m'] and e['z_min_m']<=v[2]<=e['z_max_m']
def duration(mask,t):
 dt=np.r_[np.diff(t),0];return float(dt[mask].sum())
def longest(mask,t):
 best=0;start=None
 for i,x in enumerate(mask):
  if x and start is None:start=i
  if start is not None and (not x or i==len(mask)-1):
   best=max(best,float(t[i]-t[start]));start=None
 return best
metrics=[]
for f in sorted(ROOT.glob('*_0*/**/trace.jsonl'))+sorted(ROOT.glob('new_120000/**/trace.jsonl')):
 rows=[json.loads(x) for x in f.read_text().splitlines()];result=json.loads((f.parent/'result.json').read_text())
 t=np.array([r['simulation_elapsed_s'] for r in rows]);raw=np.array([r['policy_action'][-1] for r in rows]);target=np.array([r['execution_target'][-1] for r in rows]);joint=np.array([r['joint_state'][-1] for r in rows]);pos=np.array([r['contact_status']['object_position_m'] for r in rows]);rel=np.array([r['contact_status']['relative_position_gripper_m'] for r in rows]);dist=np.linalg.norm(rel,axis=1)
 closed=target>-.45;pre=~np.maximum.accumulate(closed);bilat=np.array([r['contact_status']['left_finger_contact_object']=='beverage_bottle' and r['contact_status']['right_finger_contact_object']=='beverage_bottle' for r in rows]);env=np.array([inside(v,cfg['preclose_envelope']) for v in rel]);loaded=np.array([inside(v,cfg['loaded_envelope']) for v in rel]);i=int(dist.argmin());starts=np.flatnonzero(closed & ~np.r_[False,closed[:-1]])
 # Prior to any bottle movement greater than 1cm, proxy for undisturbed approach only.
 moved=np.linalg.norm(pos-pos[0],axis=1)>.01;undisturbed=~np.maximum.accumulate(moved)
 actions=[r for r in rows if r['execution_substep']==0]
 d=dict(model=f.relative_to(ROOT).parts[0],seed=result['seed'],success=result['success'],path=str(f.parent),rows=len(rows),raw_min=float(raw.min()),raw_max=float(raw.max()),threshold_mismatches=int(np.count_nonzero((raw>=.5)!=closed)),close_attempts=len(starts),
  min_distance_m=float(dist[i]),min_distance_time=float(t[i]),relative_at_min=rel[i].tolist(),raw_at_min=float(raw[i]),
  preclose_envelope_before_first_close_s=duration(env&pre,t),preclose_envelope_before_first_close_longest_s=longest(env&pre,t),
  preclose_envelope_undisturbed_s=duration(env&pre&undisturbed,t),loaded_envelope_before_first_close_s=duration(loaded&pre,t),
  bilateral_before_first_close_s=duration(bilat&pre,t),bilateral_before_first_close_longest_s=longest(bilat&pre,t),
  raw_max_inside_preclose=float(raw[env&pre].max()) if (env&pre).any() else None,
  raw_max_during_bilateral_open=float(raw[bilat&pre].max()) if (bilat&pre).any() else None,
  close_events=[dict(time=float(t[j]),raw=float(raw[j]),relative=rel[j].tolist(),distance_m=float(dist[j]),joint_before=float(joint[j]),actual_joint_after_1s=float(joint[min(int(np.searchsorted(t,t[j]+1)),len(t)-1)])) for j in starts],
  max_arm_command_tracking_error_rad=float(max(max(abs(a-b) for a,b in zip(r['command'][:6],r['joint_state'][:6])) for r in rows)),
  final_object_position=pos[-1].tolist(),final_joint_gripper=float(joint[-1]),final_raw_gripper=float(raw[-1]),final_phase=rows[-1]['phase'],
  stationary_final_10s_arm_range_rad=float(np.ptp(np.array([r['joint_state'][:6] for r in rows])[t>=t[-1]-10],axis=0).max()))
 metrics.append(d)
assert len(metrics)==32,len(metrics)
# Audit normalized sampling probabilities over identical training anchor frames.
old=WORK/'gentle_correction_base_20261010';new=WORK/'grip_stage_base_20261011'
a=np.load(old/'sampling_arrays.npz');b=np.load(new/'sampling_arrays.npz');records=json.loads((new/'phase_manifest.json').read_text());ep=a['episode'];held=[r['episode_index'] for r in records if r['split']=='validation'];train=~np.isin(ep,held);corr=a['correction'];assert np.array_equal(ep,b['episode'])
def norm(w):
 p=np.zeros(len(w))
 for flag,mass in [(False,.7),(True,.3)]:
  mask=train&(corr==flag);p[mask]=w[mask]*mass/w[mask].sum()
 return p
pold=norm(np.where(a['key'],2.,1.));pnew=norm(b['phase_weight']);masks={k:np.zeros(len(ep),bool) for k in ['preclose_1s','close_transition_pm05s','close_to_stable','hold_after_stable_before_release','release_transition_pm05s','horizon_contains_close_transition','horizon_contains_open_transition','arm_near_static_close_to_stable']}
raw_checks=[];durations=[]
for r in records:
 ids=np.flatnonzero(ep==r['episode_index']);n=len(ids);idx=np.arange(n);c=r['close_frame'];s=r['stable_lift_confirmed_frame'];o=r['open_frame']
 source=Path(r['source_path'])/'frames.jsonl';rows=[json.loads(x) for x in source.read_text().splitlines()];assert len(rows)==n
 assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_frames_sha256']
 states=np.array([x['observation.state'] for x in rows]);cmd=np.array([x['action'] for x in rows]);delta=np.r_[np.inf,np.max(np.abs(np.diff(cmd[:,:6],axis=0)),axis=1)]
 local={'preclose_1s':(idx>=c-10)&(idx<c),'close_transition_pm05s':(idx>=c-5)&(idx<c+5),'close_to_stable':(idx>=c)&(idx<=s),'hold_after_stable_before_release':(idx>s)&(idx<o),'release_transition_pm05s':(idx>=o-5)&(idx<o+5),'horizon_contains_close_transition':(idx<c)&(idx+14>=c),'horizon_contains_open_transition':(idx<o)&(idx+14>=o),'arm_near_static_close_to_stable':(idx>=c)&(idx<=s)&(delta<.001)}
 for k,v in local.items():masks[k][ids]=v
 raw_checks.append(dict(episode=r['episode_index'],split=r['split'],close_frame=c,close_phase=rows[c].get('phase'),joint_gripper_at_close=float(states[c,-1]),command_gripper_at_close=float(cmd[c,-1]),actual_joint_1s_after=float(states[min(c+10,n-1),-1]),open_frame=o,open_phase=rows[o].get('phase'),command_at_open=float(cmd[o,-1]),stable_delay_s=(s-c)/10))
 if r['split']=='train':durations.append((s-c)/10)
mass={k:dict(old_mass=float(pold[v].sum()),new_mass=float(pnew[v].sum()),frames=int((v&train).sum())) for k,v in masks.items()}
report=dict(episodes=metrics,sampling_mass=mass,source_hash_checks_passed=True,train_episodes=len(durations),median_close_stable_delay_s=float(np.median(durations)),p90_close_stable_delay_s=float(np.percentile(durations,90)),max_close_stable_delay_s=max(durations),raw_label_boundaries=raw_checks,limitations=['Recorded trace omits bottle orientation, tilt, inside_grasp_envelope and object_touching_table. Position envelope is diagnostic only; contact does not prove viable grasp.','Only executed action prefixes are logged; discarded chunk tail predictions cannot be checked.','Normalized expected sampling mass, not actual draw counts.','Old and new learning-rate schedules differ.'])
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('episodes',len(metrics),'threshold mismatches',sum(d['threshold_mismatches'] for d in metrics))
for d in metrics:
 print(d['model'],d['seed'],'close',d['close_attempts'],'rawmax',round(d['raw_max'],3),'mindist',round(d['min_distance_m'],3),'env_s',round(d['preclose_envelope_before_first_close_s'],2),'undist_s',round(d['preclose_envelope_undisturbed_s'],2),'bilat_open_s',round(d['bilateral_before_first_close_s'],2),'last10range',round(d['stationary_final_10s_arm_range_rad'],4))
print(json.dumps(mass,indent=2))
