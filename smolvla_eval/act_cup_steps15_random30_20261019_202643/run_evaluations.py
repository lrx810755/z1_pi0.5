"""Thirty fresh random cup ACT trials; preserve complete raw and final-box results."""
import subprocess,json,datetime,time,os,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CHECKPOINT='/home/lp531/smolvla_training/act_cup_gentle_cached_chunk16_20261019/checkpoints/050000/pretrained_model'
ENTRY='/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_sim_eval.sh'

def save(name,value):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)

def results():
 r=[(p,json.loads(p.read_text())) for p in (ROOT/'steps15').glob('act_sim_eval_*/episodes/*/result.json')]
 return sorted(r,key=lambda x:x[1]['episode_number'])

try:
 cmd=['bash',ENTRY,'--policy-type','act','--checkpoint',CHECKPOINT,'--objects','cup',
      '--target-xy-bounds','0.36','0.69','-0.27','0.074','--execute-steps','15',
      '--rate','10','--execution-rate','60','--gripper-min-close-hold-s','1',
      '--binary-gripper','--gripper-action-encoding','binary','--no-freeze-target-until-close',
      '--training-dataset-root','/home/lp531/smolvla_data/z1_tabletop_raw_cup_standard_three_camera_20261019',
      '--max-training-target-distance-m','0.03','--reject-blank-cameras',
      '--wrist-rgb-topic','/left_wrist_camera/color/image_raw',
      '--prelift-contact-stable-s','0','--prelift-contact-timeout-s','2.0',
      '--episodes-per-object','30','--timeout','60','--output',str(ROOT/'steps15')]
 save('command.json',cmd)
 save('experiment.json',dict(checkpoint=CHECKPOINT,target='cup',episodes=30,execute_steps=15,action_rate_hz=10,execution_rate_hz=60,
      minimum_close_hold_s=1,bounds=[.36,.69,-.27,.074],seed_mode='fresh random simulator seeds, not demonstration seeds',
      success_criterion='final box volume x/y +/-0.055m about (0.32,0.14), root z [0,0.12], released and not held',
      original_evaluator_results_preserved=True,policy_sha256=hashlib.sha256((Path(CHECKPOINT)/'model.safetensors').read_bytes()).hexdigest()))
 with (ROOT/'steps15.log').open('a',buffering=1) as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
  save('status.json',dict(status='running',pid=p.pid,total=30,completed=0,execute_steps=15,started=datetime.datetime.now().isoformat()))
  while p.poll() is None:
   done=results()
   bad=[str(path) for path,r in done if r.get('error_detail') or r.get('cleanup_error')]
   if bad:
    p.terminate();p.wait(timeout=30)
    raise RuntimeError('Evaluation chain errors; stopped without counting an invalid batch: '+str(bad))
   save('status.json',dict(status='running',pid=p.pid,total=30,completed=len(done),execute_steps=15,updated=datetime.datetime.now().isoformat()))
   time.sleep(3)
  code=p.returncode
 done=results();reports=list((ROOT/'steps15').glob('act_sim_eval_*/report.json'))
 if code or len(done)!=30 or not reports:raise RuntimeError(f'Incomplete evaluation: exit={code}, saved={len(done)}')
 reviewed=[];health=[]
 for path,r in done:
  assert r['target_object']=='cup' and r['execute_steps']==15 and not r.get('error_detail')
  traces=[json.loads(x) for x in (path.parent/'trace.jsonl').read_text().splitlines()];assert traces
  assert (path.parent/'prediction_chunks.jsonl').stat().st_size
  deltas=[]
  for row in traces:
   c=row.get('contact_status',{})
   if c.get('gripper_tf_stamp_s') is not None and c.get('object_pose_stamp_s') is not None:
    deltas.append(abs(c['gripper_tf_stamp_s']-c['object_pose_stamp_s']))
  assert deltas and max(deltas)<=.100001,(path,max(deltas,default=-1))
  for name in ['wrist.mp4','global_third_person.mp4','global_right.mp4']:
   assert (path.parent/'videos'/name).stat().st_size>0
  health.append(dict(episode=r['episode_number'],trace_rows=len(traces),max_tf_object_delta_s=max(deltas)))
  s=r['final_status'];x,y,z=s['object_position_m'];inside=abs(x-.32)<=.055 and abs(y-.14)<=.055 and 0<=z<=.12
  success=bool(inside and s.get('released') and not s.get('holding_observed'))
  reviewed.append(dict(episode=r['episode_number'],seed=r['seed'],success=success,inside_box=inside,final_position_m=[x,y,z],
       released=s.get('released'),holding=s.get('holding_observed'),legacy_success=r['success'],legacy_failure_category=r['failure_category'],source_result=str(path)))
 seeds=[r['seed'] for _,r in done];assert len(set(seeds))==30
 save('random_seeds.json',seeds);save('trace_video_tf_verified.json',health)
 total=sum(r['success'] for r in reviewed)
 summary=dict(completed=True,total=30,execute_steps=15,successes=total,success_rate=total/30,episodes=reviewed,finished=datetime.datetime.now().isoformat())
 save('final_box_results.json',summary);save('completed.json',summary)
 save('status.json',dict(status='complete',total=30,successes=total,success_rate=total/30))
except Exception as e:
 save('status.json',dict(status='failed',error=str(e),updated=datetime.datetime.now().isoformat()));raise
