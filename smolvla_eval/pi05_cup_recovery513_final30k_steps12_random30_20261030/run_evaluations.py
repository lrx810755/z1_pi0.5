"""Thirty fresh cup PI05 trials at10Hz, executing12 steps per prediction."""
import datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
RUN=Path('/home/lp531/pi05_lora_workspace/pi05_cup_recovery513_lora_20261029')
sys.path.insert(0,str(Path(__file__).resolve().parent))
from workflow_core import classify_error,in_box_released
R=Path(__file__).resolve().parent
CHECKPOINT=Path('/home/lp531/smolvla_training/pi05_cup_recovery513_lora_20261029/checkpoints/240000/pretrained_model')
RAW='/home/lp531/smolvla_data/z1_tabletop_raw_cup_basic_4x5x20_renderfix_20261026'
TOTAL=30

def save(name,data):
 p=R/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def state(**data):save('status.json',dict(updated=datetime.datetime.now().astimezone().isoformat(),total=TOTAL,**data))
def stop(p):
 if p.poll() is not None:return
 p.send_signal(signal.SIGINT)
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
def results(folder):
 files=list(folder.glob('pi05_sim_eval_*/episodes/*/result.json'))
 return sorted(files,key=lambda p:json.loads(p.read_text())['episode_number'])
def excluded():
 seeds=set()
 for source in [RUN,Path('/home/lp531/pi05_lora_workspace/act_cup_recovery_cached_20261020')]:
  for row in json.loads((source/'split_manifest.json').read_text())['source_records']:seeds.add(row['seed'])
 seeds.update(json.loads((R/'excluded_seed_baseline.json').read_text()))
 return seeds

def main():
 valid=[];invalid=[];used=excluded();initial_excluded=set(used)
 save('excluded_seeds.json',sorted(used))
 save('experiment.json',dict(checkpoint=str(CHECKPOINT),target='cup',episodes=TOTAL,execute_steps=12,action_rate_hz=10,execution_rate_hz=60,minimum_close_hold_s=1,bounds=[.36,.69,-.27,.074],seed_mode='30 fresh randomizations, excluding all513 source and previous evaluation seeds; resample only before trial for coverage/bounds',success_criterion='final cup root in x/y +/-0.055m about (0.32,0.14), z [0,0.12], released and not held',policy_sha256=hashlib.sha256((CHECKPOINT/'adapter_model.safetensors').read_bytes()).hexdigest(),checkpoint_selection='final PI05 LoRA:30000 optimizer updates,240000 microsteps,heldout loss0.0327',task_objects=['cup','apple','peach','mouse','beverage_bottle'],renderer_world_eps=0.0001,scene='/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd',chunk_size=30,n_action_steps=12))
 p=None
 try:
  for batch in range(1,11):
   remaining=TOTAL-len(valid)
   if remaining<=0:break
   save('excluded_seeds.json',sorted(used))
   folder=R/f'batch_{batch:02d}'
   if folder.exists():raise RuntimeError(f'Batch output already exists: {folder}')
   cmd=['/home/lp531/miniconda3/envs/unifolm-vla/bin/python',str(R/'eval_entry.py'),'--policy-type','pi05','--checkpoint',str(CHECKPOINT),'--objects','cup',
        '--target-xy-bounds','0.36','0.69','-0.27','0.074','--execute-steps','12','--rate','10','--execution-rate','60','--gripper-min-close-hold-s','1',
        '--binary-gripper','--gripper-action-encoding','binary','--no-freeze-target-until-close','--training-dataset-root',RAW,
        '--max-training-target-distance-m','0.03','--reject-blank-cameras','--wrist-rgb-topic','/left_wrist_camera/color/image_raw',
        '--prelift-contact-stable-s','0','--prelift-contact-timeout-s','2.0','--episodes-per-object',str(remaining),'--timeout','60','--output',str(folder)]
   save(f'command_batch_{batch:02d}.json',cmd)
   with (R/f'batch_{batch:02d}.log').open('w') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    while p.poll() is None:
     files=results(folder);good=0;bad=0
     for path in files:
      row=json.loads(path.read_text());kind=classify_error(row)
      if kind=='fatal':stop(p);raise RuntimeError(f'Evaluation chain error: {path}: {row.get("error_detail")}, {row.get("cleanup_error")}')
      if kind=='valid':good+=1
      else:bad+=1
     state(status='running',pid=p.pid,completed=len(valid)+good,invalid=len(invalid)+bad,batch=batch,execute_steps=12)
     time.sleep(3)
    if p.returncode:raise RuntimeError(f'Evaluator exited {p.returncode}')
   files=results(folder)
   if len(files)!=remaining:raise RuntimeError(f'Evaluator returned only {len(files)}/{remaining} trials')
   for path in files:
    row=json.loads(path.read_text());kind=classify_error(row)
    if kind=='fatal':raise RuntimeError(f'Evaluation chain error: {path}')
    (valid if kind=='valid' else invalid).append(path)
    if isinstance(row.get('seed'),int):used.add(row['seed'])
   save('valid_results.json',[str(p) for p in valid]);save('invalid_results.json',[str(p) for p in invalid])
  if len(valid)!=TOTAL:raise RuntimeError(f'Only {len(valid)} valid trials after ten batches')
  rows=[];health=[]
  for number,path in enumerate(valid,1):
   d=json.loads(path.read_text());s=d['final_status'];x,y,z=s['object_position_m']
   assert d['target_object']=='cup' and d['execute_steps']==12 and classify_error(d)=='valid'
   assert d['action_rate_hz']==10 and d['execution_rate_hz']==60
   videos={name:(path.parent/'videos'/name).stat().st_size for name in ['wrist.mp4','global_third_person.mp4','global_right.mp4']};assert min(videos.values())>0
   traces=[json.loads(line) for line in (path.parent/'trace.jsonl').read_text().splitlines()];assert traces
   assert (path.parent/'prediction_chunks.jsonl').stat().st_size and (path.parent/'input_sync.jsonl').stat().st_size
   deltas=[]
   for t in traces:
    c=t.get('contact_status',{})
    if c.get('gripper_tf_stamp_s') is not None and c.get('object_pose_stamp_s') is not None:deltas.append(abs(c['gripper_tf_stamp_s']-c['object_pose_stamp_s']))
   assert deltas and max(deltas)<=.100001,(path,max(deltas,default=-1))
   inside=abs(x-.32)<=.055 and abs(y-.14)<=.055 and 0<=z<=.12
   rows.append(dict(episode=number,source_episode=d['episode_number'],seed=d['seed'],success=in_box_released(s),inside_box=inside,final_position_m=[x,y,z],released=s.get('released'),holding=s.get('holding_observed'),legacy_success=d['success'],legacy_failure_category=d['failure_category'],source_result=str(path)))
   health.append(dict(episode=number,trace_rows=len(traces),max_tf_object_delta_s=max(deltas),videos=videos))
  all_seeds=[x['seed'] for x in rows];assert len(set(all_seeds))==TOTAL and not set(all_seeds)&initial_excluded
  save('random_seeds.json',all_seeds);save('trace_video_tf_verified.json',health)
  successes=sum(x['success'] for x in rows)
  summary=dict(completed=True,total=TOTAL,execute_steps=12,successes=successes,success_rate=successes/TOTAL,episodes=rows,excluded_invalid_trials=[str(p) for p in invalid],finished=datetime.datetime.now().astimezone().isoformat())
  save('final_box_results.json',summary);save('completed.json',summary)
  state(status='complete',completed=TOTAL,successes=successes,success_rate=successes/TOTAL,invalid=len(invalid))
 except BaseException as e:
  if p is not None:stop(p)
  state(status='failed',completed=len(valid),error=str(e));raise
if __name__=='__main__':main()
