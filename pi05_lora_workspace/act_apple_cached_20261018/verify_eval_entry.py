import json,shlex,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from z1_isaac_bridge import smolvla_sim_rollout as rollout
W=Path(__file__).resolve().parent
ENTRY='/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh'
checks=[]
for model,extra in [('12',[]),('act-apple',['--execute-steps','15']),('9',[]),('10',[]),('11',[])]:
 r=subprocess.run(['bash',ENTRY,'--model',model,'--dry-run']+extra,capture_output=True,text=True,check=True)
 args=rollout.parse_args(shlex.split(r.stdout.splitlines()[-1])[1:])
 if model in ['12','act-apple']:
  assert args.objects==['apple'] and args.target_xy_bounds==(.36,.624,-.27,.16)
  assert args.checkpoint.name=='pretrained_model' and 'act_apple_' in str(args.checkpoint)
  assert args.gripper_action_encoding=='binary' and args.rate==10 and args.execution_rate==60
  assert args.execute_steps==(15 if extra else 8)
 else:assert args.objects==['beverage_bottle']
 checks.append({'model':model,'objects':args.objects,'execute_steps':args.execute_steps})
cp=Path('/home/lp531/smolvla_training/act_apple_gentle_cached_chunk16_20261018/checkpoints/050000/pretrained_model')
node=SimpleNamespace(args=SimpleNamespace(policy_type='act',execute_steps=8),get_logger=lambda:SimpleNamespace(info=print))
policy=rollout.SmolVLASimRollout._load_policy(node,cp);node.policy=policy
predictions=[]
try:
 records=json.loads((W/'split_manifest.json').read_text())['source_records']
 for source in ['original','correction']:
  r=next(x for x in records if x['source']==source and x['split']=='validation')
  root=Path(r['source_path']);rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()];row=rows[r['close_frame']]
  batch={'observation.state':torch.tensor([row['observation.state']],dtype=torch.float32),'task':[row['task']]}
  for name in node.camera_keys:
   cam='observation.images.'+name
   with Image.open(root/row[cam]) as im:a=np.array(im.convert('RGB'))
   batch[cam]=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255
  policy.reset();out=rollout.SmolVLASimRollout._predict(node,batch)
  assert out.shape==(8,7) and np.isfinite(out).all()
  predictions.append(dict(source=source,episode=r['episode_index'],shape=list(out.shape),finite=True))
finally:policy.close()
report=dict(checkpoint=str(cp),parser_checks=checks,real_image_predictions=predictions,camera_keys=node.camera_keys,state_dim=node.state_dim,rollout_module=rollout.__file__,robot_commands_sent=False)
(W/'eval_integration_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
