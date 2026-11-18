import json,subprocess,datetime,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
models=[('new_120000','pi05_grip_stage_base_20261011/checkpoints/120000/pretrained_model',[276137783,1174105379]),('old_080000','pi05_gentle_correction_base_20261010/checkpoints/080000/pretrained_model',[276137783])]
manifest={'started_at':datetime.datetime.now().astimezone().isoformat(),'inference_seed':20261011,'execute_steps':10,'chunk_size':15,'timeout_sim_s':60,'models':[]}
for name,relative,seeds in models:
 checkpoint=Path('/home/lp531/smolvla_training')/relative
 manifest['models'].append(dict(name=name,checkpoint=str(checkpoint),seeds=seeds,adapter_sha256=hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest()))
(P/'manifest.json').write_text(json.dumps(manifest,indent=2))
for m in manifest['models']:
 name=m['name'];output=P/name
 if output.exists():raise RuntimeError('Refuse overwrite '+str(output))
 cmd=['bash',str(P/'run_diagnostic.sh'),'--checkpoint',m['checkpoint'],'--policy-type','pi05','--objects','beverage_bottle','--episodes-per-object',str(len(m['seeds'])),'--eval-seeds',*[str(s) for s in m['seeds']], '--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--training-dataset-root','/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1','--max-training-target-distance-m','0.03','--reject-blank-cameras','--rate','10','--execution-rate','60','--execute-steps','10','--timeout','60','--prelift-contact-stable-s','0','--prelift-contact-timeout-s','2.0','--binary-gripper','--gripper-action-encoding','binary','--no-freeze-target-until-close','--preview-episodes','8','--output',str(output)]
 print(datetime.datetime.now().astimezone().isoformat(),'START',name,flush=True)
 with (P/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{name} exit {r.returncode}')
 results=list(output.glob('*/episodes/*/result.json'));assert len(results)==len(m['seeds'])
 print(datetime.datetime.now().astimezone().isoformat(),'DONE',name,flush=True)
(P/'completed.json').write_text(json.dumps({'completed_at':datetime.datetime.now().astimezone().isoformat(),'episodes':3},indent=2))
