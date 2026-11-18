"""Select ten training demonstrations before observing any diagnostic outcomes."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
RUN=Path('/home/lp531/pi05_lora_workspace/act_cup_dynamic_cached_20261020')
CK=Path('/home/lp531/smolvla_training/act_cup_dynamic_gentle_cached_chunk16_20261020/checkpoints/050000/pretrained_model')
CELLS=[(0,0),(2,0),(4,0),(0,1),(3,1),(1,2),(4,2),(0,3),(1,3),(4,3)]
def prepare():
 manifest=json.loads((RUN/'split_manifest.json').read_text());chosen=[]
 for cell in CELLS:
  record=next(r for r in manifest['source_records'] if r['source']=='original' and r['split']=='train' and tuple(r['grid'])==cell)
  path=Path(record['source_path']);summary=json.loads((path/'episode_summary.json').read_text());row=json.loads(next((path/'frames.jsonl').open()))
  assert summary['task_success'] and summary['physical_grasp_ok']
  assert hashlib.sha256((path/'frames.jsonl').read_bytes()).hexdigest()==record['source_frames_sha256']
  chosen.append(dict(order=len(chosen)+1,training_episode=record['episode_index'],source_path=str(path),seed=record['seed'],grid=list(cell),object_positions_m=summary['scene_randomization']['object_positions_m'],object_orientations_xyzw=summary['scene_randomization']['object_orientations_xyzw'],initial_joint_state=row['observation.state'],frames_sha256=record['source_frames_sha256']))
 assert len(chosen)==len({c['seed'] for c in chosen})==10
 baseline=json.loads(Path('/home/lp531/smolvla_eval/act_cup_dynamic_steps15_random50_20261020/command_batch_01.json').read_text())
 cmd=baseline[:];cmd[1]=str(R/'eval_entry.py');cmd[cmd.index('--episodes-per-object')+1]='10';cmd[cmd.index('--output')+1]=str(R/'rollouts');cmd+=['--eval-seeds',*[str(c['seed']) for c in chosen]]
 info=dict(diagnostic_only=True,count=10,checkpoint=str(CK),policy_sha256=hashlib.sha256((CK/'model.safetensors').read_bytes()).hexdigest(),execute_steps=15,selection_rule='First manifest training original in each of ten preselected cells; no outcome-based selection; excludes heldout (1,1),(2,3)',position_tolerance_m=.002,orientation_tolerance_deg=2.,initial_joint_tolerance_rad=.025,scenes=chosen)
 (R/'selected_scenes.json').write_text(json.dumps(info,indent=2));(R/'command.json').write_text(json.dumps(cmd,indent=2));return info
if __name__=='__main__':
 d=prepare();print(json.dumps(dict(count=d['count'],seeds=[s['seed'] for s in d['scenes']],cells=[s['grid'] for s in d['scenes']]),indent=2))
