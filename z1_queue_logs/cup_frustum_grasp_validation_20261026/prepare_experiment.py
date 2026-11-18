from pathlib import Path
import json,shutil,os
R=Path(__file__).parent;SRC=Path('/home/lp531/ros2_ws/src/z1_ros2');cfg=json.loads((R/'original_grasp_physics.json').read_text());cfg['object_grasp_profiles']['cup']['mirror_wrist_roll_candidates']=False;(R/'grasp_physics.json').write_text(json.dumps(cfg,indent=2))
share=R/'share/config';share.mkdir(parents=True,exist_ok=True)
for p in (SRC/'z1_isaac_bridge/config').iterdir():
 dest=share/p.name
 if dest.exists():continue
 dest.symlink_to(R/'grasp_physics.json' if p.name=='grasp_physics.json' else p)
s=Path('/home/lp531/smolvla_eval/act_cup_only_renderfix_steps15_paired10_20261026/run_cup_only_scene.py').read_text();s='\n'.join(l for l in s.split('\n') if not l.startswith('CUP_ONLY_POSES ='))
start=s.index('    def _sample_randomized_object_poses(');end=s.index('    def _apply_scene_randomization(',start)
s=s[:start]+'''    def _sample_randomized_object_poses(self, seed, level="easy"):
        case = json.loads(Path("'''+str(R/'current_case.json')+'''").read_text())
        return {"cup": (np.asarray(case['cup_position_m'], dtype=np.float32),
                        np.asarray([1.,0.,0.,0.], dtype=np.float32))}

'''+s[end:];(R/'scene.py').write_text(s)
source=Path('/home/lp531/smolvla_eval/act_cup_only_renderfix_steps15_paired10_20261026/ros_overlay');shutil.copytree(source,R/'ros_overlay',dirs_exist_ok=True)
(R/'current_case.json').write_text(json.dumps(dict(id='pilot_nominal',cup_position_m=[.52,-.10,0],target_grid=[2,1],relative_target_overrides={})))
print(R)
