"""Prepare paired30 experiment without changing old datasets, models or scenes."""
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path
import shutil

R=Path(__file__).resolve().parent
OLD=Path('/home/lp531/smolvla_eval/pi05_cup_recovery513_final30k_steps15_paired30_20261030')
SOURCE=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py')
BASE_SCENE=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd')
SCENE=R/'scene_mass300g.usda'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def replace(text,old,new):
    assert old in text,old[:100]
    return text.replace(old,new)


def main():
    assert SCENE.exists(),'Create the scoped USD mass layer first'
    assert not (R/'supervisor_status.json').exists(),'Existing experiment; inspect before restart'
    for name in ['runtime_rollout.py','eval_entry.py','workflow_core.py','selected_checkpoint.json','excluded_seed_baseline.json','paired_seeds.json']:
        shutil.copyfile(OLD/name,R/name)
    physics_spec=dict(cup_mass_kg=.30,cup_table_static_friction=.40,cup_table_dynamic_friction=.30,
        baseline_cup_mass_kg=.08,baseline_cup_table_static_friction=.20,baseline_cup_table_dynamic_friction=.15,
        gripper_static_friction=6.,gripper_dynamic_friction=5.,drive_parameters='unchanged',
        selected_model=json.loads((R/'selected_checkpoint.json').read_text()),execute_steps=15,
        action_rate_hz=10,execution_rate_hz=60,total=30,baseline=str(OLD))
    (R/'physics_spec.json').write_text(json.dumps(physics_spec,indent=2))
    original=SOURCE.read_text()
    # Local source copy imports all its original helper modules from the original asset directory.
    local=replace(original,'from layout_sweep import layout_sweep_blockers',
        f'sys.path.insert(0,{str(SOURCE.parent)!r})\nfrom layout_sweep import layout_sweep_blockers')
    local=replace(local,'    configure_grasp_friction(stage)\n',
        f'    configure_grasp_friction(stage)\n    sys.path.insert(0,{str(R)!r})\n    import cup_physics_override\n    cup_physics_override.apply(stage)\n')
    local=replace(local,'    training_objects = initialize_training_objects()\n',
        '    training_objects = initialize_training_objects()\n    cup_physics_override.verify(stage,training_objects,robot,DRIVE_GAINS,GRIPPER_STATOR)\n')
    (R/'run_scene_mass300g.py').write_text(local)
    (R/'scene_runtime_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),local.splitlines(True),fromfile=str(SOURCE),tofile='run_scene_mass300g.py')))
    runner=(OLD/'run_evaluations.py').read_text()
    runner=replace(runner,str(BASE_SCENE),str(SCENE))
    runner=replace(runner,"seed_mode='exact30 seeds and order from previous12-step evaluation, excluding training513 seeds'",
        "seed_mode='same30 paired seeds as80g15-step baseline, excluding training513 seeds'")
    runner=replace(runner,'chunk_size=30,n_action_steps=15)',
        "chunk_size=30,n_action_steps=15,cup_mass_kg=.30,cup_table_static_friction=.40,cup_table_dynamic_friction=.30,baseline_cup_mass_kg=.08)")
    (R/'run_evaluations.py').write_text(runner)
    supervisor=(OLD/'supervise.py').read_text().replace('PORT=8817','PORT=8818')
    supervisor=replace(supervisor,'scene:='+str(BASE_SCENE),'scene:='+str(SCENE))
    supervisor=replace(supervisor,'isaac_script:='+str(SOURCE),'isaac_script:='+str(R/'run_scene_mass300g.py'))
    supervisor=replace(supervisor,"   eval_env=os.environ.copy();eval_env.pop('PYTHONNOUSERSITE',None)",
        "   physics=json.loads((R/'physics_verified.json').read_text())\n   assert physics['passed'] and abs(physics['actual_cup_mass_kg']-.30)<1e-6\n   assert physics['effective_cup_table_friction']==dict(static=.40,dynamic=.30)\n   eval_env=os.environ.copy();eval_env.pop('PYTHONNOUSERSITE',None)")
    (R/'supervise.py').write_text(supervisor)
    (R/'start.sh').write_text((OLD/'start.sh').read_text().replace('ROS_DOMAIN_ID=116','ROS_DOMAIN_ID=117'))
    report=(OLD/'summarize.py').read_text()
    report=replace(report,"previous_root = Path('/home/lp531/smolvla_eval/pi05_cup_recovery513_final30k_steps12_random30_20261030')",f'previous_root = Path({str(OLD)!r})')
    report=replace(report,"    rows = []\n", "    physics=json.loads((R/'physics_verified.json').read_text())\n    assert physics['passed'] and abs(physics['actual_cup_mass_kg']-.30)<1e-6\n    rows = []\n")
    report=replace(report,'                  observed_categories=counts, episodes=rows,',
        '                  physics=physics,observed_categories=counts, episodes=rows,')
    report=re.sub(r'\bsteps12_', 'baseline80g_', report)
    report=re.sub(r'\bsteps15_', 'mass300g_friction2_', report)
    report=replace(report,'# PI0.5杯子LoRA最终模型：执行15步、30次独立验证','# PI0.5杯子300克、桌面摩擦2倍：15步、30次配对评估')
    report=replace(report,'沿用上一轮12步评估的30个种子及顺序，均未进入训练集','杯子300克、静/动摩擦0.40/0.30；沿用80克15步基线的30个种子及顺序，均未进入训练集')
    report=replace(report,'使用与12步评估相同的30个未参与训练布局','使用与80克15步基线相同的30个未参与训练布局')
    report=replace(report,'上一轮12步：最终入盒7/30，完整流程4/30','原80克15步基线：最终入盒11/30，完整流程6/30')
    # Add the physical mechanism metrics relevant to this comparison, without changing success criteria.
    report=replace(report,"        initial = record['initial_scene_randomization']['object_positions_m']",
        "        first_close=next((i for i,t in enumerate(trace) if t['execution_target'][-1]>-.2),len(trace)-1)\n        preclose_tilt=max(t['contact_status'].get('cylinder_tilt_deg') or 0 for t in trace[:first_close+1])\n        initial = record['initial_scene_randomization']['object_positions_m']")
    report=replace(report,"                   trace_rows=len(trace), initial_cup_position_m=initial['cup'])",
        "                   trace_rows=len(trace), initial_cup_position_m=initial['cup'],preclose_max_tilt_deg=preclose_tilt)")
    report=replace(report,'                  physics=physics,observed_categories=counts, episodes=rows,',
        "                  physics=physics,preclose_tilt_gt60_episodes=[x['episode'] for x in rows if x['preclose_max_tilt_deg']>60],\n                  never_bilateral_episodes=[x['episode'] for x in rows if not x['bilateral_cup_contact_observed']],observed_categories=counts, episodes=rows,")
    (R/'summarize.py').write_text(report)
    for file in R.glob('*.py'):ast.parse(file.read_text(),filename=str(file))
    old_snapshot=json.loads((OLD/'source_snapshot.json').read_text())
    files=[Path(p) for p in old_snapshot if not Path(p).is_relative_to(OLD)]
    files+=list(R.glob('*.py'))+[R/'start.sh',SCENE,R/'physics_spec.json',R/'selected_checkpoint.json',R/'paired_seeds.json',R/'excluded_seed_baseline.json',OLD/'completed.json']
    (R/'source_snapshot.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2))
    print(json.dumps(physics_spec,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
