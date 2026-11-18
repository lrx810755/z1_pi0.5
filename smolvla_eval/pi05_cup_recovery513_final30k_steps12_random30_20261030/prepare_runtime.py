"""Freeze a scoped cup PI05 evaluator from the existing validated ACT runner."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

R = Path(__file__).resolve().parent
OLD = Path('/home/lp531/smolvla_eval/act_cup_chunk30_batch64_best70k_steps15_8hz_paired30_20261028_102158')
TRAIN = Path('/home/lp531/pi05_lora_workspace/pi05_cup_recovery513_lora_20261029')
CP = Path('/home/lp531/smolvla_training/pi05_cup_recovery513_lora_20261029/checkpoints/240000/pretrained_model')
REPO = Path('/home/lp531/ros2_ws/src/z1_ros2')
PACKAGE = REPO/'z1_isaac_bridge/z1_isaac_bridge'


def replace(text, old, new):
    if old not in text:
        raise ValueError('Expected source text missing: '+old[:100])
    return text.replace(old, new)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def historical_seeds(root, current):
    root, current = Path(root), Path(current).resolve()
    seeds = set()
    for path in root.rglob('result.json'):
        if path.resolve().is_relative_to(current):
            continue
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            continue
        value = record.get('seed')
        if type(value) is int and value >= 0:
            seeds.add(value)
    return seeds


def main():
    cfg = json.loads((CP/'config.json').read_text())
    assert cfg['type'] == 'pi05' and cfg['chunk_size'] == 30
    assert cfg['output_features']['action']['shape'] == [7]
    assert json.loads((TRAIN/'final_checkpoint_reload_verified.json').read_text())['passed']
    selected = dict(checkpoint=str(CP),optimizer_updates=30000,microsteps=240000,
                    selection='final model explicitly discussed with user',heldout_loss=.0327,
                    adapter_sha256=sha(CP/'adapter_model.safetensors'))
    (R/'selected_checkpoint.json').write_text(json.dumps(selected,indent=2))
    seeds = set()
    for row in json.loads((TRAIN/'split_manifest.json').read_text())['source_records']:
        seeds.add(row['seed'])
    for path in Path('/home/lp531/smolvla_eval').glob('*/random_seeds.json'):
        value = json.loads(path.read_text())
        if isinstance(value, list):
            seeds.update(x for x in value if isinstance(x,int))
    seeds.update(historical_seeds('/home/lp531/smolvla_eval', R))
    baseline = sorted(seeds)
    for name in ['excluded_seed_baseline.json', 'excluded_seeds.json']:
        (R/name).write_text(json.dumps(baseline,indent=2))
    source = PACKAGE/'smolvla_sim_rollout.py'
    original = source.read_text()
    runtime = replace(original,
        'if args.policy_type == "pi05" and args.objects != ["beverage_bottle"]:\n        parser.error("This PI05 checkpoint is trained for bottle-only evaluation")',
        'if args.policy_type == "pi05" and args.objects != ["cup"]:\n        parser.error("This scoped PI05 checkpoint is trained for cup evaluation")')
    (R/'runtime_rollout.py').write_text(runtime)
    (R/'runtime_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),runtime.splitlines(True),fromfile=str(source),tofile='runtime_rollout.py')))
    shutil.copyfile(OLD/'workflow_core.py', R/'workflow_core.py')
    runner = (OLD/'run_evaluations.py').read_text()
    runner = replace(runner, '"""Thirty paired cup ACT trials at 8Hz using the previous 10Hz evaluation seeds."""',
                     '"""Thirty fresh cup PI05 trials at10Hz, executing12 steps per prediction."""')
    runner = replace(runner, "RUN=Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_chunk30_batch64_20261028_003937')", f'RUN=Path({str(TRAIN)!r})')
    runner = replace(runner, "CHECKPOINT=Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_chunk30_batch64_20261028_003937/output/checkpoints/070000/pretrained_model')", f'CHECKPOINT=Path({str(CP)!r})')
    runner = replace(runner, "act_sim_eval_*/episodes/*/result.json", "pi05_sim_eval_*/episodes/*/result.json")
    runner = replace(runner, ' return seeds\n', " seeds.update(json.loads((R/'excluded_seed_baseline.json').read_text()))\n return seeds\n")
    start = runner.index(" paired=json.loads((R/'paired_seeds.json').read_text())")
    end = runner.index(" save('excluded_seeds.json'", start)
    runner = runner[:start]+runner[end:]
    runner = replace(runner, "execute_steps=15", "execute_steps=12")
    runner = replace(runner, "action_rate_hz=8", "action_rate_hz=10")
    runner = replace(runner, "execution_rate_hz=64", "execution_rate_hz=60")
    runner = replace(runner, "seed_mode='paired: exact seeds and order of previous 10Hz30 trials; excluding all training seeds'", "seed_mode='30 fresh randomizations, excluding all513 source and previous evaluation seeds; resample only before trial for coverage/bounds'")
    runner = replace(runner, "(CHECKPOINT/'model.safetensors')", "(CHECKPOINT/'adapter_model.safetensors')")
    runner = replace(runner, "checkpoint_selection='best heldout validation checkpoint:70000 updates,loss0.0878; trained chunk30 batch64'", "checkpoint_selection='final PI05 LoRA:30000 optimizer updates,240000 microsteps,heldout loss0.0327'")
    runner = replace(runner, "n_action_steps=15", "n_action_steps=12")
    runner = replace(runner, "'--policy-type','act'", "'--policy-type','pi05'")
    runner = replace(runner, "'--execute-steps','15','--rate','8','--execution-rate','64'", "'--execute-steps','12','--rate','10','--execution-rate','60'")
    runner = replace(runner, "   if batch != 1:raise RuntimeError('Paired trial invalid: do not replace with different seeds')\n   cmd += ['--eval-seeds',*[str(seed) for seed in paired]]\n", '')
    runner = replace(runner, "d['execute_steps']==15", "d['execute_steps']==12")
    runner = replace(runner, "d['action_rate_hz']==8 and d['execution_rate_hz']==64", "d['action_rate_hz']==10 and d['execution_rate_hz']==60")
    runner = replace(runner, '  assert all_seeds==paired\n', '')
    (R/'run_evaluations.py').write_text(runner)
    supervisor = (OLD/'supervise.py').read_text().replace('PORT=8812', 'PORT=8816').replace('steps=[15]', 'steps=[12]')
    (R/'supervise.py').write_text(supervisor)
    report = (OLD/'summarize.py').read_text()
    start = report.index('    previous=json.loads(')
    end = report.index('    rows = []', start)
    report = report[:start]+report[end:]
    report = replace(report,"record['execute_steps'] == 15", "record['execute_steps'] == 12")
    report = replace(report, '# ACT500杯子模型：30次独立验证', '# PI0.5杯子LoRA最终模型：执行12步、30次独立验证')
    start = report.index('使用70000步验证集最佳模型')
    end = report.index('。",',start)
    report = report[:start]+'使用最终30000次优化器更新模型（训练chunk30），动作频率10Hz，底层指令60Hz，每次执行12步后重新预测；30个新随机种子排除全部513条来源及此前评估种子'+report[end:]
    (R/'summarize.py').write_text(report)
    for file in R.glob('*.py'):
        ast.parse(file.read_text(), filename=str(file))
    paths = [REPO/'isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd',
        REPO/'isaac_assets/z1/scripts/run_z1_ros2_scene.py', REPO/'isaac_assets/z1/scripts/render_occlusion.py',
        PACKAGE/'smolvla_sim_rollout.py', PACKAGE/'pi05_policy_bridge.py',
        Path('/home/lp531/z1_queue_logs/cup_policy_push_recovery20_20261028_134320/grasp_physics_minus90.json'),
        TRAIN/'split_manifest.json', R/'selected_checkpoint.json',
        *CP.glob('*.json'), *CP.glob('*.safetensors'),
        *[p for p in R.glob('*.py') if p.name != 'prepare_runtime.py']]
    (R/'source_snapshot.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2))
    print(json.dumps(dict(checkpoint=str(CP),execute_steps=12,total=30,excluded_seeds=len(seeds))))


if __name__ == '__main__':
    main()
