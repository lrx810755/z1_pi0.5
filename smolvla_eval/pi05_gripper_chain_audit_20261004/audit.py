"""Read recorded predictions, commands and feedback; never publish ROS commands."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'pi05_staged_takeover_20261004'
CORE = Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/smolvla_rollout_core.py')
spec = importlib.util.spec_from_file_location('core', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

summaries, aligned = [], []
for path in sorted(SOURCE.glob('*/*/episodes/*/trace.jsonl')):
    rows = read(path)
    stage, trial = path.relative_to(SOURCE).parts[:2]
    predictions_path = path.parents[2] / 'full_predictions.jsonl'
    predictions = read(predictions_path) if predictions_path.exists() else []
    lookup = {}
    for prediction in predictions:
        for index, action in enumerate(prediction['actions'][0]):
            lookup.setdefault(tuple(action), []).append((prediction['prediction'], index))
    for i, row in enumerate(rows):
        raw, command, measured = (row[key][-1] for key in ['policy_action', 'command', 'joint_state'])
        matches = lookup.get(tuple(row['policy_action']), [])
        if stage != 'reference':
            assert matches and any(index == row['policy_action_index'] for _, index in matches)
            assert 0 <= row['policy_action_index'] < 5
        previous = rows[i-1]['command'][-1] if i else None
        aligned.append({'stage': stage, 'trial': trial, 'row': i,
                        'simulation_elapsed_s': row['simulation_elapsed_s'],
                        'action_index': row['policy_action_index'], 'substep': row['execution_substep'],
                        'raw_gripper': raw, 'published_gripper': command, 'measured_before_publish': measured,
                        'previous_published_gripper': previous,
                        'feedback_minus_previous_command': measured-previous if previous is not None else None})
    def event(predicate):
        for i, row in enumerate(rows):
            if predicate(i, row):
                return {'row': i, 't': row['simulation_elapsed_s'], 'raw': row['policy_action'][-1],
                        'command': row['command'][-1], 'measured_before_publish': row['joint_state'][-1],
                        'previous_command': rows[i-1]['command'][-1] if i else None,
                        'contacts': row['contact_status']}
        return None
    summaries.append({'stage': stage, 'trial': trial, 'trace_path': str(path),
                      'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(rows),
                      'raw_range': [min(r['policy_action'][-1] for r in rows), max(r['policy_action'][-1] for r in rows)],
                      'command_range': [min(r['command'][-1] for r in rows), max(r['command'][-1] for r in rows)],
                      'feedback_range': [min(r['joint_state'][-1] for r in rows), max(r['joint_state'][-1] for r in rows)],
                      'first_feedback_error_gt_005': event(lambda i,r: i > 0 and abs(r['joint_state'][-1]-rows[i-1]['command'][-1]) > .05),
                      'first_command_raw_error_gt_005': event(lambda i,r: abs(r['command'][-1]-r['policy_action'][-1]) > .05),
                      'first_feedback_below_train_min': event(lambda i,r: r['joint_state'][-1] < -1.1234),
                      'min_feedback_event': event(lambda i,r: r['joint_state'][-1] == min(x['joint_state'][-1] for x in rows)),
                      'negative_command_rewrites': sum(r['command'][-1] < r['policy_action'][-1]-.05 for r in rows),
                      'rewrites_equal_feedback_plus_01': sum(r['command'][-1] < r['policy_action'][-1]-.05 and abs(r['command'][-1]-(r['joint_state'][-1]+.1)) < 1e-8 for r in rows)})

model = [s for s in summaries if s['stage'] != 'reference']
meta = {'scope': '15 recorded model rollouts and 2 reference rollouts; no new simulation execution',
        'note': 'joint_state is sampled before publishing the command on the same trace row; use previous command to assess feedback error. Contact statuses are asynchronous.',
        'model_rollouts': len(model), 'model_trace_rows': sum(s['rows'] for s in model),
        'executed_raw_gripper_range': [min(s['raw_range'][0] for s in model), max(s['raw_range'][1] for s in model)],
        'feedback_below_train_min_rollouts': sum(s['first_feedback_below_train_min'] is not None for s in model),
        'negative_command_rewrites': sum(s['negative_command_rewrites'] for s in model),
        'rewrites_equal_feedback_plus_01': sum(s['rewrites_equal_feedback_plus_01'] for s in model),
        'source_core_sha256': hashlib.sha256(CORE.read_bytes()).hexdigest()}

# Reproduce the observed limiter result using the actual shared helper.
examples = []
for s in model:
    rows = read(Path(s['trace_path']))
    row = next((r for r in rows if r['command'][-1] < r['policy_action'][-1]-.05 and abs(r['command'][-1]-r['joint_state'][-1]-.1) < 1e-8), None)
    if row:
        limited = core.safe_joint_target(row['policy_action'], row['joint_state'], .1)[-1]
        assert np.isclose(limited, row['command'][-1], atol=1e-8)
        examples.append({'stage': s['stage'], 'trial': s['trial'], 't': row['simulation_elapsed_s'],
                         'raw': row['policy_action'][-1], 'measured': row['joint_state'][-1],
                         'published': row['command'][-1], 'offline_limiter_result': float(limited)})
(ROOT/'summary.json').write_text(json.dumps({'summary': meta, 'trials': summaries, 'limiter_reproductions': examples}, indent=2))
with (ROOT/'aligned_commands_feedback.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(aligned[0])); writer.writeheader(); writer.writerows(aligned)

worst = min(model, key=lambda s:s['feedback_range'][0])
fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
for ax, s in zip(axes, [next(s for s in model if s['stage']=='preclose' and s['trial']=='trial_01'), worst]):
    rows = read(Path(s['trace_path'])); t = [r['simulation_elapsed_s'] for r in rows]
    for key,label in [('policy_action','Model requested'),('command','Published command'),('joint_state','Measured before command')]:
        ax.plot(t, [r[key][-1] for r in rows], label=label, linewidth=1.3)
    ax.axhline(-1.1234, color='gray', linestyle='--', label='Training state minimum')
    ax.set(title=s['stage']+'/'+s['trial'], xlabel='Simulation seconds after handoff', ylabel='Gripper joint (rad)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.savefig(ROOT/'gripper_timeline.png', dpi=160)
print(json.dumps(meta, indent=2));print('Worst trial:',worst['stage'],worst['trial']);print('First limiter reproduction:',examples[0])
