import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
results = json.loads((ROOT / 'results.json').read_text()) if (ROOT / 'results.json').exists() else []
summary = []
for result in results:
    stage, repeat = result['stage'], result['repetition']
    folder = ROOT / stage / f'trial_{repeat:02d}'
    trace_path = next(folder.glob('episodes/*/trace.jsonl'))
    rows = [json.loads(s) for s in trace_path.read_text().splitlines()]
    predictions_path = folder / 'full_predictions.jsonl'
    predictions = [json.loads(s) for s in predictions_path.read_text().splitlines()] if predictions_path.exists() else []
    first_close = next((r['simulation_elapsed_s'] for r in rows if r['command'][-1] >= -.75), None)
    offsets = []
    for prediction in predictions:
        jaw = np.asarray(prediction['actions'])[0, :, -1]
        offsets.append(next((i for i, value in enumerate(jaw) if value >= -.75), None))
    inside = 0
    envelope = result['final_status'].get('grasp_envelope', {})
    for row in rows:
        p = row['contact_status'].get('relative_position_gripper_m')
        if p and envelope:
            inside += int(envelope['x_min_m'] <= p[0] <= envelope['x_max_m'] and abs(p[1]) <= envelope['y_abs_max_m'] and envelope['z_min_m'] <= p[2] <= envelope['z_max_m'])
    evidence = result['physical_predicates']
    summary.append({'stage': stage, 'repeat': repeat, 'success': result['success'],
        'stop_reason': result['stop_reason'] or result['failure_category'],
        'first_close_sim_s': first_close, 'physical_grasp': evidence.get('physical_grasp_ok', False),
        'bilateral_contact': evidence.get('bilateral_contact', False),
        'inside_envelope_frames': inside, 'trace_frames': len(rows),
        'prediction_chunks': len(predictions), 'chunks_close_in_first5': sum(o is not None and o < 5 for o in offsets),
        'chunks_close_only_later': sum(o is not None and o >= 5 for o in offsets),
        'chunks_no_close': offsets.count(None), 'predicted_close_offsets': offsets,
        'command_jaw_range': [min(r['command'][-1] for r in rows), max(r['command'][-1] for r in rows)] if rows else None,
        'handoff': result.get('handoff'), 'trace': str(trace_path)})
(ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
for stage in ['reference','home','midway','preclose']:
    group = [r for r in summary if r['stage'] == stage]
    print(stage, 'trials', len(group), 'success', sum(r['success'] for r in group),
          'close', sum(r['first_close_sim_s'] is not None for r in group),
          'physical', sum(r['physical_grasp'] for r in group),
          'reasons', [r['stop_reason'] for r in group],flush=True)
