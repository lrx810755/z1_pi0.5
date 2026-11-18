"""Explicit same-cell layout alternatives for otherwise unreachable source scenes."""
import copy

def selected_task(task, stage, round_index):
    choices=stage.get('layout_fallbacks',{}).get(task['id'],[])
    result=copy.deepcopy(task)
    if not choices:return result
    choice=choices[(round_index-1)%len(choices)]
    if choice['scene']['target_grid']!=task['grid']:
        raise ValueError('Fallback must preserve the source grid cell')
    result.update(original_seed=task['seed'],seed=choice['seed'],scene=choice['scene'],
        layout_reference=choice['path'],layout_reference_frames_sha256=choice['frames_sha256'])
    return result

def layout_is_allowed(task,stage,summary,proof):
    scene=summary['scene_randomization']
    if scene['target_grid']!=task['grid'] or proof.get('seed')!=scene['seed']:return False
    if scene['seed']==task['seed'] and not proof.get('layout_reference'):return True
    if proof.get('original_seed')!=task['seed']:return False
    return any(choice['seed']==scene['seed'] and choice['path']==proof.get('layout_reference')
        and choice['frames_sha256']==proof.get('layout_reference_frames_sha256')
        and choice['scene']['target_grid']==task['grid']
        for choice in stage.get('layout_fallbacks',{}).get(task['id'],[]))
