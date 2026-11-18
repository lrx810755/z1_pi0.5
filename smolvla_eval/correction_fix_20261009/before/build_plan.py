"""Estimate curriculum magnitudes from actual failed rollouts, not joint nudges."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from core import atomic_json,digest,make_tasks
from geometry import Geometry

def build(destination):
    geo=Geometry();root=Path('/home/lp531/smolvla_eval/pi05_small4_autonomous_20261008')
    labels={Path(x['raw_root']).name:x for x in json.loads(Path('/home/lp531/pi05_lora_workspace/binary_gripper_20261007/episode_labels.json').read_text())}
    samples=[]
    for path in sorted(root.glob('repeat*/episode_*/episodes/*/result.json')):
        result=json.loads(path.read_text())
        if result['success']:continue
        episode=path.parts[-4];label=labels[episode]
        reference=[json.loads(x) for x in (Path(label['raw_root'])/'frames.jsonl').read_text().splitlines()][label['close_frame']-1]
        refp,refr=geo.fk(reference['observation.state'][:6])
        trace=[json.loads(x) for x in (path.parent/'trace.jsonl').read_text().splitlines()]
        first=next((i for i,x in enumerate(trace) if x['contact_status'].get('left_finger_touching_table') or x['contact_status'].get('right_finger_touching_table')),len(trace)-1)
        cutoff=trace[first]['simulation_elapsed_s']-.3
        safe=[x for x in trace[:first+1] if x['simulation_elapsed_s']<=cutoff]
        if not safe:continue
        row=safe[-1];p,r=geo.fk(row['joint_state'][:6]);delta=p-refp
        rot=Rotation.from_matrix(refr.T@r).as_rotvec()*180/np.pi
        samples.append(dict(source=str(path.parent/'trace.jsonl'),episode=episode,simulation_elapsed_s=row['simulation_elapsed_s'],
                            offset_world_m=delta.tolist(),rotation_tool_vector_deg=rot.tolist(),finger_height_m=geo.height(row['joint_state'])))
    if len(samples)<4:raise RuntimeError('失败样本不足，拒绝使用猜测幅度')
    raw=np.quantile(np.abs([x['offset_world_m'] for x in samples]),.5,axis=0)
    angle=float(np.quantile([np.linalg.norm(x['rotation_tool_vector_deg']) for x in samples],.5))
    amplitudes=dict(xyz_m=np.clip(raw,[.008,.008,.006],[.030,.030,.018]).tolist(),tilt_deg=float(np.clip(angle,6,20)))
    plan=dict(version=1,amplitudes=amplitudes,raw_median_abs_xyz_m=raw.tolist(),raw_median_rotation_deg=angle,
              note='12 failed small4 rollouts, 0.3 s before contact/end vs same-scene demonstrated preclose. Magnitudes bounded for feasible synthetic recovery; directions mirrored, not exact replay.',
              source_samples=samples,tasks=make_tasks(amplitudes))
    plan['hash']=digest(plan);atomic_json(destination,plan);return plan

if __name__=='__main__':
    import sys
    p=build(Path(sys.argv[1]));print(json.dumps({'amplitudes':p['amplitudes'],'samples':len(p['source_samples']),'tasks':len(p['tasks'])},indent=2))
