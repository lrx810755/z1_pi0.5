"""Fixed 40 training observations; preserve training RNG and mode during inference."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

RUN = Path(__file__).resolve().parent
sys.path.insert(0, '/home/lp531/smolvla_eval/pi05_action_contract_audit_20261007')
from geometry import audit, height

def check_actions(*, checkpoint_dir, step, policy, preprocessor, postprocessor, accelerator, **kwargs):
    model = accelerator.unwrap_model(policy)
    was_training = model.training
    records = []
    model.eval()
    try:
        with torch.random.fork_rng(), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for label in json.loads((RUN/'selected.json').read_text()):
                root = Path(label['raw_root'])
                rows = [json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
                c, o = label['close_frame'], label['open_frame']
                points = [('home',0),('motion_start',label['first_motion_frame']),('approach25',c//4),('approach50',c//2),('approach75',3*c//4),('preclose',c-1),('close',c),('lift',c+20),('prerelease',o-1),('release',o+5)]
                for stage, i in points:
                    row = rows[i]
                    batch = {'observation.state':torch.tensor([row['observation.state']],dtype=torch.float32), 'task':[row['task']]}
                    for cam in ('wrist','global','global_right'):
                        key = 'observation.images.'+cam
                        array = np.array(Image.open(root/row[key]).convert('RGB'))
                        batch[key] = torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255
                    ref = np.array([x['action'] for x in rows[i:i+15]])
                    ref[:,-1] = [float(c<=j<o) for j in range(i,i+15)]
                    torch.manual_seed(123)
                    model.reset()
                    pred = postprocessor(model.predict_action_chunk(preprocessor(batch)))[0].float().cpu().numpy()
                    assert pred.shape == (15,7) and np.isfinite(pred).all()
                    poses = [(audit.fk(a[:6]), audit.fk(b[:6])) for a,b in zip(pred,ref)]
                    mapped = pred.copy()
                    mapped[:,-1] = np.where(pred[:,-1]>=.5,0.,-.9)
                    records.append(dict(episode=label['raw_episode'],stage=stage,frame=i,
                        prediction=pred.tolist(),reference=ref.tolist(),
                        position_error_mm=[float(np.linalg.norm(a[0]-b[0])*1000) for a,b in poses],
                        rotation_error_deg=[float(np.rad2deg(Rotation.from_matrix(a[1]@b[1].T).magnitude())) for a,b in poses],
                        joint_mae_rad=float(np.abs(pred[:,:6]-ref[:,:6]).mean()),
                        gripper_correct=((pred[:,-1]>=.5)==(ref[:,-1]>=.5)).tolist(),
                        first_close_frame=next((j+1 for j,x in enumerate(pred) if x[-1]>=.5),None),
                        raw_limit_change_rad=np.max(np.abs(mapped[:,:6]-np.clip(mapped[:,:6],audit.limits[:6,0],audit.limits[:6,1])),axis=1).tolist(),
                        raw_height_m=[height(x) for x in mapped]))
    finally:
        model.reset()
        model.train(was_training)
    assert len(records) == 40
    output = Path(checkpoint_dir)/'action_check.json'
    output.write_text(json.dumps(records,indent=2))
    summary = dict(microstep=step,optimizer_updates=step//8,observations=len(records),checkpoint=str(checkpoint_dir),stages={})
    for stage in dict.fromkeys(r['stage'] for r in records):
        group = [r for r in records if r['stage']==stage]
        summary['stages'][stage] = dict(first_position_mean_mm=float(np.mean([r['position_error_mm'][0] for r in group])),
            first_position_max_mm=max(r['position_error_mm'][0] for r in group),
            first_rotation_mean_deg=float(np.mean([r['rotation_error_deg'][0] for r in group])),
            gripper_accuracy=float(np.mean([r['gripper_correct'] for r in group])))
    with (RUN/'action_checks.jsonl').open('a') as f:
        f.write(json.dumps(summary)+'\n')
    print('ACTION_CHECK',json.dumps(summary),flush=True)
