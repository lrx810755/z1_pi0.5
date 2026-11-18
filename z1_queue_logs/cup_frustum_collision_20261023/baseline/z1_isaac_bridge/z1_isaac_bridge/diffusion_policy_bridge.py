"""Run Diffusion in the training Python environment; ROS sends two synchronized frames."""
from contextlib import redirect_stdout
import json,os,sys
from pathlib import Path
if __package__:
    from .pi05_policy_bridge import PI05ProcessPolicy,decode_batch
else:
    from pi05_policy_bridge import PI05ProcessPolicy,decode_batch

class DiffusionProcessPolicy(PI05ProcessPolicy):
    policy_name='Diffusion'
    def __init__(self,checkpoint):
        super().__init__(Path(checkpoint),worker_script=Path(__file__).resolve(),
                         python=os.environ.get('Z1_DIFFUSION_PYTHON','/home/lp531/pi05_lora_workspace/.venv/bin/python'))

def worker(checkpoint):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    torch.set_num_threads(4)
    with redirect_stdout(sys.stderr):
        config=PreTrainedConfig.from_pretrained(str(checkpoint),local_files_only=True)
        if config.type!='diffusion' or config.n_obs_steps!=2:raise ValueError('Expected two-observation Diffusion checkpoint')
        config.device='cuda' if torch.cuda.is_available() else 'cpu';config.pretrained_backbone_weights=None
        policy=DiffusionPolicy.from_pretrained(str(checkpoint),config=config,local_files_only=True,strict=True);policy.eval()
        pre,post=make_pre_post_processors(config,pretrained_path=str(checkpoint),preprocessor_overrides={'device_processor':{'device':config.device}})
    print(json.dumps({'status':'ready','device':config.device}),flush=True)
    for line in sys.stdin:
        try:
            message=json.loads(line)
            with redirect_stdout(sys.stderr):
                if message['command']=='reset':
                    torch.manual_seed(20261020);policy.reset();reply={'status':'reset'}
                elif message['command']=='predict':
                    batch=decode_batch(message['batch'])
                    if batch['observation.state'].shape!=(1,2,7):raise ValueError('Diffusion requires synchronized state [1,2,7]')
                    flat={k:v.flatten(0,1) for k,v in batch.items() if k.startswith('observation.')}
                    processed=pre(flat)
                    processed={k:v.reshape(1,2,*v.shape[1:]) for k,v in processed.items() if k.startswith('observation.')}
                    with torch.inference_mode():actions=post(policy.predict_action_chunk(processed))
                    if actions.shape!=(1,15,7) or not torch.isfinite(actions).all():raise ValueError('Invalid diffusion actions')
                    reply={'actions':actions.detach().float().cpu().tolist()}
                else:raise ValueError('Unknown Diffusion worker command')
            print(json.dumps(reply,allow_nan=False),flush=True)
        except Exception as exc:print(json.dumps({'error':f'{type(exc).__name__}: {exc}'}),flush=True)
if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(Path(sys.argv[2]))
    else:raise SystemExit('Usage: diffusion_policy_bridge.py --worker CHECKPOINT')
