"""Use the ACT training environment and saved processors behind the ROS process protocol."""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

if __package__:
    from .pi05_policy_bridge import PI05ProcessPolicy, decode_batch
else:
    from pi05_policy_bridge import PI05ProcessPolicy, decode_batch


class ACTProcessPolicy(PI05ProcessPolicy):
    policy_name = 'ACT'

    def __init__(self, checkpoint):
        super().__init__(Path(checkpoint), worker_script=Path(__file__).resolve(),
                         python=os.environ.get('Z1_ACT_PYTHON',
                                               '/home/lp531/pi05_lora_workspace/.venv/bin/python'))


def worker(checkpoint):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    with redirect_stdout(sys.stderr):
        config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
        if config.type != 'act':
            raise ValueError('ACT worker requires an ACT checkpoint')
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.pretrained_backbone_weights = None
        policy = ACTPolicy.from_pretrained(str(checkpoint), config=config,
                                           local_files_only=True, strict=True)
        policy.eval()
        pre, post = make_pre_post_processors(
            config, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': config.device}})
    print(json.dumps({'status': 'ready', 'device': config.device}), flush=True)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            with redirect_stdout(sys.stderr):
                if message['command'] == 'reset':
                    policy.reset()
                    reply = {'status': 'reset'}
                elif message['command'] == 'predict':
                    with torch.inference_mode():
                        actions = post(policy.predict_action_chunk(pre(decode_batch(message['batch']))))
                    reply = {'actions': actions.detach().float().cpu().tolist()}
                else:
                    raise ValueError('Unknown ACT worker command')
            print(json.dumps(reply, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        raise SystemExit('Usage: act_policy_bridge.py --worker CHECKPOINT')
