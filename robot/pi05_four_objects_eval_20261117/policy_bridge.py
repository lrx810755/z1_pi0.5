"""Run PI05 in its Python 3.12 environment while ROS stays in Python 3.10."""
from __future__ import annotations

import atexit
import base64
from contextlib import redirect_stdout
import json
from io import StringIO
import os
from pathlib import Path
import select
import subprocess
import sys
from types import SimpleNamespace


def encode_batch(batch):
    import torch
    result = {}
    for key, value in batch.items():
        if key.startswith('observation.images.'):
            image = (value.detach().cpu().clamp(0, 1) * 255).round().to(dtype=torch.uint8)
            result[key] = {'shape': list(image.shape), 'data': base64.b64encode(image.numpy().tobytes()).decode('ascii')}
        elif key == 'observation.state':
            result[key] = value.detach().cpu().tolist()
        else:
            result[key] = value
    return result


def decode_batch(message):
    import numpy as np
    import torch
    batch = {}
    for key, value in message.items():
        if key.startswith('observation.images.'):
            array = np.frombuffer(base64.b64decode(value['data']), dtype=np.uint8).copy().reshape(value['shape'])
            batch[key] = torch.from_numpy(array).float() / 255
        elif key == 'observation.state':
            batch[key] = torch.tensor(value, dtype=torch.float32)
        else:
            batch[key] = value
    return batch


def load_base_policy(policy_class, path, config):
    # This pinned LeRobot loader catches errors and returns an uninitialized model.
    # Require its explicit strict-load success marker before accepting that model.
    output = StringIO()
    with redirect_stdout(output):
        policy = policy_class.from_pretrained(path, config=config, local_files_only=True, strict=True)
    log = output.getvalue()
    print(log, file=sys.stderr, end='')
    if 'All keys loaded successfully!' not in log:
        raise RuntimeError('PI05 base weights did not load successfully; refusing inference')
    return policy


def reset_policy(policy, seed=None):
    if seed is not None:
        import torch
        torch.manual_seed(int(seed))
    policy.reset()


class PI05ProcessPolicy:
    policy_name = 'PI05'

    def __init__(self, checkpoint: Path, *, worker_script=None, python=None):
        config = json.loads((checkpoint / 'config.json').read_text())
        config.update(device='cpu', use_amp=False)
        for key in ('input_features', 'output_features'):
            config[key] = {name: SimpleNamespace(**value) for name, value in config[key].items()}
        self.config = SimpleNamespace(**config)
        python = python or os.environ.get('Z1_PI05_PYTHON', '/home/lp531/pi05_lora_workspace/.venv/bin/python')
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')
        self.process = subprocess.Popen([python, '-u', str(worker_script or Path(__file__).resolve()), '--worker', str(checkpoint)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        atexit.register(self.close)
        ready = self._read(180)
        if ready.get('status') != 'ready':
            self.close()
            raise RuntimeError(f'{self.policy_name} failed to load: {ready}')

    def _read(self, timeout=60):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            self.close()
            raise TimeoutError(f'{self.policy_name} inference process timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'{self.policy_name} inference process exited: {self.process.poll()}')
        message = json.loads(line)
        if 'error' in message:
            raise RuntimeError(message['error'])
        return message

    def _request(self, message):
        self.process.stdin.write(json.dumps(message, allow_nan=False) + '\n')
        self.process.stdin.flush()
        return self._read()

    def eval(self):
        return self

    def reset(self):
        self._request({'command': 'reset'})

    def predict_action_chunk(self, batch):
        import torch
        reply = self._request({'command': 'predict', 'batch': encode_batch(batch)})
        actions = torch.tensor(reply['actions'], dtype=torch.float32)
        if tuple(actions.shape) != (1, 30, 7) or not torch.isfinite(actions).all():
            raise ValueError(f'{self.policy_name} returned invalid joint actions')
        return actions

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def worker(checkpoint):
    import torch
    from peft import PeftModel
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    with redirect_stdout(sys.stderr):
        config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing silent CPU model inference')
        config.device = 'cuda'
        adapter = json.loads((checkpoint / 'adapter_config.json').read_text())
        if not (Path(adapter['base_model_name_or_path']) / 'model.safetensors').is_file():
            raise FileNotFoundError('PI05 base weights are missing')
        policy = load_base_policy(PI05Policy, adapter['base_model_name_or_path'], config)
        policy = PeftModel.from_pretrained(policy, str(checkpoint), is_trainable=False, local_files_only=True)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': config.device}})
    print(json.dumps({'status': 'ready', 'device': config.device}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        try:
            with redirect_stdout(sys.stderr):
                if message['command'] == 'reset':
                    reset_policy(policy, os.environ.get('Z1_PI05_INFERENCE_SEED'))
                    reply = {'status': 'reset'}
                else:
                    raw_batch = decode_batch(message['batch'])
                    validate_batch(raw_batch)
                    batch = preprocessor(raw_batch)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.device == 'cuda'):
                        actions = policy.predict_action_chunk(batch)
                        actions = postprocessor(actions)
                    reply = {'actions': actions.detach().float().cpu().tolist()}
            print(json.dumps(reply, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)



def validate_batch(batch):
    import torch
    from contracts import TASKS
    if tuple(batch['observation.state'].shape) != (1, 7) or not torch.isfinite(batch['observation.state']).all():
        raise ValueError('Expected seven finite continuous joint states')
    if batch.get('task') not in [[task] for task in TASKS.values()]:
        raise ValueError('Task text differs from frozen four-object tasks')
    for camera, height in [('wrist', 360), ('global', 480), ('global_right', 480)]:
        value = batch['observation.images.' + camera]
        if tuple(value.shape) != (1, 3, height, 640) or not torch.isfinite(value).all():
            raise ValueError('Wrong raw camera shape/value: ' + camera)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        raise SystemExit('Usage: pi05_policy_bridge.py --worker CHECKPOINT')
