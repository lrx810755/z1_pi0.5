"""Persistent one-shot workflow. No monitoring loop or CPU fallback."""
import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys

import torch

from config import RUN, OUTPUT, BASE
from dataset_prepare import write


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: current environment cannot access NVIDIA GPU; no training started')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA device does not support approved BF16 training')


def status(stage, **details):
    write('workflow_status.json', dict(stage=stage,
          updated_at=datetime.datetime.now().astimezone().isoformat(), **details))


def preflight():
    evidence = dict(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                    nvidia_devices=__import__('glob').glob('/dev/nvidia*'),
                    base_weights_exist=(BASE/'model.safetensors').is_file(),
                    output_exists=OUTPUT.exists(), training_started=False)
    try:
        require_cuda()
        evidence.update(gpu=torch.cuda.get_device_name(),
                        total_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
        if not evidence['base_weights_exist']:
            raise RuntimeError('Official base weights missing')
        if OUTPUT.exists():
            raise RuntimeError('Output already exists; refusing to overwrite or duplicate experiment')
        evidence['passed'] = True
        write('launch_preflight.json', evidence)
    except Exception as exc:
        evidence.update(passed=False, error=str(exc))
        write('launch_preflight.json', evidence)
        status('blocked', error=str(exc), training_started=False)
        raise


def command(stage, script, *args):
    status(stage, training_started=stage == 'training')
    with (RUN/(stage+'.log')).open('a') as log:
        subprocess.run([sys.executable, '-u', str(RUN/script), *args],
                        cwd=RUN, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    with (RUN/'.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight()
        try:
            # Preparation may have been done in an earlier session; verify immutable artifacts again.
            command('verifying_data_and_config', 'verify.py')
            if not (RUN/'smoke_completed.json').exists():
                command('smoke', 'train.py', '--smoke')
            command('verifying_smoke_checkpoint', 'verify.py', '--reload-smoke')
            command('training', 'train.py', '--train')
            command('verifying_final_checkpoint', 'verify.py', '--reload-final')
            complete = json.loads((RUN/'completed.json').read_text())
            status('completed', **complete)
        except Exception as exc:
            status('failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()
