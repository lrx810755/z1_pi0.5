"""Read-only evidence that the detached experiment is actually training."""
import datetime
import hashlib
import json
import math
import subprocess
from pathlib import Path

RUN = Path(__file__).resolve().parent
SERVICE = 'pi05-grip-stage-base-20261011.service'


def verify():
    status = subprocess.check_output(['systemctl', '--user', 'show', SERVICE,
                  '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
                  '-p', 'ExecMainStartTimestamp'], text=True)
    service = dict(line.split('=', 1) for line in status.splitlines())
    assert service['ActiveState'] == 'active' and service['SubState'] == 'running', service
    pid = int(service['MainPID'])
    assert pid > 0 and Path(f'/proc/{pid}').is_dir()
    read = lambda name: json.loads((RUN / name).read_text())
    cfg = read('resolved_train_config.json')
    setup = read('training_setup_verification.json')
    init = read('initialization_verified.json')
    first = read('first_update_verified.json')
    live = read('live_training.json')
    sampling = read('sampling_audit.json')
    assert cfg['steps'] == 120000 and cfg['accelerator']['gradient_accumulation']['steps'] == 8
    assert not cfg['resume'] and cfg['batch_size'] == 1
    assert cfg['scheduler']['num_decay_steps'] == 120000 and cfg['scheduler']['num_warmup_steps'] == 4000
    assert setup['passed'] and setup['decoded_episode_boundaries'] == 192 and setup['all_checked_chunks_exact']
    assert init['fresh_adapter'] and init['zero_lora_b_matrices'] == 252 and init['full_projection_tensors_match_base']
    assert first['actual_optimizer_steps'] == [1] and first['changed_tensors'] > 0 and first['finite']
    assert live['microsteps'] >= 64 and live['optimizer_updates'] >= 8 and math.isfinite(live['loss'])
    assert live['actual_optimizer_steps'] == [live['optimizer_updates']]
    assert live['scheduler_microsteps'] == live['microsteps']
    assert abs(sampling['original_mass'] - .7) < 1e-10 and abs(sampling['correction_mass'] - .3) < 1e-10
    assert sampling['heldout_excluded']
    log = (RUN / 'training.log').read_text(errors='replace')
    assert 'Traceback (most recent call last)' not in log and 'CUDA out of memory' not in log
    evidence = dict(passed=True, verified_at=datetime.datetime.now().astimezone().isoformat(),
                    service=service, current_progress=live, first_update=first,
                    initialization={k: v for k, v in init.items() if k != 'trainable_names'},
                    sampling=sampling, config=str(RUN / 'resolved_train_config.json'),
                    output=cfg['output_dir'], source_sha256={name: hashlib.sha256((RUN/name).read_bytes()).hexdigest()
                    for name in ['phases.py','prepare.py','sampling.py','train.py','run_training.sh']})
    (RUN / 'launch_verified.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: evidence[k] for k in ['passed', 'verified_at', 'service', 'current_progress', 'first_update']}, indent=2))


if __name__ == '__main__':
    verify()
