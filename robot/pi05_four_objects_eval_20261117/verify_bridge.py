"""Final model raw-input process-boundary inference, manually invoked on GPU."""
import json
from pathlib import Path

from contracts import HERE, MODEL_PYTHON, check_final, check_pins, checkpoint_fingerprints, file_identity, read, sha
from policy_bridge import PI05ProcessPolicy, validate_batch
from verify_cpu import raw_samples


def deployment_input(batch):
    return {key: value for key, value in batch.items() if key.startswith('observation.') or key == 'task'}


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; final deployment verification has not run')
    pins = check_pins()
    checkpoint = check_final(require_reload=True)
    base_file = Path(read(checkpoint / 'adapter_config.json')['base_model_name_or_path']) / 'model.safetensors'
    base_identity = file_identity(base_file)
    base_digest = sha(base_file)
    policy = PI05ProcessPolicy(checkpoint, python=MODEL_PYTHON)
    tests = []
    try:
        for source, record, frame, batch in raw_samples():
            validate_batch(batch)
            policy.reset()
            prediction = policy.predict_action_chunk(deployment_input(batch))
            if prediction.shape != (1, 30, 7) or not torch.isfinite(prediction).all():
                raise ValueError('Invalid final bridge prediction')
            tests.append(dict(source=source, task=record['task'], frame=frame, finite=True, shape=list(prediction.shape)))
    finally:
        policy.close()
    if file_identity(base_file) != base_identity:
        raise ValueError('Base weights changed during verification')
    report = dict(passed=True, checkpoint=str(checkpoint), pins=pins, tests=tests,
                  checkpoint_files=checkpoint_fingerprints(checkpoint),
                  base_sha256=base_digest, base_file_identity=base_identity)
    (HERE / 'final_bridge_verified.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
