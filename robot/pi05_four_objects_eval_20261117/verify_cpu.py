"""Verify deployment preprocessing against eight real frozen raw RGB inputs."""
import json
from pathlib import Path

from contracts import HERE, RUN, CAMERAS, TASKS, check_pins, sha


def raw_samples():
    import numpy as np
    import torch
    from PIL import Image
    records = json.loads((RUN / 'split_manifest.json').read_text())['source_records']
    with np.load(RUN / 'sampling_arrays.npz') as a:
        eligible, episodes = a['eligible_train'], a['episode']
    with np.load(RUN / 'lowdim.npz') as a:
        states, actions, frames = a['states'], a['actions'], a['frame']
    for obj in ['apple', 'peach', 'cup', 'bottle']:
        for pool in ['basic', 'correction']:
            source = obj + '_' + pool
            record = next(r for r in records if r['source'] == source)
            anchor = int(np.flatnonzero(eligible & (episodes == record['episode_index']))[0])
            frame = int(frames[anchor])
            with (Path(record['source_path']) / 'frames.jsonl').open() as stream:
                import itertools
                row = json.loads(next(itertools.islice(stream, frame, None)))
            batch = {'observation.state': torch.from_numpy(states[anchor].copy()).unsqueeze(0),
                     'action': torch.from_numpy(actions[anchor:anchor + 30].copy()).unsqueeze(0),
                     'task': [record['task']]}
            for camera in CAMERAS:
                key = 'observation.images.' + camera
                with Image.open(Path(record['source_path']) / row[key]) as im:
                    pixels = np.asarray(im.convert('RGB')).copy()
                batch[key] = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255
            yield source, record, frame, batch


def main():
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from policy_bridge import validate_batch, encode_batch, decode_batch
    from torchvision.transforms import functional as F
    from contracts import CROPS
    torch.set_num_threads(2)
    check_pins()
    # This is the frozen processor prepared for this training round; no intermediate model is loaded.
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    config = json.loads((RUN / 'resolved_train_config.json').read_text())['policy']
    config.pop('type', None)
    config['device'] = 'cpu'
    cfg = PI05Config(**config)
    cfg.device = 'cpu'
    pre, post = make_pre_post_processors(cfg, pretrained_path=RUN / 'processor_verification',
        preprocessor_overrides={'device_processor': {'device': 'cpu'}})
    tests = []
    for source, record, frame, batch in raw_samples():
        validate_batch(batch)
        transported = decode_batch(encode_batch(batch))
        processed = pre(transported)
        for camera in CAMERAS:
            key = 'observation.images.' + camera
            expected = F.resize(F.crop(batch[key], *CROPS[key]), [224, 224], antialias=True)
            if processed[key].shape != (1, 3, 224, 224) or not torch.allclose(processed[key], expected, atol=1e-6):
                raise ValueError('Raw deployment preprocessing differs from training letterbox: ' + key)
        restored = post(processed['action'])
        if restored.shape != (1, 30, 7) or not torch.allclose(restored, batch['action'], atol=2e-6):
            raise ValueError('Saved action quantiles do not round-trip')
        tests.append(dict(source=source, episode=record['source_path'], frame=frame,
                          task=record['task'], raw_rgb=True, rgb224=True, action_roundtrip=True))
    report = dict(passed=True, model_loaded=False, gpu_used=False, tests=tests,
                  preparation_sha256=sha(RUN / 'preparation_complete.json'))
    (HERE / 'cpu_verified.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
