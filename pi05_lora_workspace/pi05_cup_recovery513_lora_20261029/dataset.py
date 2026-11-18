"""Immutable old500 crop cache plus on-demand accepted recovery image decoding."""
import json
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from config import RUN
from dataset_prepare import OLD, CAMERAS, CROPS


class CupDataset(torch.utils.data.Dataset):
    def __init__(self, episodes):
        self.episodes = list(episodes)
        self.records = json.loads((RUN/'split_manifest.json').read_text())['source_records']
        with np.load(RUN/'lowdim.npz') as a:
            self.arrays = {k: a[k] for k in a.files}
        with np.load(RUN/'sampling_arrays.npz') as a:
            self.approved = a['eligible_train'] | a['eligible_validation']
        self.indices = np.flatnonzero(np.isin(self.arrays['episode'], self.episodes))
        self.absolute_to_relative_idx = {int(x): i for i, x in enumerate(self.indices)}
        self.num_frames = len(self.indices)
        self.num_episodes = len(self.episodes)
        self.root = RUN
        self.fps = 10
        features = {k: dict(dtype='float32', shape=[7], names=[f'joint{i}' for i in range(1, 7)]+['jointGripper'])
                    for k in ['action', 'observation.state']}
        for camera in CAMERAS:
            features[camera] = dict(dtype='image', shape=[360 if camera.endswith('wrist') else 480, 640, 3],
                                    names=['height', 'width', 'channels'])
        stats = {k: {n: np.array(v, dtype=np.float32) for n, v in s.items()}
                 for k, s in json.loads((RUN/'train_stats.json').read_text()).items()}
        self.meta = SimpleNamespace(features=features, stats=stats, camera_keys=CAMERAS,
            has_language_columns=False, fps=10, robot_type='unitree_z1',
            repo_id='lp531/z1_cup_recovery513_pi05',
            episodes={k: [r[k] for r in self.records] for k in ['dataset_from_index', 'dataset_to_index']})

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        absolute = int(self.indices[index])
        ep, frame = int(self.arrays['episode'][absolute]), int(self.arrays['frame'][absolute])
        record = self.records[ep]
        if not self.approved[absolute] or absolute+30 > record['dataset_to_index']:
            raise ValueError('Dataset access requires approved complete30-action expert window')
        result = {'observation.state': torch.from_numpy(self.arrays['states'][absolute].copy()),
                  'action': torch.from_numpy(self.arrays['actions'][absolute:absolute+30].copy()),
                  'action_is_pad': torch.zeros(30, dtype=torch.bool),
                  'episode_index': torch.tensor(ep), 'frame_index': torch.tensor(frame),
                  'index': torch.tensor(absolute), 'timestamp': torch.tensor(frame/10.),
                  'task': record['task'], 'task_index': torch.tensor(0)}
        if getattr(self, '_pid', None) != os.getpid():
            self._maps, self._rows, self._pid = OrderedDict(), OrderedDict(), os.getpid()
        for camera in CAMERAS:
            if ep < 500:
                pair = (ep, camera)
                if pair not in self._maps:
                    if len(self._maps) >= 24:
                        self._maps.popitem(last=False)
                    self._maps[pair] = np.load(OLD/'cache'/camera/f'episode_{ep:06d}.npy', mmap_mode='r')
                self._maps.move_to_end(pair)
                result[camera] = torch.from_numpy(self._maps[pair][frame].copy())
            else:
                from PIL import Image
                from torchvision.transforms import functional as F
                root = Path(record['source_path'])
                if ep not in self._rows:
                    if len(self._rows) >= 20:
                        self._rows.popitem(last=False)
                    self._rows[ep] = [json.loads(l) for l in (root/'frames.jsonl').read_text().splitlines()]
                with Image.open(root/self._rows[ep][frame][camera]) as image:
                    pixels = torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1).float()/255
                result[camera] = F.resize(F.crop(pixels, *CROPS[camera]), [224, 224], antialias=True).clamp(0, 1)
        return result
